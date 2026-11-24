"""FastAPI production inference service with resilient inference logging.

Exposes:
  - GET  /health: health check and model loading status
  - POST /predict: payload validation, model scoring, and thread-safe logging
    to logs/inference_logs.csv for data drift monitoring.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union
import logging
import os
import threading

from fastapi import FastAPI, HTTPException, status
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import (
    FEATURE_NAMES,
    INFERENCE_LOGS_PATH,
    LOGS_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
    MODEL_STAGE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict_service")

# Thread lock for file logging
_LOG_LOCK = threading.Lock()

# Global model holder
_LOADED_MODEL: Any = None
_MODEL_VERSION_TAG: str = "unknown"


def load_production_model() -> tuple[Any, str]:
    """Load the Production / Champion model from MLflow Model Registry.

    Attempts alias lookup first (@champion, @Production), then stage fallback
    (/Production), and lastly searches for latest registered version.

    Returns:
        Tuple of (loaded_pyfunc_model, version_identifier).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # 1. Modern Model Registry Aliases: @champion or @Production
    candidates = [
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        f"models:/{MODEL_NAME}@Production",
        f"models:/{MODEL_NAME}/{MODEL_STAGE}",
    ]

    for model_uri in candidates:
        try:
            logger.info("Attempting to load model from URI: %s", model_uri)
            loaded = mlflow.pyfunc.load_model(model_uri)
            logger.info("Successfully loaded model from %s", model_uri)
            return loaded, model_uri
        except Exception as err:
            logger.debug("Failed loading model from %s: %s", model_uri, err)

    # 2. Fallback to latest registered version via MlflowClient
    try:
        latest_versions = client.get_latest_versions(MODEL_NAME)
        if latest_versions:
            latest_v = max(int(v.version) for v in latest_versions)
            version_uri = f"models:/{MODEL_NAME}/{latest_v}"
            logger.info("Fallback: Loading latest registered version via %s", version_uri)
            loaded = mlflow.pyfunc.load_model(version_uri)
            return loaded, version_uri
    except Exception as err:
        logger.warning("Failed latest version fallback for model '%s': %s", MODEL_NAME, err)

    raise RuntimeError(
        f"Could not load any Production model for '{MODEL_NAME}' from {MLFLOW_TRACKING_URI}."
    )


def log_inference_records(records: list[dict[str, Any]]) -> None:
    """Thread-safe resilient append to logs/inference_logs.csv.

    Automatically creates the logs directory and writes CSV headers if the
    file does not exist.

    Args:
        records: List of dictionaries matching logging schema.
    """
    if not records:
        return

    df = pd.DataFrame(records)

    with _LOG_LOCK:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_exists = INFERENCE_LOGS_PATH.exists() and INFERENCE_LOGS_PATH.stat().st_size > 0

        df.to_csv(
            INFERENCE_LOGS_PATH,
            mode="a",
            header=not file_exists,
            index=False,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager to preload model on service start."""
    global _LOADED_MODEL, _MODEL_VERSION_TAG
    try:
        _LOADED_MODEL, _MODEL_VERSION_TAG = load_production_model()
        logger.info("Production model initialized on startup: %s", _MODEL_VERSION_TAG)
    except Exception as exc:
        logger.warning(
            "Model preloading deferred (pipeline training may not have executed yet): %s",
            exc,
        )
    yield


app = FastAPI(
    title="MLOps Production Inference API",
    description="Production-grade FastAPI prediction service with MLflow registry and inference logging.",
    version="1.0.0",
    lifespan=lifespan,
)


class DataPoint(BaseModel):
    """Schema for individual feature input."""

    feature_1: float = Field(..., description="Numeric feature 1")
    feature_2: float = Field(..., description="Numeric feature 2")
    feature_3: float = Field(..., description="Numeric feature 3")
    feature_4: float = Field(..., description="Numeric feature 4")
    feature_5: float = Field(..., description="Numeric feature 5")


class BatchInferencePayload(BaseModel):
    """Batch payload schema."""

    inputs: list[DataPoint] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    """Schema for prediction response item."""

    prediction: int
    probability: float


class PredictResponse(BaseModel):
    """Overall API prediction response."""

    predictions: list[int]
    probabilities: list[float]
    model_version: str
    count: int


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, Any]:
    """Health check endpoint exposing runtime and model status."""
    global _LOADED_MODEL, _MODEL_VERSION_TAG
    is_loaded = _LOADED_MODEL is not None
    return {
        "status": "healthy",
        "model_loaded": is_loaded,
        "model_name": MODEL_NAME,
        "model_version": _MODEL_VERSION_TAG if is_loaded else None,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(payload: Union[BatchInferencePayload, DataPoint]) -> PredictResponse:
    """Run model inference on single or batch payloads and log request data."""
    global _LOADED_MODEL, _MODEL_VERSION_TAG

    # Ensure model is ready or attempt dynamic on-demand loading
    if _LOADED_MODEL is None:
        try:
            _LOADED_MODEL, _MODEL_VERSION_TAG = load_production_model()
        except Exception as exc:
            logger.error("Failed to load production model: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Production model '{MODEL_NAME}' is currently unavailable: {exc}",
            ) from exc

    # Normalize single DataPoint vs BatchInferencePayload
    if isinstance(payload, DataPoint):
        items = [payload]
    else:
        items = payload.inputs

    # Build input DataFrame
    input_records = [item.model_dump() for item in items]
    input_df = pd.DataFrame(input_records)[FEATURE_NAMES]

    try:
        raw_preds = _LOADED_MODEL.predict(input_df)
        predictions = [int(p) for p in np.asarray(raw_preds).ravel()]

        # Extract probability for positive class if classifier supports it
        probabilities: list[float] = []
        if hasattr(_LOADED_MODEL, "predict_proba"):
            probs = _LOADED_MODEL.predict_proba(input_df)
            probabilities = [float(p[1]) for p in probs]
        elif hasattr(_LOADED_MODEL, "unwrap_python_model") and hasattr(
            _LOADED_MODEL.unwrap_python_model(), "predict_proba"
        ):
            probs = _LOADED_MODEL.unwrap_python_model().predict_proba(input_df)
            probabilities = [float(p[1]) for p in probs]
        else:
            # Fallback for models without predict_proba
            probabilities = [float(p) for p in predictions]

    except Exception as exc:
        logger.error("Inference execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc

    # Format log records for drift monitoring
    current_time = datetime.now(timezone.utc).isoformat()
    log_records = []
    for rec, pred, prob in zip(input_records, predictions, probabilities):
        logged_item = dict(rec)
        logged_item["prediction"] = pred
        logged_item["probability"] = prob
        logged_item["timestamp"] = current_time
        log_records.append(logged_item)

    # Persist inference log
    try:
        log_inference_records(log_records)
    except Exception as log_err:
        logger.warning("Non-fatal: failed to write inference logs: %s", log_err)

    return PredictResponse(
        predictions=predictions,
        probabilities=probabilities,
        model_version=_MODEL_VERSION_TAG,
        count=len(predictions),
    )
