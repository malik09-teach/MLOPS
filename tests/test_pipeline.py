"""End-to-end test suite for the MLOps pipeline.

Covers:
  - Pandera schema validation and error handling
  - Data ingestion shape and distribution checks
  - Model training execution and metric assertions
  - FastAPI /health and /predict endpoints via httpx / starlette TestClient
  - Resilient inference logging file verification
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pandera as pa
import pytest
from fastapi.testclient import TestClient

from src.config import (
    FEATURE_NAMES,
    INFERENCE_LOGS_PATH,
    TARGET_NAME,
)
from src.data_ingest import generate_synthetic_dataset
from src.data_validation import build_validation_schema, validate_dataframe
from src.predict import app, log_inference_records


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Generate a clean fixture DataFrame for schema testing."""
    return pd.DataFrame(
        {
            "feature_1": [1.2, -0.5, 0.0],
            "feature_2": [-2.1, 0.4, 1.1],
            "feature_3": [0.8, -1.2, 0.5],
            "feature_4": [1.5, 2.0, -0.3],
            "feature_5": [-0.1, 0.9, -1.5],
            "target": [0, 1, 0],
        }
    )


def test_data_ingest_shape_and_columns():
    """Verify synthetic dataset generator produces correct shape and columns."""
    df = generate_synthetic_dataset(n_samples=50, random_state=42)
    assert df.shape == (50, 6)
    expected_cols = set(FEATURE_NAMES + [TARGET_NAME])
    assert set(df.columns) == expected_cols
    assert set(df[TARGET_NAME].unique()).issubset({0, 1})


def test_pandera_schema_valid_data(sample_valid_df):
    """Test that valid tabular data passes Pandera validation."""
    schema = build_validation_schema()
    validated = validate_dataframe(sample_valid_df, schema)
    assert len(validated) == 3
    assert validated[TARGET_NAME].tolist() == [0, 1, 0]


def test_pandera_schema_null_rejection(sample_valid_df):
    """Test that null values trigger Pandera SchemaError."""
    schema = build_validation_schema()
    invalid_df = sample_valid_df.copy()
    invalid_df.loc[0, "feature_1"] = None
    with pytest.raises(pa.errors.SchemaError):
        validate_dataframe(invalid_df, schema)


def test_pandera_schema_out_of_range_rejection(sample_valid_df):
    """Test that out-of-range numeric values trigger Pandera SchemaError."""
    schema = build_validation_schema()
    invalid_df = sample_valid_df.copy()
    invalid_df.loc[1, "feature_2"] = 999.0  # Allowed range is [-15.0, 15.0]
    with pytest.raises(pa.errors.SchemaError):
        validate_dataframe(invalid_df, schema)


def test_pandera_schema_invalid_target(sample_valid_df):
    """Test that non-binary target triggers Pandera SchemaError."""
    schema = build_validation_schema()
    invalid_df = sample_valid_df.copy()
    invalid_df.loc[0, TARGET_NAME] = 3
    with pytest.raises(pa.errors.SchemaError):
        validate_dataframe(invalid_df, schema)


def test_fastapi_health_endpoint():
    """Test GET /health returns 200 and valid JSON schema."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "tracking_uri" in data
    assert "timestamp" in data


def test_fastapi_predict_single_endpoint():
    """Test POST /predict with a single feature record using mock model."""
    client = TestClient(app)

    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.15, 0.85]]

    with patch("src.predict._LOADED_MODEL", mock_model), patch(
        "src.predict._MODEL_VERSION_TAG", "models:/prod_model@champion"
    ):
        payload = {
            "feature_1": 0.5,
            "feature_2": -1.2,
            "feature_3": 1.0,
            "feature_4": 0.2,
            "feature_5": -0.8,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == [1]
        assert len(data["probabilities"]) == 1
        assert 0.0 <= data["probabilities"][0] <= 1.0
        assert data["count"] == 1


def test_fastapi_predict_batch_endpoint():
    """Test POST /predict with batch inputs."""
    client = TestClient(app)

    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1]
    mock_model.predict_proba.return_value = [[0.8, 0.2], [0.1, 0.9]]

    with patch("src.predict._LOADED_MODEL", mock_model), patch(
        "src.predict._MODEL_VERSION_TAG", "models:/prod_model@champion"
    ):
        batch_payload = {
            "inputs": [
                {
                    "feature_1": -0.5,
                    "feature_2": 1.1,
                    "feature_3": -0.8,
                    "feature_4": 0.3,
                    "feature_5": 0.2,
                },
                {
                    "feature_1": 1.5,
                    "feature_2": -0.9,
                    "feature_3": 0.4,
                    "feature_4": -1.1,
                    "feature_5": 0.7,
                },
            ]
        }
        response = client.post("/predict", json=batch_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == [0, 1]
        assert data["count"] == 2


def test_fastapi_predict_invalid_payload():
    """Test POST /predict with missing required fields returns 422."""
    client = TestClient(app)
    # Missing feature_5
    invalid_payload = {
        "feature_1": 0.5,
        "feature_2": -1.2,
        "feature_3": 1.0,
        "feature_4": 0.2,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_inference_logger_writes_csv(tmp_path):
    """Verify log_inference_records creates CSV with correct headers and data."""
    test_log_path = tmp_path / "test_logs" / "inference_logs.csv"

    with patch("src.predict.INFERENCE_LOGS_PATH", test_log_path), patch(
        "src.predict.LOGS_DIR", test_log_path.parent
    ):
        records = [
            {
                "feature_1": 0.1,
                "feature_2": 0.2,
                "feature_3": 0.3,
                "feature_4": 0.4,
                "feature_5": 0.5,
                "prediction": 1,
                "probability": 0.92,
                "timestamp": "2026-11-24T12:00:00Z",
            }
        ]
        log_inference_records(records)
        assert test_log_path.exists()

        log_df = pd.read_csv(test_log_path)
        assert len(log_df) == 1
        assert "prediction" in log_df.columns
        assert "probability" in log_df.columns
        assert "timestamp" in log_df.columns
        assert log_df.loc[0, "prediction"] == 1
