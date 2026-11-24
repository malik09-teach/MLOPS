"""Model training, evaluation, and MLflow Model Registry module.

Trains a RandomForestClassifier on processed training data, evaluates on
test data, logs metrics/params/artifacts to MLflow, registers the model as
'prod_model', and transitions it to the 'Production' alias/stage.
"""

from pathlib import Path
import json
import logging
import os
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    CRITERION,
    EXPERIMENT_NAME,
    FEATURE_NAMES,
    MAX_DEPTH,
    MIN_SAMPLES_SPLIT,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
    MODEL_STAGE,
    N_ESTIMATORS,
    RANDOM_STATE,
    TARGET_NAME,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def train_and_register_model() -> tuple[str, str]:
    """Train RandomForestClassifier, log to MLflow, and register Production model.

    Returns:
        Tuple of (run_id, model_version).
    """
    if not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed datasets missing. Run src/data_validation.py first."
        )

    logger.info("Loading training data from %s", TRAIN_DATA_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df[TARGET_NAME]
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df[TARGET_NAME]

    # Configure MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("MLflow tracking URI set to: %s", MLFLOW_TRACKING_URI)

    # Initialize / retrieve experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    params = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "criterion": CRITERION,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "random_state": RANDOM_STATE,
    }

    with mlflow.start_run(experiment_id=experiment_id, run_name="rf_classification_run") as run:
        run_id = run.info.run_id
        logger.info("Started MLflow run: %s", run_id)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }

        # Log params & metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        logger.info("Logged metrics: %s", metrics)

        # Save and log confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred).tolist()
        cm_artifact = {
            "confusion_matrix": cm,
            "classes": [0, 1],
            "metrics": metrics,
        }
        cm_path = Path("confusion_matrix.json")
        with open(cm_path, "w", encoding="utf-8") as f:
            json.dump(cm_artifact, f, indent=2)
        mlflow.log_artifact(str(cm_path))
        if cm_path.exists():
            cm_path.unlink()

        # Infer model signature and input example
        signature = infer_signature(X_train, y_train)
        input_example = X_train.head(3)

        # Log and register model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_NAME,
        )
        logger.info("Model logged to MLflow with URI: %s", model_info.model_uri)

    # Manage Model Registry stages and modern aliases
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Get latest version for this model
    latest_versions = client.get_latest_versions(MODEL_NAME)
    if not latest_versions:
        raise RuntimeError(f"No versions found in registry for model {MODEL_NAME}")

    latest_version = max(int(v.version) for v in latest_versions)
    version_str = str(latest_version)
    logger.info("Latest registered model version: %s", version_str)

    # Assign modern aliases: @champion and @Production
    for alias in [MODEL_ALIAS, "Production"]:
        try:
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias=alias,
                version=version_str,
            )
            logger.info("Assigned alias '@%s' to version %s", alias, version_str)
        except Exception as e:
            logger.warning("Could not set alias '@%s': %s", alias, e)

    # Maintain stage compatibility for legacy clients
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version_str,
            stage=MODEL_STAGE,
            archive_existing_versions=True,
        )
        logger.info(
            "Transitioned model '%s' version %s to stage '%s'",
            MODEL_NAME,
            version_str,
            MODEL_STAGE,
        )
    except Exception as e:
        logger.warning("Stage transition warning (expected on alias-only servers): %s", e)

    return run_id, version_str


def main() -> None:
    """Execute model training and registry workflow."""
    run_id, version = train_and_register_model()
    logger.info(
        "Model training and registry completed. Run ID: %s, Registered Version: %s",
        run_id,
        version,
    )


if __name__ == "__main__":
    main()
