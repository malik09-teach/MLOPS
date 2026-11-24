"""Centralized configuration module for the MLOps pipeline.

Manages all directory paths, MLflow tracking configurations, model registry
metadata, and dataset feature specifications.
"""

from pathlib import Path
import os

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "dataset.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

# Logging and Reporting paths
LOGS_DIR = BASE_DIR / "logs"
INFERENCE_LOGS_PATH = LOGS_DIR / "inference_logs.csv"
REPORTS_DIR = BASE_DIR / "reports"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.html"

# MLflow Tracking and Registry configuration
MLRUNS_DIR = BASE_DIR / "mlruns"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

# Local SQLite backend and default local artifact location
DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-classification-experiment")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "prod_model")
MODEL_ALIAS = "champion"
MODEL_STAGE = "Production"

# Dataset feature definitions
FEATURE_NAMES = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
]
TARGET_NAME = "target"

# Dataset generation and split parameters
N_SAMPLES = 1000
N_FEATURES = 5
N_INFORMATIVE = 4
N_REDUNDANT = 1
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model training hyperparameters
N_ESTIMATORS = 100
MAX_DEPTH = 8
CRITERION = "gini"
MIN_SAMPLES_SPLIT = 2
