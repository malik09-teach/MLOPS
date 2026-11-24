"""Synthetic data ingestion module.

Generates a tabular binary classification dataset with 1,000 samples and
5 numeric features, saving the raw dataset to data/raw/dataset.csv.
"""

from pathlib import Path
import logging
import pandas as pd
from sklearn.datasets import make_classification

from src.config import (
    FEATURE_NAMES,
    N_FEATURES,
    N_INFORMATIVE,
    N_REDUNDANT,
    N_SAMPLES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data_ingest")


def generate_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    n_informative: int = N_INFORMATIVE,
    n_redundant: int = N_REDUNDANT,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate synthetic tabular binary classification data.

    Args:
        n_samples: Total number of rows to generate.
        n_features: Total number of numeric features.
        n_informative: Informative features count.
        n_redundant: Redundant features count.
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame containing feature columns and binary target.
    """
    logger.info(
        "Generating synthetic dataset: %d samples, %d features",
        n_samples,
        n_features,
    )
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=2,
        weights=[0.55, 0.45],
        flip_y=0.02,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df[TARGET_NAME] = y.astype(int)
    logger.info(
        "Dataset generated successfully. Class distribution: %s",
        df[TARGET_NAME].value_counts().to_dict(),
    )
    return df


def save_raw_dataset(df: pd.DataFrame, output_path: Path = RAW_DATA_PATH) -> Path:
    """Persist generated dataset to disk.

    Args:
        df: DataFrame to save.
        output_path: Target CSV file path.

    Returns:
        Path to the saved CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Raw dataset saved to: %s (shape: %s)", output_path, df.shape)
    return output_path


def main() -> None:
    """Execute raw data generation pipeline."""
    df = generate_synthetic_dataset()
    save_raw_dataset(df)


if __name__ == "__main__":
    main()
