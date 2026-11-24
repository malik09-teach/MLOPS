"""Data validation module using Pandera.

Validates schema types, null constraints, and value boundaries for raw data,
then splits validated data into train.csv and test.csv in data/processed/.
"""

from pathlib import Path
import logging
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_NAMES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_NAME,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data_validation")


def build_validation_schema() -> DataFrameSchema:
    """Construct Pandera DataFrameSchema for the binary classification dataset.

    Returns:
        pandera.DataFrameSchema validating types, nulls, and expected ranges.
    """
    columns: dict[str, Column] = {
        col: Column(
            pa.Float64,
            nullable=False,
            checks=[
                Check(
                    lambda s: s.notnull() & ~s.isin([float("inf"), float("-inf")]),
                    element_wise=False,
                    name=f"{col}_finite_check",
                    error=f"{col} contains null or non-finite values",
                ),
                Check.in_range(
                    min_value=-15.0,
                    max_value=15.0,
                    name=f"{col}_range_check",
                    error=f"{col} out of expected range [-15.0, 15.0]",
                ),
            ],
            coerce=True,
        )
        for col in FEATURE_NAMES
    }

    columns[TARGET_NAME] = Column(
        pa.Int64,
        nullable=False,
        checks=[
            Check.isin([0, 1], name="binary_target_check"),
        ],
        coerce=True,
    )

    return DataFrameSchema(columns=columns, strict=True, coerce=True)


def validate_dataframe(df: pd.DataFrame, schema: DataFrameSchema) -> pd.DataFrame:
    """Validate DataFrame against the specified Pandera schema.

    Args:
        df: Input DataFrame.
        schema: Configured Pandera DataFrameSchema.

    Returns:
        Validated (and coerced) DataFrame.

    Raises:
        pa.errors.SchemaError: If schema validation fails.
    """
    logger.info("Validating DataFrame with %d rows and %d columns", len(df), len(df.columns))
    validated_df = schema.validate(df, lazy=False)
    logger.info("Data validation succeeded with zero schema violations.")
    return validated_df


def split_and_save_data(
    df: pd.DataFrame,
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validated dataset into train and test sets and persist to disk.

    Args:
        df: Validated DataFrame.
        train_path: Destination path for train.csv.
        test_path: Destination path for test.csv.
        test_size: Ratio of test partition.
        random_state: Seed for reproducible split.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_NAME],
        random_state=random_state,
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Train dataset written to %s (shape: %s)", train_path, train_df.shape)
    logger.info("Test dataset written to %s (shape: %s)", test_path, test_df.shape)

    return train_df, test_df


def main() -> None:
    """Execute raw data validation and train/test generation."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATA_PATH}. Run src/data_ingest.py first."
        )

    raw_df = pd.read_csv(RAW_DATA_PATH)
    schema = build_validation_schema()
    validated_df = validate_dataframe(raw_df, schema)
    split_and_save_data(validated_df)


if __name__ == "__main__":
    main()
