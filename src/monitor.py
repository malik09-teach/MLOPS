"""Data drift and model monitoring module using Evidently AI.

Compares reference training data (data/processed/train.csv) against live
inference logs (logs/inference_logs.csv) and generates an HTML drift report
saved to reports/drift_report.html.
"""

from pathlib import Path
import logging
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
import pandas as pd

from src.config import (
    DRIFT_REPORT_PATH,
    FEATURE_NAMES,
    INFERENCE_LOGS_PATH,
    REPORTS_DIR,
    TRAIN_DATA_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monitor")


def generate_drift_report(
    reference_path: Path = TRAIN_DATA_PATH,
    current_path: Path = INFERENCE_LOGS_PATH,
    output_report_path: Path = DRIFT_REPORT_PATH,
) -> Path:
    """Generate Evidently HTML data drift and quality report.

    Args:
        reference_path: Path to baseline training dataset.
        current_path: Path to live inference logs CSV.
        output_report_path: Destination path for HTML report.

    Returns:
        Path to the saved drift report HTML file.
    """
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {reference_path}. "
            "Run src/data_validation.py before running monitoring."
        )

    logger.info("Loading reference training data from %s", reference_path)
    ref_df = pd.read_csv(reference_path)

    if not current_path.exists() or current_path.stat().st_size == 0:
        logger.warning(
            "Inference logs not found or empty at %s. "
            "Simulating a batch of current inference records for report generation.",
            current_path,
        )
        # Sample records from reference with slight Gaussian noise to demonstrate drift detection
        simulated_current = ref_df[FEATURE_NAMES].sample(min(100, len(ref_df)), random_state=42).copy()
        for col in FEATURE_NAMES:
            simulated_current[col] += 0.2
        curr_df = simulated_current
    else:
        logger.info("Loading inference logs from %s", current_path)
        curr_df = pd.read_csv(current_path)

    # Filter to identical feature sets
    available_features = [f for f in FEATURE_NAMES if f in ref_df.columns and f in curr_df.columns]
    if not available_features:
        raise ValueError(
            f"No overlapping features between reference ({ref_df.columns.tolist()}) "
            f"and current data ({curr_df.columns.tolist()})."
        )

    ref_subset = ref_df[available_features]
    curr_subset = curr_df[available_features]

    logger.info("Running Evidently Data Drift and Data Quality analysis...")
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )

    report.run(
        reference_data=ref_subset,
        current_data=curr_subset,
    )

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_report_path))
    logger.info("Evidently drift report successfully written to: %s", output_report_path)

    return output_report_path


def main() -> None:
    """Execute monitoring report generation."""
    generate_drift_report()


if __name__ == "__main__":
    main()
