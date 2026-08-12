import sys
from pathlib import Path

# Add project root directory to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import joblib
import mlflow
import numpy as np
import pandas as pd

import evidently

# Standard Evidently Imports with clean version fallbacks
try:
    from evidently.report import Report
except ImportError:
    from evidently import Report

try:
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    try:
        from evidently.presets import DataDriftPreset
    except ImportError:
        from evidently.metrics import DataDriftPreset

from src.features import extract_temporal_features, engineer_clustering_ratios
from src.ingest import ingest_and_validate

MODEL_PATH = BASE_DIR / "models" / "isolation_forest.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstructs feature matrix for monitoring."""
    transformed_df = extract_temporal_features(df)
    transformed_df = engineer_clustering_ratios(transformed_df)

    feature_df = pd.DataFrame()
    feature_df["amt_log"] = np.log1p(transformed_df["amt"])
    feature_df["amt_avg_daily_log"] = np.log1p(transformed_df["Amt_Avg_Daily"])
    feature_df["trxn_count_avg_daily_log"] = np.log1p(
        transformed_df["TrxnCount_Avg_Daily"]
    )
    feature_df["amt_to_avg_ratio_log"] = np.log1p(
        transformed_df["amt_to_avg_ratio"]
    )
    feature_df["hour"] = transformed_df["hour"]
    feature_df["is_weekend"] = transformed_df["is_weekend"]

    return feature_df


def save_report_html(report: Report, output_path: Path):
    """Exports Evidently Report to HTML across all package versions."""
    if hasattr(report, "save_html"):
        report.save_html(str(output_path))
    elif hasattr(report, "save"):
        report.save(str(output_path))
    else:
        html_str = None
        if hasattr(report, "get_html"):
            html_str = report.get_html()
        elif hasattr(report, "_repr_html_"):
            html_str = report._repr_html_()
        
        if html_str:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_str)
        else:
            raise AttributeError(
                f"Report object of type {type(report)} lacks standard HTML export methods."
            )


def run_evidently_monitoring():
    print("📥 Ingesting dataset for drift analysis...")
    raw_df = ingest_and_validate("debit_transactions.csv")
    feature_df = build_feature_dataframe(raw_df)

    # Split data into Reference (70% Baseline) and Current (30% Recent) sets
    split_idx = int(len(feature_df) * 0.7)
    reference_df = feature_df.iloc[:split_idx]
    current_df = feature_df.iloc[split_idx:]

    print("📊 Generating Evidently Visual Data Drift Report...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)

    html_report_path = REPORTS_DIR / "data_drift_report.html"
    save_report_html(drift_report, html_report_path)
    print(f"✅ Visual Drift Report generated at: {html_report_path}")

    MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("Debit_Anomaly_Detection")

    with mlflow.start_run(run_name="Evidently_Drift_Monitoring"):
        mlflow.log_artifact(str(html_report_path))
        print("🚀 HTML visual drift report logged as MLflow artifact.")


if __name__ == "__main__":
    run_evidently_monitoring()