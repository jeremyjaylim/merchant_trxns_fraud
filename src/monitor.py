import json
import sys
from pathlib import Path

# Add project root directory to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features import engineer_clustering_ratios, extract_temporal_features
from src.ingest import ingest_and_validate

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstructs feature matrix for drift monitoring."""
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


def calculate_data_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame
) -> pd.DataFrame:
    """Performs two-sample Kolmogorov-Smirnov test per feature to detect distribution drift."""
    drift_results = []
    for col in reference_df.columns:
        stat, p_value = ks_2samp(reference_df[col], current_df[col])
        is_drifted = bool(p_value < 0.05)
        drift_results.append(
            {
                "feature": col,
                "ks_stat": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": is_drifted,
                "ref_mean": round(float(reference_df[col].mean()), 4),
                "curr_mean": round(float(current_df[col].mean()), 4),
            }
        )
    return pd.DataFrame(drift_results)


def generate_drift_html_report(
    drift_df: pd.DataFrame, ref_count: int, curr_count: int
) -> str:
    """Renders an executive-grade HTML dashboard for data drift statistics."""
    drifted_features = int(drift_df["drift_detected"].sum())
    total_features = len(drift_df)
    overall_status = (
        "🔴 High Drift Detected"
        if drifted_features > 0
        else "🟢 Baseline Stable"
    )

    rows_html = ""
    for _, row in drift_df.iterrows():
        status_badge = (
            '<span style="background:#431418; color:#ff4d4f; padding:4px 12px; border-radius:12px; font-weight:bold;">Drifted</span>'
            if row["drift_detected"]
            else '<span style="background:#1c3b2b; color:#52c41a; padding:4px 12px; border-radius:12px; font-weight:bold;">Stable</span>'
        )
        rows_html += f"""
        <tr>
            <td><strong>{row['feature']}</strong></td>
            <td>{status_badge}</td>
            <td>{row['ks_stat']}</td>
            <td>{row['p_value']}</td>
            <td>{row['ref_mean']}</td>
            <td>{row['curr_mean']}</td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Data Drift Monitoring Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0e1117; color: #ffffff; padding: 30px; }}
        .card {{ background: #1e222d; padding: 24px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #2e3440; }}
        h1 {{ color: #40a9ff; margin-top: 0; font-size: 24px; }}
        .summary-box {{ display: flex; gap: 20px; margin-top: 15px; }}
        .metric-card {{ background: #252a37; padding: 15px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #343a46; }}
        .metric-val {{ font-size: 22px; font-weight: bold; margin-top: 5px; color: #e6f7ff; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #2e3440; }}
        th {{ background-color: #2a2e3d; color: #a0a6b5; text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px; }}
        tr:hover {{ background-color: #252a37; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>📊 Data Drift Statistical Analysis (KS-Test)</h1>
        <p>Comparing baseline historical transactions against recent incoming batches.</p>
        <div class="summary-box">
            <div class="metric-card">
                <div>Overall Status</div>
                <div class="metric-val">{overall_status}</div>
            </div>
            <div class="metric-card">
                <div>Drifted Features</div>
                <div class="metric-val">{drifted_features} / {total_features}</div>
            </div>
            <div class="metric-card">
                <div>Reference Sample Size</div>
                <div class="metric-val">{ref_count:,}</div>
            </div>
            <div class="metric-card">
                <div>Current Sample Size</div>
                <div class="metric-val">{curr_count:,}</div>
            </div>
        </div>
    </div>
    <div class="card">
        <h3>Feature Breakdown</h3>
        <table>
            <thead>
                <tr>
                    <th>Feature Name</th>
                    <th>Drift Status</th>
                    <th>KS Statistic</th>
                    <th>p-value</th>
                    <th>Ref Mean</th>
                    <th>Curr Mean</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>"""


def run_drift_monitoring():
    print("📥 Ingesting dataset for drift analysis...")
    raw_df = ingest_and_validate("debit_transactions.csv")
    feature_df = build_feature_dataframe(raw_df)

    # 1. Split data into Reference (70%) and Current (30%)
    split_idx = int(len(feature_df) * 0.7)
    reference_df = feature_df.iloc[:split_idx]
    current_df = feature_df.iloc[split_idx:]

    print("📊 Executing Kolmogorov-Smirnov Statistical Drift Analysis...")
    drift_df = calculate_data_drift(reference_df, current_df)

    # 2. Generate HTML Report
    html_report_path = REPORTS_DIR / "data_drift_report.html"
    html_content = generate_drift_html_report(
        drift_df, len(reference_df), len(current_df)
    )

    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Data Drift HTML Report generated at: {html_report_path}")

    # 3. Log to MLflow
    MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("Debit_Anomaly_Detection")

    with mlflow.start_run(run_name="Data_Drift_Monitoring"):
        mlflow.log_metric(
            "drifted_features_count", int(drift_df["drift_detected"].sum())
        )
        mlflow.log_artifact(str(html_report_path))
        print("🚀 Visual drift report successfully logged to MLflow artifacts.")


if __name__ == "__main__":
    run_drift_monitoring()