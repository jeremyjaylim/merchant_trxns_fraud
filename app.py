import sys
from pathlib import Path

# Add project root directory to Python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.features import engineer_clustering_ratios, extract_temporal_features
from src.ingest import ingest_and_validate

st.set_page_config(
    page_title="Debit Anomaly & Drift Monitoring Dashboard",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = BASE_DIR / "models" / "isolation_forest.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
REPORT_PATH = BASE_DIR / "reports" / "data_drift_report.html"


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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


@st.cache_data
def load_data_and_predict():
    raw_df = ingest_and_validate("debit_transactions.csv")
    feature_df = build_feature_dataframe(raw_df)

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error(
            "❌ Missing model artifacts! Run `python -m src.train` first."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    scaled_features = scaler.transform(feature_df)
    predictions = model.predict(scaled_features)
    decision_scores = model.decision_function(scaled_features)

    df_results = raw_df.copy()
    df_results["anomaly_score"] = decision_scores
    df_results["is_anomaly"] = np.where(predictions == -1, 1, 0)

    return df_results


st.title("🛡️ Debit Transactions Anomaly & Drift Dashboard")
st.caption(
    "Real-time MLOps Monitoring, Isolation Forest Anomaly Detection, and Data Drift Analysis"
)

with st.spinner("Loading production data and model predictions..."):
    df = load_data_and_predict()

total_records = len(df)
anomalies_count = int(df["is_anomaly"].sum())
anomaly_rate = anomalies_count / total_records

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions Scanned", f"{total_records:,}")
col2.metric("Flagged Anomalies", f"{anomalies_count:,}", delta_color="inverse")
col3.metric("Current Anomaly Rate", f"{anomaly_rate:.2%}")
col4.metric(
    "System Health Status",
    "🟢 Normal" if anomaly_rate <= 0.05 else "🚨 High Anomaly Rate",
)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Transaction & Anomaly Investigation",
        "📈 Risk Feature Analysis",
        "🧪 Data Drift Report",
    ]
)

with tab1:
    st.subheader("Filter and Investigate High-Risk Transactions")
    show_only_anomalies = st.checkbox("Show Only Flagged Anomalies", value=True)
    filtered_df = df[df["is_anomaly"] == 1] if show_only_anomalies else df
    st.dataframe(
        filtered_df.sort_values(by="anomaly_score", ascending=True).head(500),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Transaction Distribution & Anomaly Score Spread")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_score = px.histogram(
            df,
            x="anomaly_score",
            color="is_anomaly",
            title="Distribution of Anomaly Decision Scores",
            color_discrete_map={0: "#1f77b4", 1: "#d62728"},
            barmode="overlay",
        )
        st.plotly_chart(fig_score, use_container_width=True)

    with col_chart2:
        fig_amt = px.scatter(
            df.sample(min(5000, len(df))),
            x="amt",
            y="Amt_Avg_Daily",
            color="is_anomaly",
            title="Transaction Amount vs. Daily Average Baseline",
            color_discrete_map={0: "#1f77b4", 1: "#d62728"},
            hover_data=["transaction_id", "customer_id"],
        )
        st.plotly_chart(fig_amt, use_container_width=True)

with tab3:
    st.subheader("Data Drift Statistical Monitoring")

    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, height=750, scrolling=True)
    else:
        st.info("💡 Generating live feature distribution drift comparison...")
        feature_df = build_feature_dataframe(df)
        split_idx = int(len(feature_df) * 0.7)
        ref_df = feature_df.iloc[:split_idx].assign(Dataset="Reference (70%)")
        curr_df = feature_df.iloc[split_idx:].assign(Dataset="Current (30%)")
        combined_df = pd.concat([ref_df, curr_df])

        fig_drift = px.histogram(
            combined_df,
            x="amt_log",
            color="Dataset",
            barmode="overlay",
            title="Log Transaction Amount Distribution: Reference vs Current Batch",
        )
        st.plotly_chart(fig_drift, use_container_width=True)