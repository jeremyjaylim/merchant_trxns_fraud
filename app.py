from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.features import prepare_clustering_features
from src.ingest import ingest_and_validate

# Set page configuration
st.set_page_config(
    page_title="Debit Anomaly Investigation Dashboard",
    page_icon="🚨",
    layout="wide",
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "debit_transactions.csv"


@st.cache_resource
def load_model_artifacts():
    """Loads saved Isolation Forest model."""
    if not MODEL_PATH.exists():
        st.error(
            f"Model artifact not found at {MODEL_PATH}. Please run training first!"
        )
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_and_score_data():
    """Ingests data, generates features, and scores all transactions."""
    raw_df = ingest_and_validate("debit_transactions.csv")
    X_scaled, metadata_df, _ = prepare_clustering_features(raw_df)

    model = load_model_artifacts()

    # Predict scores and anomaly flags
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    # Combine results with raw dataframe for display
    scored_df = raw_df.copy()
    scored_df["anomaly_score"] = np.round(scores, 5)
    scored_df["is_anomaly"] = np.where(predictions == -1, "Flagged", "Normal")

    return scored_df


# --- MAIN APP LAYOUT ---
st.title("🚨 Debit Transactions Anomaly & Investigation Dashboard")
st.markdown(
    "Monitor posting patterns, inspect flagged suspicious debit activity, and investigate individual transactions."
)

with st.spinner("Loading transaction dataset and running anomaly detection..."):
    df = load_and_score_data()

# --- TOP METRICS CARDS ---
total_trx = len(df)
flagged_df = df[df["is_anomaly"] == "Flagged"]
total_flagged = len(flagged_df)
anomaly_rate = (total_flagged / total_trx) * 100
total_flagged_amt = flagged_df["amt"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_trx:,}")
col2.metric("Flagged Anomalies", f"{total_flagged:,}", delta=f"{anomaly_rate:.2f}% Rate", delta_color="inverse")
col3.metric("Total Flagged Value", f"${total_flagged_amt:,.2f}")
col4.metric("Avg Flagged Amount", f"${flagged_df['amt'].mean():,.2f}" if total_flagged > 0 else "$0.00")

st.divider()

# --- CHARTS SECTION ---
st.subheader("📊 Anomaly Distribution & Channel Breakdown")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Anomaly count by Entry Mode (Channel)
    channel_summary = (
        flagged_df.groupby("entry_mode")["transaction_id"]
        .count()
        .reset_index()
        .rename(columns={"entry_mode": "Channel / Entry Mode", "transaction_id": "Flagged Count"})
    )
    fig_channel = px.bar(
        channel_summary,
        x="Channel / Entry Mode",
        y="Flagged Count",
        text="Flagged Count",
        title="Flagged Anomalies by Channel (entry_mode)",
        color="Channel / Entry Mode",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_channel, use_container_width=True)

with chart_col2:
    # Transaction Amount vs Daily Average Baseline Scatter Plot
    fig_scatter = px.scatter(
        df,
        x="Amt_Avg_Daily",
        y="amt",
        color="is_anomaly",
        color_discrete_map={"Normal": "#A0A0A0", "Flagged": "#FF4B4B"},
        hover_data=["transaction_id", "customer_id", "entry_mode", "amt"],
        labels={"Amt_Avg_Daily": "Historical Avg Daily Spend ($)", "amt": "Transaction Amount ($)"},
        title="Transaction Amount vs Customer Daily Baseline",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# --- INVESTIGATION TABLE ---
st.subheader("🔎 Flagged Transactions Requiring Investigation")

# Sidebar Filters
st.sidebar.header("Filter Investigation Queue")
selected_channel = st.sidebar.multiselect(
    "Filter by Channel (entry_mode):",
    options=df["entry_mode"].unique(),
    default=df["entry_mode"].unique(),
)

score_threshold = st.sidebar.slider(
    "Max Anomaly Score (Lower = More Outlier):",
    min_value=float(df["anomaly_score"].min()),
    max_value=float(df["anomaly_score"].max()),
    value=float(df["anomaly_score"].max()),
    step=0.01,
)

# Apply filters to flagged dataframe
filtered_flagged = flagged_df[
    (flagged_df["entry_mode"].isin(selected_channel))
    & (flagged_df["anomaly_score"] <= score_threshold)
].sort_values(by="anomaly_score")

# Select and order required columns explicitly
display_cols = [
    "transaction_id",
    "customer_id",
    "entry_mode",  # Channel
    "amt",
    "Amt_Avg_Daily",
    "anomaly_score",
    "MerchantType",
    "post_ts",
    "terminal_id",
]

st.dataframe(
    filtered_flagged[display_cols].rename(
        columns={
            "transaction_id": "Transaction ID",
            "customer_id": "Customer ID",
            "entry_mode": "Channel",
            "amt": "Amount ($)",
            "Amt_Avg_Daily": "Daily Avg Baseline ($)",
            "anomaly_score": "Anomaly Score",
            "MerchantType": "Merchant Category",
            "post_ts": "Timestamp",
            "terminal_id": "Terminal ID",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Showing **{len(filtered_flagged)}** flagged transactions requiring review based on selected filters."
)