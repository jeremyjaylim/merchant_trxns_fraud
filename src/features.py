import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts hour and is_weekend flag from posting timestamp."""
    df = df.copy()
    post_ts = pd.to_datetime(df["post_ts"])

    df["hour"] = post_ts.dt.hour
    df["is_weekend"] = post_ts.dt.dayofweek.isin([5, 6]).astype(int)

    return df


def engineer_clustering_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Generates relative ratio features to capture spending anomalies."""
    df = df.copy()

    # Relative magnitude: current transaction amt vs daily customer baseline
    df["amt_to_avg_ratio"] = df["amt"] / (df["Amt_Avg_Daily"] + 0.01)

    # Z-score deviation against personal historical variance
    df["amt_z_score"] = (df["amt"] - df["Amt_Avg_Daily"]) / (
        df["Amount_StdDevn_Daily"] + 0.01
    )

    return df


def prepare_clustering_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """Prepares feature matrix for clustering while retaining tracking metadata.

    Returns:
    --------
    X_scaled : np.ndarray
        Scaled matrix ready to feed into KMeans / distance models.
    metadata_df : pd.DataFrame
        DataFrame containing transaction_id, customer_id, and post_ts for tracking.
    scaler : StandardScaler
        Fitted scaler instance for inference transform.
    """
    # 1. Preserve Metadata Identifiers
    metadata_cols = ["transaction_id", "customer_id", "post_ts"]
    metadata_df = df[metadata_cols].copy()

    # 2. Derive Temporal & Ratio Features
    transformed_df = extract_temporal_features(df)
    transformed_df = engineer_clustering_ratios(transformed_df)

    # 3. Apply Log Transformation to Right-Skewed Numeric Columns
    feature_df = pd.DataFrame()
    feature_df["amt_log"] = np.log1p(transformed_df["amt"])
    feature_df["amt_avg_daily_log"] = np.log1p(transformed_df["Amt_Avg_Daily"])
    feature_df["trxn_count_avg_daily_log"] = np.log1p(
        transformed_df["TrxnCount_Avg_Daily"]
    )
    feature_df["amt_to_avg_ratio_log"] = np.log1p(
        transformed_df["amt_to_avg_ratio"]
    )

    # 4. Include Bounded Temporal Features
    feature_df["hour"] = transformed_df["hour"]
    feature_df["is_weekend"] = transformed_df["is_weekend"]

    # 5. Standardize Features (Zero Mean, Unit Variance for Distance Metrics)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    return X_scaled, metadata_df, scaler


if __name__ == "__main__":
    # Test script locally with ingest module
    from src.ingest import ingest_and_validate

    raw_df = ingest_and_validate("debit_transactions.csv")
    X_scaled, metadata_df, scaler = prepare_clustering_features(raw_df)

    print("Clustering feature preparation successful!")
    print(f"Scaled feature matrix shape: {X_scaled.shape}")
    print(f"Metadata index preserved with {len(metadata_df)} rows.")
    print("Metadata Sample:\n", metadata_df.head(3))