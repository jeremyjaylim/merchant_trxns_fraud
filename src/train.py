import sys
from pathlib import Path

# Define paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from src.features import prepare_clustering_features
from src.ingest import ingest_and_validate

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Point MLflow explicitly to local SQLite backend store
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")


def train_anomaly_model(
    file_name: str = "debit_transactions.csv",
    contamination: float = 0.02,  # Expect ~2% anomalous transactions
    random_state: int = 42,
) -> tuple[IsolationForest, pd.DataFrame]:
    """Ingests data, prepares features, trains an Isolation Forest model,

    computes unsupervised evaluation metrics, and logs results to MLflow.
    """
    mlflow.set_experiment("Debit_Anomaly_Detection")

    with mlflow.start_run():
        print("--- STEP 1: Ingesting & Validating Data ---")
        raw_df = ingest_and_validate(file_name)

        print("\n--- STEP 2: Preparing Features ---")
        X_scaled, metadata_df, scaler = prepare_clustering_features(raw_df)

        print("\n--- STEP 3: Training Isolation Forest Model ---")
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_scaled)

        # Calculate decision scores (-1 for anomaly, 1 for normal)
        scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)

        # Attach results to metadata
        results_df = metadata_df.copy()
        results_df["anomaly_score"] = scores
        results_df["is_anomaly"] = np.where(predictions == -1, 1, 0)

        anomaly_count = int(results_df["is_anomaly"].sum())
        total_records = len(results_df)
        anomaly_rate = float(anomaly_count / total_records)

        print(
            f"Training complete. Identified {anomaly_count} anomalies out of {total_records} transactions ({anomaly_rate:.2%})."
        )

        print("\n--- STEP 4: Computing Model Evaluation Metrics ---")
        # 1. Decision score distribution metrics
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        min_score = float(np.min(scores))

        # 2. Subsample for Silhouette Score to maintain fast training speed on 140k+ rows
        sample_size = min(10000, len(X_scaled))
        np.random.seed(random_state)
        sample_idx = np.random.choice(
            len(X_scaled), size=sample_size, replace=False
        )

        sil_score = float(
            silhouette_score(X_scaled[sample_idx], predictions[sample_idx])
        )
        ch_score = float(calinski_harabasz_score(X_scaled, predictions))

        print(f"  - Silhouette Score (Sampled): {sil_score:.4f}")
        print(f"  - Calinski-Harabasz Index: {ch_score:.2f}")
        print(f"  - Mean Anomaly Decision Score: {mean_score:.4f}")

        # --- MLFLOW LOGGING ---
        # Hyperparameters
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", random_state)

        # Dataset & Anomaly Metrics
        mlflow.log_metric("total_transactions", total_records)
        mlflow.log_metric("anomaly_count", anomaly_count)
        mlflow.log_metric("anomaly_rate", anomaly_rate)

        # Unsupervised Model Evaluation Metrics
        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("mean_decision_score", mean_score)
        mlflow.log_metric("std_decision_score", std_score)
        mlflow.log_metric("min_decision_score", min_score)

        # Log scikit-learn model artifact to MLflow
        mlflow.sklearn.log_model(model, name="isolation_forest_model")

        print("\n--- STEP 5: Saving Artifacts Locally ---")
        model_path = MODEL_DIR / "isolation_forest.joblib"
        scaler_path = MODEL_DIR / "scaler.joblib"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Log local files as artifacts
        mlflow.log_artifact(str(scaler_path))

        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

    return model, results_df


if __name__ == "__main__":
    model, results_df = train_anomaly_model()
    print("\nSample High-Risk Flagged Transactions:")
    print(
        results_df[results_df["is_anomaly"] == 1]
        .sort_values(by="anomaly_score")
        .head(5)
    )