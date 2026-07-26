from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features import prepare_clustering_features
from src.ingest import ingest_and_validate

# Define paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_anomaly_model(
    file_name: str = "debit_transactions.csv",
    contamination: float = 0.02,  # Expect ~2% anomalous transactions
    random_state: int = 42,
) -> tuple[IsolationForest, pd.DataFrame]:
    #Ingests data, prepares features, trains an Isolation Forest model,with MLFlow tracking
# Set or create MLflow experiment
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

        # Calculate anomaly scores (-1 for anomaly, 1 for normal)
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

        # --- MLFLOW LOGGING ---
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("total_transactions", total_records)
        mlflow.log_metric("anomaly_count", anomaly_count)
        mlflow.log_metric("anomaly_rate", anomaly_rate)

        # Log scikit-learn model artifact to MLflow
        mlflow.sklearn.log_model(model, name="isolation_forest_model")

        print("\n--- STEP 4: Saving Artifacts locally ---")
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
    