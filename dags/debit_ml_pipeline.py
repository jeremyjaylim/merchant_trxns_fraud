from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'jeremy',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'debit_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for merchant transaction fraud detection',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # Task 1: Ingest and Validate Data
    ingest_task = BashOperator(
        task_id='ingest_data',
        bash_command='python -m src.ingest',
    )

    # Task 2: Feature Engineering
    features_task = BashOperator(
        task_id='generate_features',
        bash_command='python -m src.features',
    )

    # Task 3: Train Model & Log to MLflow
    train_task = BashOperator(
        task_id='train_model',
        bash_command='python -m src.train',
    )

    # Define execution order
    ingest_task >> features_task >> train_task