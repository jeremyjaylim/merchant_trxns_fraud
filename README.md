💳 Debit Transaction Anomaly Detection Engine
An end-to-end Machine Learning pipeline designed to detect anomalous debit card transactions and potential fraud using unsupervised clustering (K-Means) and statistical distance metrics.



📌 Project Overview
This repository implements a production-grade machine learning workflow for real-time transaction monitoring. By grouping debit card transactions into behavioral clusters based on historical spending baselines, transaction velocity, and temporal features, the pipeline identifies high-risk outliers (anomalies) that deviate significantly from expected normal behavior.



📁 Repository Structure
Plaintext
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # Ruff linting, pytest, Pandera schema checks, CI build
├── data/
│   └── debit\_transactions.csv  # Local transaction source files
├── dags/
│   └── debit\_ml\_pipeline.py    # Airflow Pipeline DAG orchestrator
├── src/
│   ├── **init**.py
│   ├── data\_validation.py      # Pandera schema contract for quality enforcement
│   ├── ingest.py               # Data loading and ingestion module
│   ├── features.py             # Feature engineering, ratio creation, \& scaling
│   ├── train.py                # KMeans training \& MLflow experiment tracking
│   └── serve.py                # Inference service \& anomaly scoring endpoint
├── Dockerfile.airflow          # Airflow container configuration
├── Dockerfile.mlflow           # MLflow tracking server container
├── docker-compose.yml          # Local container orchestration
├── pyproject.toml              # Project dependencies \& tool configurations (uv, ruff)
└── README.md                   # Project documentation


🧠 Model Architecture \& Methodology
Why Tabular Clustering (Point Anomaly) vs. Time Series Modeling?
A common question in transaction monitoring is why this case study uses unsupervised tabular clustering instead of a time series model.

┌────────────────────────────────────────────────────────────────────────┐
│                        Tabular Event Clustering                        │
│                                                                        │
│   Incoming Row  ──►  \[Features + Baseline Ratios]  ──►  Distance to   │
│  (Single Event)                                        Centroid (Score)│
└────────────────────────────────────────────────────────────────────────┘

1. Unit of Prediction \& Operational Real-Time Scoring
Time Series Modeling: Operates on continuous, sequential curves indexed strictly over time. The goal is forecasting future values or finding sequential curve pattern similarities (e.g., using Dynamic Time Warping).

Tabular Event Clustering: Treats every transaction event as an independent observation evaluated in real time. Payment authorization systems require sub-second decisions on a single incoming transaction, making point-anomaly detection far more computationally efficient.

2. Encapsulation of Time via Pre-Aggregated Baselines
Instead of evaluating long time sequences during inference, temporal behavior is encapsulated directly into the feature space using pre-calculated customer baselines:

Amt\_Avg\_Daily (Historical average spend scale)

TrxnCount\_Avg\_Daily (Historical velocity/frequency baseline)

Amount\_StdDevn\_Daily (Personal spend variance)

This transforms a complex time dependency problem into a fast, highly scalable multivariate spatial distance calculation.



🔄 End-to-End Pipeline Workflow
\[ Ingest (Data/SQL) ] ──► \[ Pandera Schema Validation ] ──► \[ Feature Engineering ]
│
▼
\[ Fraud Alert Engine ] ◄── \[ Anomaly Scoring Threshold ] ◄── \[ KMeans Model ]

1. Data Ingestion (src/ingest.py)
Loads raw transaction batches from local storage (\\data) or remote database engines into Pandas DataFrames.
2. Strict Schema Validation (src/data\_validation.py)
Data quality is enforced using Pandera. Input data must pass rigorous checks prior to feature transformation:

Unique Key Check: transaction\_id must be unique and non-null.

Format Matching: customer\_id (^C\\d+$) and terminal\_id (^T\\d+$) pattern rules.

Range \& Allowed Values: amt > 0, entry\_mode in \["Swipe", "Chip", "Contactless", "Online"].

Cross-Column Logic: Online transactions must not have physical terminal IDs assigned.

Schema Strictness: Extra unexpected columns trigger immediate validation errors (strict=True).

3. Feature Engineering \& Metadata Isolation (src/features.py)
Metadata Preservation: Identifiers (transaction\_id, customer\_id, post\_ts) are separated into a tracking matrix to avoid interfering with distance math while enabling full production traceability.

Skew Compression: Numerical features (amt, Amt\_Avg\_Daily, TrxnCount\_Avg\_Daily) undergo logarithmic transformation (log(1+x)) to normalize extreme right-skewed distributions.

Ratio \& Temporal Generation: Derives amt\_to\_avg\_ratio (Amt\_Avg\_Daily+0.01,amt), hour, and is\_weekend (1 for Sat/Sun, 0 for Mon-Fri).



Standardization: All feature vectors are scaled using StandardScaler (μ=0,σ=1) so high dollar amounts do not drown out transaction frequency.

4. Model Training \& MLflow Tracking (src/train.py)
Fits an unsupervised KMeans algorithm on normal behavioral profiles, tuning K using inertia and silhouette scoring. Experiment parameters, inertia plots, and model artifacts are tracked via MLflow.
5. Production Inference \& Deployment (src/serve.py)
During transaction monitoring, the model predicts the assigned cluster centroid for each transaction vector and calculates its Euclidean distance:

&#x20;exceeds the defined statistical threshold (e.g., top 99th percentile of distance distribution), the transaction is flagged as an anomaly.



🛡️ Best Practices for Fraud Monitoring using Clustering
Decouple Identifiers from Spatial Features

Never pass raw transaction\_id or customer\_id into a distance model. Maintain an index map alongside feature vectors to route alerts back to risk teams.

Logarithmic Scaling for Monetary Attributes

Financial amounts follow heavy power-law distributions. Applying log(1+x) transformations ensures that standard distance metrics behave linearly across both low and high dollar amounts.

Multi-Tier Risk Thresholding

Instead of a binary pass/fail rule, distance scores should be mapped into risk tiers:

Low Risk (d≤2.0σ): Auto-approve transaction.

Medium Risk (2.0σ<d≤3.5σ): Queue for step-up authentication (OTP / Push notification).

High Risk (d>3.5σ): Immediate block and analyst review alert.

Periodic Centroid \& Baseline Recalibration

Customer spending shifts seasonally (e.g., holiday seasons). Clustering centers and historical baseline metrics (Amt\_Avg\_Daily) must be periodically re-calculated via automated pipelines (Airflow) to avoid false-positive inflation.



⚙️ Quickstart \& Local Setup

1. Requirements
Windows OS with PowerShell / Command Prompt

Python 3.12+

uv package manager (optional, recommended)

2. Environment Setup
DOS
:: Clone the repository
git clone https://github.com/your-username/debit-ml-pipeline.git
cd debit-ml-pipeline

:: Install dependencies using uv or pip
pip install -r pyproject.toml
3. Run Pipeline Locally
DOS
:: 1. Validate and Ingest Data
python -m src.ingest

:: 2. Feature Engineering Test
python -m src.features

:: 3. Train Model \& Log to MLflow
python -m src.train
🛠 Tech Stack
Language: Python 3.12

Data Validation: Pandera, Pandas

Machine Learning \& Preprocessing: Scikit-Learn, NumPy

Orchestration: Apache Airflow

Experiment Tracking: MLflow

Code Quality: Ruff, Pytest

Containerization: Docker \& Docker Compose

