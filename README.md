💳 Debit Transaction Anomaly Detection Engine
An end-to-end Machine Learning pipeline designed to detect anomalous debit card transactions and potential fraud using unsupervised anomaly detection (Isolation Forest) and statistical distance metrics.



📌 Project Overview
This repository implements a production-grade machine learning workflow for real-time transaction monitoring. By evaluating debit card transactions against learned behavioral profiles based on historical spending patterns, the system identifies outliers and potential fraud in real time.



📁 Repository Structure
```
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # Ruff linting, pytest, Pandera schema checks, CI build
├── data/
│   └── debit_transactions.csv  # Local transaction source files
├── dags/
│   └── debit_ml_pipeline.py    # Airflow Pipeline DAG orchestrator
├── src/
│   ├── __init__.py
│   ├── data_validation.py      # Pandera schema contract for quality enforcement
│   ├── ingest.py               # Data loading and ingestion module
│   ├── features.py             # Feature engineering, ratio creation, & scaling
│   ├── train.py                # Isolation Forest training & MLflow experiment tracking
│   └── serve.py                # Inference service & anomaly scoring endpoint
├── Dockerfile.airflow          # Airflow container configuration
├── Dockerfile.mlflow           # MLflow tracking server container
├── docker-compose.yml          # Local container orchestration
├── pyproject.toml              # Project dependencies & tool configurations (uv, ruff)
└── README.md                   # Project documentation
```


🧠 Model Architecture & Methodology

Why Isolation Forest (Point Anomaly) vs. K-Means Clustering?
A common question in transaction monitoring is why this case study uses Isolation Forest instead of traditional clustering or time series modeling.

```
┌─────────────────────────────────────────────────────────────────┐
│             Isolation Forest Point Anomaly Detection             │
│                                                                  │
│   Incoming Row  ──►  [Features + Baseline Ratios]  ──►  Anomaly │
│  (Single Event)                                       Score      │
└─────────────────────────────────────────────────────────────────┘
```

**1. Unit of Prediction & Operational Real-Time Scoring**

**Time Series Modeling:** Operates on continuous, sequential curves indexed strictly over time. The goal is forecasting future values or finding sequential curve pattern similarities (e.g., using Dynamic Time Warping). Requires historical context windows.

**Isolation Forest (Point Anomaly):** Evaluates every transaction event as an independent observation in real time. Payment authorization systems require sub-second decisions on a single incoming transaction without waiting for temporal sequences. Each transaction is scored independently based on how "isolated" it is from the normal feature distribution.

**2. Encapsulation of Time via Pre-Aggregated Baselines**

Instead of evaluating long time sequences during inference, temporal behavior is encapsulated directly into the feature space using pre-calculated customer baselines:

- **Amt_Avg_Daily** (Historical average spend scale)
- **TrxnCount_Avg_Daily** (Historical velocity/frequency baseline)
- **Amount_StdDevn_Daily** (Personal spend variance)

This transforms a complex time dependency problem into a fast, highly scalable multivariate spatial distance calculation using decision trees.



🔄 End-to-End Pipeline Workflow
```
[ Ingest (Data/SQL) ] ──► [ Pandera Schema Validation ] ──► [ Feature Engineering ]
                                                                    │
                                                                    ▼
[ Fraud Alert Engine ] ◄── [ Anomaly Scoring Threshold ] ◄── [ Isolation Forest ]
```

**1. Data Ingestion (src/ingest.py)**
Loads raw transaction batches from local storage (\data) or remote database engines into Pandas DataFrames.

**2. Strict Schema Validation (src/data_validation.py)**
Data quality is enforced using Pandera. Input data must pass rigorous checks prior to feature transformation:

- **Unique Key Check:** transaction_id must be unique and non-null.
- **Format Matching:** customer_id (^C\d+$) and terminal_id (^T\d+$) pattern rules.
- **Range & Allowed Values:** amt > 0, entry_mode in ["Swipe", "Chip", "Contactless", "Online"].
- **Cross-Column Logic:** Online transactions must not have physical terminal IDs assigned.
- **Schema Strictness:** Extra unexpected columns trigger immediate validation errors (strict=True).

**3. Feature Engineering & Metadata Isolation (src/features.py)**

- **Metadata Preservation:** Identifiers (transaction_id, customer_id, post_ts) are separated into a tracking matrix to avoid interfering with distance math while enabling full production traceability.
- **Skew Compression:** Numerical features (amt, Amt_Avg_Daily, TrxnCount_Avg_Daily) undergo logarithmic transformation (log(1+x)) to normalize extreme right-skewed distributions.
- **Ratio & Temporal Generation:** Derives amt_to_avg_ratio (amt / (Amt_Avg_Daily+0.01)), hour, and is_weekend (1 for Sat/Sun, 0 for Mon-Fri).
- **Standardization:** All feature vectors are scaled using StandardScaler (μ=0, σ=1) so high dollar amounts do not drown out transaction frequency.

**4. Model Training & MLflow Tracking (src/train.py)**

Fits an unsupervised Isolation Forest algorithm on transaction features. The model learns to identify rare, isolated points in the feature space that deviate significantly from normal transaction patterns. Experiment parameters, metrics, and model artifacts are tracked via MLflow for reproducibility and model governance.

**5. Production Inference & Deployment (src/serve.py)**

During transaction monitoring, the model computes an anomaly score for each incoming transaction. If the score exceeds the defined statistical threshold (e.g., top 2% anomaly scores), the transaction is flagged as an anomaly for further review.



🛡️ Best Practices for Fraud Monitoring using Anomaly Detection

**Decouple Identifiers from Spatial Features**

Never pass raw transaction_id or customer_id into a distance model. Maintain an index map alongside feature vectors to route alerts back to risk teams.

**Logarithmic Scaling for Monetary Attributes**

Financial amounts follow heavy power-law distributions. Applying log(1+x) transformations ensures that standard distance metrics behave linearly across both low and high dollar amounts.

**Multi-Tier Risk Thresholding**

Instead of a binary pass/fail rule, anomaly scores should be mapped into risk tiers:

- **Low Risk (d≤2.0σ):** Auto-approve transaction.
- **Medium Risk (2.0σ<d≤3.5σ):** Queue for step-up authentication (OTP / Push notification).
- **High Risk (d>3.5σ):** Immediate block and analyst review alert.

**Periodic Centroid & Baseline Recalibration**

Customer spending shifts seasonally (e.g., holiday seasons). Model training and historical baseline metrics (Amt_Avg_Daily) must be periodically re-calculated via automated pipelines (Airflow) to maintain detection accuracy.



⚙️ Quickstart & Local Setup

**1. Requirements**
- Windows OS with PowerShell / Command Prompt
- Python 3.12+
- uv package manager (optional, recommended)

**2. Environment Setup**
```bash
# Clone the repository
git clone https://github.com/your-username/merchant_trxns_fraud.git
cd merchant_trxns_fraud

# Install dependencies using uv or pip
uv sync
# or
pip install -r pyproject.toml
```

**3. Run Pipeline Locally**
```bash
# 1. Validate and Ingest Data
python -m src.ingest

# 2. Feature Engineering Test
python -m src.features

# 3. Train Model & Log to MLflow
python -m src.train
```

🛠 Tech Stack
- **Language:** Python 3.12
- **Data Validation:** Pandera, Pandas
- **Machine Learning & Preprocessing:** Scikit-Learn, NumPy
- **Orchestration:** Apache Airflow
- **Experiment Tracking:** MLflow
- **Code Quality:** Ruff, Pytest
- **Containerization:** Docker & Docker Compose
