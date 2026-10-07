"""Central configuration. All paths + hyperparameters in one place (no magic numbers)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "credit_customers.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RISK_MODEL_PATH = MODELS_DIR / "credit_risk_pipeline.pkl"
RISK_METRICS_PATH = MODELS_DIR / "credit_risk_metrics.json"
CLUSTER_MODEL_PATH = MODELS_DIR / "kmeans_pipeline.pkl"
CLUSTER_SUMMARY_PATH = MODELS_DIR / "cluster_summary.json"
ANOMALY_MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"

TARGET_COL = "class"
POSITIVE_LABEL = "good"  # low risk
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

# Clustering / anomaly defaults (chosen after silhouette comparison k=2..5)
N_CLUSTERS = 4
ANOMALY_CONTAMINATION = 0.10

APP_NAME = "AI-Powered Credit Risk & Financial Decision Support Platform"
MODEL_VERSION = "1.0.0"
