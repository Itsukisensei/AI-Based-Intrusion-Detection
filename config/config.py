import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
ALERTS_DIR = BASE_DIR / "alerts"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, TRAINED_MODELS_DIR, VISUALIZATIONS_DIR, ALERTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Data Files
RAW_DATA_PATH = RAW_DATA_DIR / "user_activity.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_data.csv"
SAMPLE_DATA_PATH = SAMPLE_DATA_DIR / "sample_activity.csv"

# Feature Definitions
RAW_FEATURE_COLUMNS = [
    "timestamp",
    "user_id",
    "role",
    "ip_address",
    "location",
    "login_time_hour",
    "session_duration_min",
    "access_frequency",
    "failed_logins",
    "file_access_count",
    "sensitive_file_access_count",
    "data_transferred_mb",
    "unusual_location_flag",
    "unusual_device_flag",
]

NUMERICAL_FEATURES = [
    "login_time_hour",
    "session_duration_min",
    "access_frequency",
    "failed_logins",
    "file_access_count",
    "sensitive_file_access_count",
    "data_transferred_mb",
    "hour_sin",
    "hour_cos",
    "sensitive_file_ratio",
    "failure_rate",
    "mb_per_minute",
    "user_zscore_data_transfer",
    "user_zscore_session_duration",
    "unusual_location_flag",
    "unusual_device_flag",
]

CATEGORICAL_FEATURES = ["role", "location"]

TARGET_COLUMN = "is_anomaly"

# Model Hyperparameters
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 150,
    "contamination": 0.08,
    "random_state": 42,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "class_weight": "balanced",
}

AUTOENCODER_PARAMS = {
    "hidden_layers": [32, 16, 8, 16, 32],
    "learning_rate": 0.005,
    "epochs": 45,
    "batch_size": 64,
    "random_state": 42,
}

# Risk Scoring Thresholds
RISK_LEVELS = {
    "LOW": (0.0, 30.0),
    "MEDIUM": (30.0, 70.0),
    "HIGH": (70.0, 90.0),
    "CRITICAL": (90.0, 100.0),
}

# Alert Threshold (Risk Score >= 70 triggers security alert)
ALERT_TRIGGER_SCORE = 70.0

# Random Seed for Reproducibility
RANDOM_SEED = 42
