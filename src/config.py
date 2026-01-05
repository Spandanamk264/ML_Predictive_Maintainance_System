"""
Configuration Module for Predictive Maintenance System
========================================================
Centralized configuration for all modules including paths, model parameters,
API settings, and operational thresholds.
"""
import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SHAP_PLOTS_DIR = OUTPUTS_DIR / "shap_plots"
ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts"

# Ensure directories exist
for dir_path in [PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, SHAP_PLOTS_DIR, ARTIFACTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
@dataclass
class DatasetConfig:
    """Configuration for dataset handling."""
    # Raw data file
    raw_data_filename: str = "Predictive Maintainance dataset.csv"
    processed_data_filename: str = "processed_data.csv"
    
    # Target column
    target_column: str = "Machine failure"
    
    # Sensor columns (primary features)
    sensor_columns: List[str] = field(default_factory=lambda: [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ])
    
    # Columns to exclude from features
    exclude_columns: List[str] = field(default_factory=lambda: [
        "Machine failure", "Type", "UDI", "Product ID",
        "TWF", "HDF", "PWF", "OSF", "RNF"
    ])
    
    # Product type categories
    product_types: List[str] = field(default_factory=lambda: ["L", "M", "H"])
    
    # Failure type columns (for analysis, not modeling)
    failure_type_columns: List[str] = field(default_factory=lambda: [
        "TWF", "HDF", "PWF", "OSF", "RNF"
    ])

DATASET_CONFIG = DatasetConfig()


# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Timestamp generation
    timestamp_start: str = "2024-01-01 00:00:00"
    timestamp_interval_minutes: int = 10
    
    # Lag features
    lag_steps: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Rolling window sizes (in number of observations)
    # At 10-min intervals: 1h=6, 4h=24, 8h=48
    rolling_windows: Dict[str, int] = field(default_factory=lambda: {
        "1h": 6,
        "4h": 24,
        "8h": 48
    })
    
    # Rolling statistics to compute
    rolling_stats: List[str] = field(default_factory=lambda: [
        "mean", "std", "min", "max"
    ])
    
    # Exponential moving average spans
    ema_spans: List[int] = field(default_factory=lambda: [6, 12, 24])
    
    # Binary failure target configuration
    failure_leading_window_hours: int = 24
    
    # Data quality thresholds
    min_temperature_kelvin: float = 200.0  # Absolute zero is 0K
    max_temperature_kelvin: float = 500.0  # Reasonable upper bound
    min_rpm: float = 0.0
    max_rpm: float = 10000.0
    min_torque: float = 0.0
    max_torque: float = 200.0
    min_tool_wear: float = 0.0
    max_tool_wear: float = 500.0

FEATURE_CONFIG = FeatureConfig()


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""
    # Random seed for reproducibility
    random_state: int = 42
    
    # Train/test split ratio
    train_ratio: float = 0.8
    
    # Cross-validation settings
    cv_splits: int = 5
    cv_splits_tuning: int = 3
    
    # Logistic Regression (Baseline)
    baseline_max_iter: int = 1000
    
    # XGBoost default parameters
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # XGBoost hyperparameter search space
    xgb_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2]
    })
    
    # Random Forest hyperparameter search space
    rf_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"]
    })
    
    # Hyperparameter tuning iterations
    n_iter_randomized_search: int = 50
    
    # SMOTE configuration (for handling class imbalance)
    smote_sampling_strategy: float = 0.5  # Target minority/majority ratio
    smote_k_neighbors: int = 5
    
    # Model file names
    baseline_model_filename: str = "baseline_model.joblib"
    xgb_model_filename: str = "final_xgb_model.joblib"
    rf_model_filename: str = "rf_model.joblib"
    preprocessing_pipeline_filename: str = "preprocessing_pipeline.joblib"
    scaler_filename: str = "scaler.joblib"
    full_pipeline_filename: str = "full_inference_pipeline.joblib"

MODEL_CONFIG = ModelConfig()


# =============================================================================
# API CONFIGURATION
# =============================================================================
@dataclass
class APIConfig:
    """Configuration for Flask REST API."""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    
    # Authentication
    api_key_header: str = "X-API-Key"
    api_keys: List[str] = field(default_factory=lambda: [
        os.environ.get("API_KEY", "dev-key-12345")
    ])
    require_auth: bool = False  # Set to True in production
    
    # Rate limiting
    rate_limit_per_minute: int = 100
    
    # Request validation
    max_content_length_mb: int = 10
    
    # Latency targets (in milliseconds)
    target_p50_latency_ms: float = 20.0
    target_p95_latency_ms: float = 50.0
    target_p99_latency_ms: float = 100.0
    
    # Logging
    log_requests: bool = True
    log_responses: bool = True
    log_inference_latency: bool = True

API_CONFIG = APIConfig()


# =============================================================================
# PREDICTION THRESHOLDS
# =============================================================================
@dataclass
class ThresholdConfig:
    """Configuration for prediction decision thresholds."""
    # Default probability threshold for failure prediction
    failure_probability_threshold: float = 0.5
    
    # High-risk threshold (for alerts)
    high_risk_threshold: float = 0.8
    
    # Medium-risk threshold
    medium_risk_threshold: float = 0.3
    
    # Risk levels
    risk_levels: Dict[str, Dict] = field(default_factory=lambda: {
        "critical": {"min": 0.8, "max": 1.0, "action": "immediate_maintenance"},
        "high": {"min": 0.5, "max": 0.8, "action": "schedule_maintenance"},
        "medium": {"min": 0.3, "max": 0.5, "action": "monitor_closely"},
        "low": {"min": 0.0, "max": 0.3, "action": "normal_operation"}
    })

THRESHOLD_CONFIG = ThresholdConfig()


# =============================================================================
# SHAP/EXPLAINABILITY CONFIGURATION
# =============================================================================
@dataclass
class ExplainabilityConfig:
    """Configuration for model explainability."""
    # Number of samples for SHAP background
    shap_background_samples: int = 100
    
    # Number of top features to display
    top_features_count: int = 10
    
    # Plot settings
    plot_dpi: int = 300
    plot_format: str = "png"
    
    # Domain validation rules
    # Maps feature patterns to expected SHAP direction
    # Positive means higher values should increase failure probability
    expected_shap_directions: Dict[str, float] = field(default_factory=lambda: {
        "Torque [Nm]": 1.0,           # Higher torque -> higher failure risk
        "Tool wear [min]": 1.0,        # More wear -> higher failure risk
        "Process temperature [K]": 1.0, # Higher temp -> higher failure risk
        "Rotational speed [rpm]": 0.0,  # Complex relationship
        "Temp_diff": 1.0,              # Higher temp diff -> higher risk
        "Power": 1.0,                  # Higher power -> higher risk
    })
    
    # Anomaly detection threshold for SHAP validation
    shap_anomaly_threshold: float = 0.3

EXPLAINABILITY_CONFIG = ExplainabilityConfig()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = str(OUTPUTS_DIR / "app.log")
    log_to_console: bool = True
    log_to_file: bool = True

LOGGING_CONFIG = LoggingConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_raw_data_path() -> Path:
    """Get the path to raw data file."""
    return RAW_DATA_DIR / DATASET_CONFIG.raw_data_filename


def get_processed_data_path() -> Path:
    """Get the path to processed data file."""
    return PROCESSED_DATA_DIR / DATASET_CONFIG.processed_data_filename


def get_model_path(model_name: str) -> Path:
    """Get the path to a model file."""
    return MODELS_DIR / model_name


def get_all_config() -> Dict[str, Any]:
    """Return all configuration as a dictionary."""
    from dataclasses import asdict
    return {
        "dataset": asdict(DATASET_CONFIG),
        "feature": asdict(FEATURE_CONFIG),
        "model": asdict(MODEL_CONFIG),
        "api": asdict(API_CONFIG),
        "threshold": asdict(THRESHOLD_CONFIG),
        "explainability": asdict(EXPLAINABILITY_CONFIG),
        "logging": asdict(LOGGING_CONFIG),
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "outputs_dir": str(OUTPUTS_DIR)
        }
    }


if __name__ == "__main__":
    # Print configuration summary
    import json
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE CONFIGURATION")
    print("=" * 60)
    config = get_all_config()
    print(json.dumps(config, indent=2, default=str))
