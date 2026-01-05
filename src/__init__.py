"""
Predictive Maintenance Source Package
=====================================
Enterprise-grade IoT predictive maintenance system.

Modules:
- data_pipeline: Data preprocessing and feature engineering
- modeling: Model training, tuning, and evaluation
- explain: SHAP-based model explainability
- api: REST API for inference
- config: Centralized configuration

Author: Spandana Mahadevappa Kandagal
"""

__version__ = "1.0.0"
__author__ = "IoT Predictive Maintenance Team"

from . import config
from .config import (
    PROJECT_ROOT,
    MODELS_DIR,
    DATA_DIR,
    OUTPUTS_DIR,
    DATASET_CONFIG,
    FEATURE_CONFIG,
    MODEL_CONFIG,
    API_CONFIG,
    THRESHOLD_CONFIG,
    EXPLAINABILITY_CONFIG
)

__all__ = [
    'config',
    'PROJECT_ROOT',
    'MODELS_DIR',
    'DATA_DIR', 
    'OUTPUTS_DIR',
    'DATASET_CONFIG',
    'FEATURE_CONFIG',
    'MODEL_CONFIG',
    'API_CONFIG',
    'THRESHOLD_CONFIG',
    'EXPLAINABILITY_CONFIG'
]
