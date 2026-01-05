"""
API Module
==========
REST API and inference engine for production deployment.
"""

from .inference import InferenceEngine, InputValidator, predict_single, predict_batch
from .app import app, create_app

__all__ = [
    'InferenceEngine',
    'InputValidator',
    'predict_single',
    'predict_batch',
    'app',
    'create_app'
]
