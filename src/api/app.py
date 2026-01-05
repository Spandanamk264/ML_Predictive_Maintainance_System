"""
Flask REST API for Predictive Maintenance
==========================================
Production-ready REST API for real-time machine failure prediction.

Features:
- POST /predict: Single prediction with SHAP explanation
- POST /batch_predict: Batch predictions
- GET /health: Health check endpoint
- GET /model/info: Model metadata
- Input schema validation
- API key authentication (optional)
- Request/response logging
- Latency instrumentation

Author: IoT Predictive Maintenance Team
"""
from flask import Flask, request, jsonify, g
from functools import wraps
import time
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import API_CONFIG, THRESHOLD_CONFIG, MODELS_DIR
from api.inference import InferenceEngine, InputValidator

# =============================================================================
# APP CONFIGURATION
# =============================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = API_CONFIG.max_content_length_mb * 1024 * 1024

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL STATE
# =============================================================================
engine: Optional[InferenceEngine] = None
request_count = 0
latency_samples = []


def get_engine() -> InferenceEngine:
    """Get or initialize the inference engine."""
    global engine
    if engine is None:
        logger.info("Initializing inference engine...")
        engine = InferenceEngine(str(MODELS_DIR))
        logger.info("Inference engine ready")
    return engine


# =============================================================================
# MIDDLEWARE
# =============================================================================
@app.before_request
def before_request():
    """Pre-request processing."""
    g.start_time = time.time()
    g.request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{request_count}"
    
    if API_CONFIG.log_requests:
        logger.info(f"[{g.request_id}] {request.method} {request.path} - IP: {request.remote_addr}")


@app.after_request
def after_request(response):
    """Post-request processing."""
    global request_count, latency_samples
    
    # Calculate latency
    if hasattr(g, 'start_time'):
        latency_ms = (time.time() - g.start_time) * 1000
        latency_samples.append(latency_ms)
        
        # Keep only last 1000 samples
        if len(latency_samples) > 1000:
            latency_samples = latency_samples[-1000:]
        
        if API_CONFIG.log_inference_latency:
            logger.info(f"[{g.request_id}] Response: {response.status_code} - Latency: {latency_ms:.2f}ms")
        
        # Add latency header
        response.headers['X-Inference-Latency-Ms'] = str(round(latency_ms, 2))
        response.headers['X-Request-Id'] = g.request_id
    
    request_count += 1
    return response


# =============================================================================
# AUTHENTICATION
# =============================================================================
def require_api_key(f):
    """Decorator for API key authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_CONFIG.require_auth:
            return f(*args, **kwargs)
        
        api_key = request.headers.get(API_CONFIG.api_key_header)
        
        if not api_key:
            return jsonify({
                'error': 'API key required',
                'message': f'Please provide API key in {API_CONFIG.api_key_header} header'
            }), 401
        
        if api_key not in API_CONFIG.api_keys:
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is not valid'
            }), 403
        
        return f(*args, **kwargs)
    
    return decorated


# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    return jsonify({
        'error': 'Bad Request',
        'message': str(error.description) if hasattr(error, 'description') else 'Invalid request'
    }), 400


@app.errorhandler(500)
def internal_error(error):
    """Handle internal errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON with health status and model state
    """
    eng = get_engine()
    
    # Calculate latency percentiles
    if latency_samples:
        sorted_latencies = sorted(latency_samples)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    else:
        p50 = p95 = p99 = 0
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': eng.model is not None,
        'scaler_loaded': eng.scaler is not None,
        'shap_available': eng.explainer is not None,
        'uptime_requests': request_count,
        'latency_ms': {
            'p50': round(p50, 2),
            'p95': round(p95, 2),
            'p99': round(p99, 2),
            'target_p95': API_CONFIG.target_p95_latency_ms
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model/info', methods=['GET'])
@require_api_key
def model_info():
    """
    Get model metadata and configuration.
    
    Returns:
        JSON with model information
    """
    eng = get_engine()
    
    return jsonify({
        'model_type': 'XGBoost Classifier',
        'version': '1.0',
        'features_count': len(eng.feature_names) if eng.feature_names else 'unknown',
        'required_inputs': DATASET_CONFIG.sensor_columns,
        'thresholds': {
            'failure_threshold': THRESHOLD_CONFIG.failure_probability_threshold,
            'high_risk': THRESHOLD_CONFIG.high_risk_threshold,
            'medium_risk': THRESHOLD_CONFIG.medium_risk_threshold
        },
        'risk_levels': list(THRESHOLD_CONFIG.risk_levels.keys())
    })


@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    """
    Single prediction endpoint.
    
    Request body (JSON):
    {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
        "machine_id": "MACHINE_001",  // optional
        "timestamp": "2024-01-01T00:00:00"  // optional
    }
    
    Returns:
        JSON with prediction, probability, risk level, and explanation
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'expected_format': {
                    'Air temperature [K]': 'float (required)',
                    'Process temperature [K]': 'float (required)',
                    'Rotational speed [rpm]': 'float (required)',
                    'Torque [Nm]': 'float (required)',
                    'Tool wear [min]': 'float (required)',
                    'machine_id': 'string (optional)',
                    'timestamp': 'ISO format string (optional)'
                }
            }), 400
        
        # Get optional parameters
        include_explanation = data.pop('include_explanation', True)
        explanation_top_n = data.pop('explanation_top_n', 5)
        
        # Make prediction
        eng = get_engine()
        result = eng.predict(
            data,
            include_explanation=include_explanation,
            explanation_top_n=explanation_top_n
        )
        
        if not result['success']:
            return jsonify({
                'error': 'Validation failed',
                'validation_errors': result['errors']
            }), 400
        
        # Format response
        response = {
            'prediction': result['prediction'],
            'failure_probability': result['failure_probability'],
            'no_failure_probability': result['no_failure_probability'],
            'risk_level': result['risk_level'],
            'recommended_action': result['recommended_action'],
            'binary_decision': result['binary_decision'],
            'metadata': result['metadata']
        }
        
        if result.get('explanation'):
            response['explanation'] = result['explanation']
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
@require_api_key
def batch_predict():
    """
    Batch prediction endpoint.
    
    Request body (JSON):
    {
        "records": [
            {
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0
            },
            ...
        ],
        "include_explanation": false  // optional, default false for batch
    }
    
    Returns:
        JSON with batch predictions
    """
    try:
        data = request.get_json()
        
        if not data or 'records' not in data:
            return jsonify({
                'error': 'Invalid request format',
                'expected_format': {
                    'records': 'array of sensor readings',
                    'include_explanation': 'boolean (optional, default false)'
                }
            }), 400
        
        records = data['records']
        include_explanation = data.get('include_explanation', False)
        
        if not isinstance(records, list):
            return jsonify({
                'error': 'records must be an array'
            }), 400
        
        if len(records) > 1000:
            return jsonify({
                'error': 'Batch size limited to 1000 records'
            }), 400
        
        # Make batch prediction
        eng = get_engine()
        result = eng.predict_batch(records, include_explanation)
        
        # Simplify output
        predictions = []
        for pred in result['predictions']:
            if pred['success']:
                predictions.append({
                    'index': pred['_index'],
                    'prediction': pred['prediction'],
                    'failure_probability': pred['failure_probability'],
                    'risk_level': pred['risk_level']
                })
        
        response = {
            'total_records': result['total_records'],
            'successful': result['valid_records'],
            'failed': result['invalid_records'],
            'predictions': predictions,
            'validation_errors': result['validation_errors'],
            'metadata': result['metadata']
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500


@app.route('/validate', methods=['POST'])
def validate_input():
    """
    Validate input data without making prediction.
    
    Useful for testing data format before batch processing.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'records' in data:
            # Batch validation
            valid, invalid = InputValidator.validate_batch(data['records'])
            return jsonify({
                'total': len(data['records']),
                'valid': len(valid),
                'invalid': len(invalid),
                'errors': invalid
            })
        else:
            # Single validation
            is_valid, errors, normalized = InputValidator.validate_single(data)
            return jsonify({
                'is_valid': is_valid,
                'errors': errors,
                'normalized': normalized if is_valid else None
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/thresholds', methods=['GET'])
def get_thresholds():
    """Get current prediction thresholds and risk levels."""
    return jsonify({
        'failure_probability_threshold': THRESHOLD_CONFIG.failure_probability_threshold,
        'risk_levels': THRESHOLD_CONFIG.risk_levels
    })


# =============================================================================
# CLI & MAIN
# =============================================================================
def create_app():
    """Factory function for creating app instance."""
    # Pre-load the model
    get_engine()
    return app


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("PREDICTIVE MAINTENANCE API SERVER")
    logger.info("=" * 60)
    
    # Pre-load model
    get_engine()
    
    logger.info(f"Starting server on {API_CONFIG.host}:{API_CONFIG.port}")
    logger.info(f"Authentication required: {API_CONFIG.require_auth}")
    
    app.run(
        host=API_CONFIG.host,
        port=API_CONFIG.port,
        debug=API_CONFIG.debug
    )
