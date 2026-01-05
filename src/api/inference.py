"""
Inference Module for Predictive Maintenance
============================================
Production-ready inference engine with full pipeline support.

This module provides:
- Model and pipeline loading at startup
- Raw sensor input preprocessing
- Feature engineering for inference
- Prediction with probability outputs
- SHAP-based explanation generation
- Input validation and error handling
- Batch prediction support

Author: IoT Predictive Maintenance Team
"""
import numpy as np
import pandas as pd
import joblib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import json

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, DATASET_CONFIG, FEATURE_CONFIG,
    THRESHOLD_CONFIG, EXPLAINABILITY_CONFIG,
    get_processed_data_path
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InputValidator:
    """Validates and normalizes input data for inference."""
    
    # Expected input schema
    REQUIRED_FIELDS = DATASET_CONFIG.sensor_columns
    
    # Valid ranges for sensors
    VALID_RANGES = {
        'Air temperature [K]': (FEATURE_CONFIG.min_temperature_kelvin, FEATURE_CONFIG.max_temperature_kelvin),
        'Process temperature [K]': (FEATURE_CONFIG.min_temperature_kelvin, FEATURE_CONFIG.max_temperature_kelvin),
        'Rotational speed [rpm]': (FEATURE_CONFIG.min_rpm, FEATURE_CONFIG.max_rpm),
        'Torque [Nm]': (FEATURE_CONFIG.min_torque, FEATURE_CONFIG.max_torque),
        'Tool wear [min]': (FEATURE_CONFIG.min_tool_wear, FEATURE_CONFIG.max_tool_wear)
    }
    
    @classmethod
    def validate_single(cls, data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate a single input record.
        
        Args:
            data: Input dictionary with sensor readings
            
        Returns:
            Tuple of (is_valid, error_messages, normalized_data)
        """
        errors = []
        normalized = {}
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                errors.append(f"Missing required field: '{field}'")
                continue
            
            value = data[field]
            
            # Check type
            try:
                value = float(value)
            except (ValueError, TypeError):
                errors.append(f"Invalid type for '{field}': expected numeric, got {type(value).__name__}")
                continue
            
            # Check range
            if field in cls.VALID_RANGES:
                min_val, max_val = cls.VALID_RANGES[field]
                if value < min_val or value > max_val:
                    errors.append(f"Out of range for '{field}': {value} (expected {min_val}-{max_val})")
            
            normalized[field] = value
        
        # Add optional fields
        for field in ['machine_id', 'timestamp', 'Type']:
            if field in data:
                normalized[field] = data[field]
        
        is_valid = len(errors) == 0
        return is_valid, errors, normalized
    
    @classmethod
    def validate_batch(cls, records: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate a batch of input records.
        
        Args:
            records: List of input dictionaries
            
        Returns:
            Tuple of (valid_records, invalid_records_with_errors)
        """
        valid = []
        invalid = []
        
        for i, record in enumerate(records):
            is_valid, errors, normalized = cls.validate_single(record)
            
            if is_valid:
                normalized['_index'] = i
                valid.append(normalized)
            else:
                invalid.append({
                    '_index': i,
                    'original': record,
                    'errors': errors
                })
        
        return valid, invalid


class InferenceEngine:
    """
    Production inference engine for predictive maintenance.
    
    Handles the complete inference pipeline from raw sensor readings
    to predictions with explanations.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        load_explainer: bool = True
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to model directory
            load_explainer: Whether to load SHAP explainer
        """
        self.model_path = Path(model_path) if model_path else MODELS_DIR
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.explainer = None
        self._load_explainer = load_explainer
        
        self._load_models()
        
    def _load_models(self):
        """Load model and preprocessing artifacts."""
        logger.info(f"Loading models from {self.model_path}...")
        
        # Load model
        model_file = self.model_path / 'final_xgb_model.joblib'
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        self.model = joblib.load(model_file)
        logger.info("Model loaded successfully")
        
        # Load scaler
        scaler_file = self.model_path / 'preprocessing_pipeline.joblib'
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.info("Scaler loaded successfully")
        else:
            logger.warning("Scaler not found, using identity transform")
            self.scaler = None
        
        # Load feature names
        features_file = self.model_path / 'feature_names.joblib'
        if features_file.exists():
            self.feature_names = joblib.load(features_file)
            logger.info(f"Feature names loaded: {len(self.feature_names)} features")
        else:
            logger.warning("Feature names not found")
            self.feature_names = None
        
        # Load SHAP explainer
        if self._load_explainer:
            try:
                import shap
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP explainer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer: {e}")
                self.explainer = None
    
    def preprocess_raw_input(
        self,
        sensor_data: Dict[str, float],
        history: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Preprocess raw sensor data into feature vector.
        
        For full feature engineering (lag, rolling, etc.), historical
        data is required. Without history, only basic features are computed.
        
        Args:
            sensor_data: Dictionary of current sensor readings
            history: Optional DataFrame of historical readings
            
        Returns:
            Feature vector as numpy array
        """
        # Basic features from current reading
        basic_features = [
            sensor_data.get('Air temperature [K]', 0),
            sensor_data.get('Process temperature [K]', 0),
            sensor_data.get('Rotational speed [rpm]', 0),
            sensor_data.get('Torque [Nm]', 0),
            sensor_data.get('Tool wear [min]', 0),
        ]
        
        # Interaction features
        temp_diff = sensor_data.get('Process temperature [K]', 0) - sensor_data.get('Air temperature [K]', 0)
        power = sensor_data.get('Torque [Nm]', 0) * sensor_data.get('Rotational speed [rpm]', 0)
        
        interaction_features = [temp_diff, power]
        
        if history is not None and len(history) >= 3:
            # Compute lag and rolling features from history
            extended_features = self._compute_extended_features(sensor_data, history)
            features = extended_features
        else:
            # Without history, pad with zeros or use basic features only
            # This will work if model was trained on basic features
            if self.feature_names and len(self.feature_names) > 7:
                # Need to create a feature vector of expected size
                # Use zeros for missing temporal features
                features = np.zeros(len(self.feature_names))
                features[:5] = basic_features
                features[5:7] = interaction_features
            else:
                features = basic_features + interaction_features
        
        return np.array(features).reshape(1, -1)
    
    def _compute_extended_features(
        self,
        current: Dict[str, float],
        history: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute full feature set using current reading and history.
        
        Args:
            current: Current sensor readings
            history: Historical readings DataFrame
            
        Returns:
            Complete feature vector
        """
        # Append current reading to history
        current_df = pd.DataFrame([current])
        df = pd.concat([history, current_df], ignore_index=True)
        
        sensor_cols = DATASET_CONFIG.sensor_columns
        features = []
        
        # Original features
        for col in sensor_cols:
            features.append(current.get(col, 0))
        
        # Lag features (from history)
        for col in sensor_cols:
            for lag in [1, 2, 3]:
                if len(history) >= lag and col in history.columns:
                    features.append(history[col].iloc[-lag])
                else:
                    features.append(0)
        
        # Rolling features (simplified - using history mean/std)
        for col in sensor_cols:
            if col in history.columns:
                features.append(history[col].tail(6).mean())  # 1h mean
                features.append(history[col].tail(6).std())   # 1h std
                features.append(history[col].tail(24).mean()) # 4h mean
                features.append(history[col].tail(24).std())  # 4h std
            else:
                features.extend([0, 0, 0, 0])
        
        # Interaction features
        features.append(current.get('Process temperature [K]', 0) - current.get('Air temperature [K]', 0))
        features.append(current.get('Torque [Nm]', 0) * current.get('Rotational speed [rpm]', 0))
        
        return np.array(features)
    
    def predict(
        self,
        sensor_data: Dict[str, float],
        include_explanation: bool = True,
        explanation_top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Make prediction with optional explanation.
        
        Args:
            sensor_data: Dictionary of sensor readings
            include_explanation: Whether to include SHAP explanation
            explanation_top_n: Number of top features in explanation
            
        Returns:
            Prediction result dictionary
        """
        start_time = datetime.now()
        
        # Validate input
        is_valid, errors, normalized = InputValidator.validate_single(sensor_data)
        if not is_valid:
            return {
                'success': False,
                'errors': errors,
                'prediction': None
            }
        
        # Preprocess
        features = self.preprocess_raw_input(normalized)
        
        # Scale if scaler available
        if self.scaler is not None:
            try:
                features_scaled = self.scaler.transform(features)
            except ValueError as e:
                # Feature count mismatch - use basic prediction
                logger.warning(f"Scaler mismatch: {e}, using unscaled features")
                features_scaled = features
        else:
            features_scaled = features
        
        # Predict
        try:
            prediction = int(self.model.predict(features_scaled)[0])
            probabilities = self.model.predict_proba(features_scaled)[0]
            failure_prob = float(probabilities[1])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'errors': [f"Prediction failed: {str(e)}"],
                'prediction': None
            }
        
        # Determine risk level
        risk_level = self._get_risk_level(failure_prob)
        
        result = {
            'success': True,
            'prediction': prediction,
            'failure_probability': failure_prob,
            'no_failure_probability': float(probabilities[0]),
            'risk_level': risk_level['level'],
            'recommended_action': risk_level['action'],
            'binary_decision': 'FAILURE_PREDICTED' if prediction == 1 else 'NORMAL',
        }
        
        # Add SHAP explanation
        if include_explanation and self.explainer is not None:
            try:
                explanation = self._generate_explanation(
                    features_scaled, 
                    failure_prob,
                    explanation_top_n
                )
                result['explanation'] = explanation
            except Exception as e:
                logger.warning(f"Explanation generation failed: {e}")
                result['explanation'] = None
        
        # Metadata
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        result['metadata'] = {
            'inference_time_ms': round(inference_time, 2),
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0',
            'machine_id': normalized.get('machine_id'),
            'input_timestamp': normalized.get('timestamp')
        }
        
        return result
    
    def _get_risk_level(self, probability: float) -> Dict[str, str]:
        """Determine risk level and action from probability."""
        for level, config in THRESHOLD_CONFIG.risk_levels.items():
            if config['min'] <= probability < config['max']:
                return {'level': level.upper(), 'action': config['action']}
        
        return {'level': 'UNKNOWN', 'action': 'review_required'}
    
    def _generate_explanation(
        self,
        features: np.ndarray,
        failure_prob: float,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """Generate SHAP-based explanation."""
        shap_values = self.explainer.shap_values(features)
        
        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_vals = shap_values[0]
        
        # Get feature names
        if self.feature_names:
            names = self.feature_names[:len(shap_vals)]
        else:
            names = [f'feature_{i}' for i in range(len(shap_vals))]
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        
        contributions = []
        for i in sorted_idx[:top_n]:
            contributions.append({
                'feature': names[i] if i < len(names) else f'feature_{i}',
                'shap_value': float(shap_vals[i]),
                'direction': 'increases_risk' if shap_vals[i] > 0 else 'decreases_risk'
            })
        
        # Generate text summary
        text_summary = self._generate_text_explanation(contributions, failure_prob)
        
        return {
            'top_factors': contributions,
            'base_value': float(self.explainer.expected_value[1]) if isinstance(self.explainer.expected_value, list) else float(self.explainer.expected_value),
            'text_summary': text_summary
        }
    
    def _generate_text_explanation(
        self,
        contributions: List[Dict],
        failure_prob: float
    ) -> str:
        """Generate human-readable explanation text."""
        risk_level = 'HIGH' if failure_prob > 0.5 else 'MEDIUM' if failure_prob > 0.3 else 'LOW'
        
        text = f"Failure risk assessment: {risk_level} ({failure_prob:.1%}). "
        
        increasing = [c for c in contributions if c['direction'] == 'increases_risk']
        decreasing = [c for c in contributions if c['direction'] == 'decreases_risk']
        
        if increasing:
            factors = [c['feature'] for c in increasing[:2]]
            text += f"Primary risk factors: {', '.join(factors)}. "
        
        if decreasing:
            factors = [c['feature'] for c in decreasing[:2]]
            text += f"Protective factors: {', '.join(factors)}."
        
        return text
    
    def predict_batch(
        self,
        records: List[Dict[str, float]],
        include_explanation: bool = False
    ) -> Dict[str, Any]:
        """
        Make batch predictions.
        
        Args:
            records: List of sensor reading dictionaries
            include_explanation: Whether to include explanations
            
        Returns:
            Batch prediction results
        """
        start_time = datetime.now()
        
        # Validate all inputs
        valid_records, invalid_records = InputValidator.validate_batch(records)
        
        predictions = []
        
        for record in valid_records:
            result = self.predict(
                record,
                include_explanation=include_explanation,
                explanation_top_n=3
            )
            result['_index'] = record['_index']
            predictions.append(result)
        
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'success': True,
            'total_records': len(records),
            'valid_records': len(valid_records),
            'invalid_records': len(invalid_records),
            'predictions': predictions,
            'validation_errors': invalid_records,
            'metadata': {
                'batch_inference_time_ms': round(inference_time, 2),
                'avg_time_per_record_ms': round(inference_time / len(records), 2) if records else 0,
                'timestamp': datetime.now().isoformat()
            }
        }


# Singleton instance for API usage
_engine: Optional[InferenceEngine] = None


def get_engine(model_path: Optional[str] = None) -> InferenceEngine:
    """Get or create singleton inference engine."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine(model_path)
    return _engine


def predict_single(
    sensor_data: Dict[str, float],
    include_explanation: bool = True
) -> Dict[str, Any]:
    """Convenience function for single prediction."""
    engine = get_engine()
    return engine.predict(sensor_data, include_explanation)


def predict_batch(
    records: List[Dict[str, float]],
    include_explanation: bool = False
) -> Dict[str, Any]:
    """Convenience function for batch prediction."""
    engine = get_engine()
    return engine.predict_batch(records, include_explanation)


if __name__ == "__main__":
    # Example usage
    engine = InferenceEngine()
    
    # Sample prediction
    sample_input = {
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1551,
        'Torque [Nm]': 42.8,
        'Tool wear [min]': 100,
        'machine_id': 'MACHINE_001'
    }
    
    print("=" * 60)
    print("SINGLE PREDICTION EXAMPLE")
    print("=" * 60)
    result = engine.predict(sample_input)
    print(json.dumps(result, indent=2, default=str))
    
    # Batch prediction example
    print("\n" + "=" * 60)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 60)
    batch_input = [
        {
            'Air temperature [K]': 298.1,
            'Process temperature [K]': 308.6,
            'Rotational speed [rpm]': 1551,
            'Torque [Nm]': 42.8,
            'Tool wear [min]': 0
        },
        {
            'Air temperature [K]': 300.5,
            'Process temperature [K]': 315.2,
            'Rotational speed [rpm]': 1400,
            'Torque [Nm]': 60.0,
            'Tool wear [min]': 200
        },
        {
            # Invalid record - missing fields
            'Air temperature [K]': 298.0
        }
    ]
    
    batch_result = engine.predict_batch(batch_input, include_explanation=False)
    print(json.dumps(batch_result, indent=2, default=str))
