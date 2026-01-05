"""
Hyperparameter Tuning Module for Predictive Maintenance
========================================================
Advanced hyperparameter optimization using time-series cross-validation.

This module provides:
- Random Forest hyperparameter tuning
- XGBoost hyperparameter tuning
- Time-series aware cross-validation
- Comprehensive model comparison
- Best model selection and export

Author: IoT Predictive Maintenance Team
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    accuracy_score, roc_auc_score, make_scorer
)
import joblib
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, Any, List
import logging
import json
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_CONFIG, DATASET_CONFIG, MODELS_DIR, REPORTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning for predictive maintenance models.
    
    Uses time-series cross-validation to prevent data leakage.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_splits: int = 3
    ):
        """
        Initialize the tuner.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional list of feature names
            n_splits: Number of CV splits
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Calculate class weight for XGBoost
        self.scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
        
        self.results = {}
        
    def tune_random_forest(
        self,
        param_grid: Optional[Dict] = None,
        n_iter: int = 30,
        scoring: str = 'f1'
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        Tune Random Forest hyperparameters.
        
        Args:
            param_grid: Parameter distribution for search
            n_iter: Number of iterations for RandomizedSearchCV
            scoring: Scoring metric
            
        Returns:
            Tuple of (best_model, results_dict)
        """
        logger.info("=" * 60)
        logger.info("TUNING RANDOM FOREST")
        logger.info("=" * 60)
        
        if param_grid is None:
            param_grid = MODEL_CONFIG.rf_param_grid
        
        base_model = RandomForestClassifier(
            random_state=MODEL_CONFIG.random_state,
            n_jobs=-1
        )
        
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=self.tscv,
            scoring=scoring,
            random_state=MODEL_CONFIG.random_state,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        search.fit(self.X_scaled, self.y)
        
        best_model = search.best_estimator_
        
        # Evaluate on full data
        y_pred = best_model.predict(self.X_scaled)
        y_pred_proba = best_model.predict_proba(self.X_scaled)[:, 1]
        
        results = {
            'model_name': 'Random Forest',
            'best_params': search.best_params_,
            'best_cv_score': float(search.best_score_),
            'cv_results_summary': {
                'mean_train_score': float(search.cv_results_['mean_train_score'][search.best_index_]),
                'std_train_score': float(search.cv_results_['std_train_score'][search.best_index_]),
                'mean_test_score': float(search.cv_results_['mean_test_score'][search.best_index_]),
                'std_test_score': float(search.cv_results_['std_test_score'][search.best_index_])
            },
            'full_data_metrics': {
                'accuracy': float(accuracy_score(self.y, y_pred)),
                'precision': float(precision_score(self.y, y_pred, zero_division=0)),
                'recall': float(recall_score(self.y, y_pred, zero_division=0)),
                'f1': float(f1_score(self.y, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(self.y, y_pred_proba))
            },
            'n_iter': n_iter,
            'scoring': scoring
        }
        
        logger.info(f"Best RF Parameters: {search.best_params_}")
        logger.info(f"Best RF CV {scoring}: {search.best_score_:.4f}")
        
        self.results['random_forest'] = results
        
        return best_model, results
    
    def tune_xgboost(
        self,
        param_grid: Optional[Dict] = None,
        n_iter: int = 50,
        scoring: str = 'f1'
    ) -> Tuple[XGBClassifier, Dict[str, Any]]:
        """
        Tune XGBoost hyperparameters.
        
        Args:
            param_grid: Parameter distribution for search
            n_iter: Number of iterations for RandomizedSearchCV
            scoring: Scoring metric
            
        Returns:
            Tuple of (best_model, results_dict)
        """
        logger.info("=" * 60)
        logger.info("TUNING XGBOOST")
        logger.info("=" * 60)
        
        if param_grid is None:
            param_grid = MODEL_CONFIG.xgb_param_grid
        
        base_model = XGBClassifier(
            scale_pos_weight=self.scale_pos_weight,
            random_state=MODEL_CONFIG.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist'
        )
        
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=self.tscv,
            scoring=scoring,
            random_state=MODEL_CONFIG.random_state,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        search.fit(self.X_scaled, self.y)
        
        best_model = search.best_estimator_
        
        # Evaluate on full data
        y_pred = best_model.predict(self.X_scaled)
        y_pred_proba = best_model.predict_proba(self.X_scaled)[:, 1]
        
        results = {
            'model_name': 'XGBoost',
            'best_params': search.best_params_,
            'best_cv_score': float(search.best_score_),
            'cv_results_summary': {
                'mean_train_score': float(search.cv_results_['mean_train_score'][search.best_index_]),
                'std_train_score': float(search.cv_results_['std_train_score'][search.best_index_]),
                'mean_test_score': float(search.cv_results_['mean_test_score'][search.best_index_]),
                'std_test_score': float(search.cv_results_['std_test_score'][search.best_index_])
            },
            'full_data_metrics': {
                'accuracy': float(accuracy_score(self.y, y_pred)),
                'precision': float(precision_score(self.y, y_pred, zero_division=0)),
                'recall': float(recall_score(self.y, y_pred, zero_division=0)),
                'f1': float(f1_score(self.y, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(self.y, y_pred_proba))
            },
            'scale_pos_weight': float(self.scale_pos_weight),
            'n_iter': n_iter,
            'scoring': scoring
        }
        
        logger.info(f"Best XGB Parameters: {search.best_params_}")
        logger.info(f"Best XGB CV {scoring}: {search.best_score_:.4f}")
        
        self.results['xgboost'] = results
        
        return best_model, results
    
    def compare_models(
        self,
        save_path: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compare all tuned models and select the best.
        
        Args:
            save_path: Path to save models and results
            
        Returns:
            Tuple of (comparison_dict, best_model_info)
        """
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        # Tune both models if not already done
        if 'random_forest' not in self.results:
            rf_model, _ = self.tune_random_forest()
        else:
            rf_model = None
            
        if 'xgboost' not in self.results:
            xgb_model, _ = self.tune_xgboost()
        else:
            xgb_model = None
        
        # Compare
        comparison = {}
        
        for name, results in self.results.items():
            metrics = results['full_data_metrics']
            comparison[name] = {
                'cv_score': results['best_cv_score'],
                **metrics
            }
            
            logger.info(f"\n{results['model_name']}:")
            logger.info(f"  CV F1: {results['best_cv_score']:.4f}")
            logger.info(f"  Full Data - F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Select best model based on F1, then Recall
        best_name = max(
            comparison.keys(),
            key=lambda x: (comparison[x]['f1'], comparison[x]['recall'])
        )
        
        best_info = {
            'name': best_name,
            'metrics': comparison[best_name],
            'params': self.results[best_name]['best_params']
        }
        
        logger.info(f"\n🏆 Best Model: {best_name}")
        
        # Save results
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save comparison
            comparison_path = save_path / 'tuning_comparison.json'
            with open(comparison_path, 'w') as f:
                json.dump({
                    'comparison': comparison,
                    'best_model': best_info,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Comparison saved to {comparison_path}")
        
        return comparison, best_info


def tune_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function to tune all models.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: Optional feature names
        save_path: Path to save results
        
    Returns:
        Tuple of (comparison, best_model_info)
    """
    tuner = HyperparameterTuner(X, y, feature_names)
    
    # Tune RF
    rf_model, _ = tuner.tune_random_forest(n_iter=30)
    
    # Tune XGBoost
    xgb_model, _ = tuner.tune_xgboost(n_iter=50)
    
    # Compare
    comparison, best_info = tuner.compare_models(save_path)
    
    # Save best model
    if save_path:
        save_path = Path(save_path)
        
        if best_info['name'] == 'xgboost':
            joblib.dump(xgb_model, save_path / 'tuned_xgb_model.joblib')
            joblib.dump(tuner.scaler, save_path / 'tuned_scaler.joblib')
        else:
            joblib.dump(rf_model, save_path / 'tuned_rf_model.joblib')
            joblib.dump(tuner.scaler, save_path / 'tuned_scaler.joblib')
    
    return comparison, best_info


if __name__ == "__main__":
    # Load data
    from train_xgb import load_processed_data, prepare_features
    
    df = load_processed_data()
    X, y, feature_names = prepare_features(df)
    
    # Tune all models
    comparison, best_info = tune_models(
        X, y, feature_names,
        save_path=str(MODELS_DIR)
    )
    
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print(f"Best Model: {best_info['name']}")
    print(f"Best F1: {best_info['metrics']['f1']:.4f}")
    print("=" * 60)
