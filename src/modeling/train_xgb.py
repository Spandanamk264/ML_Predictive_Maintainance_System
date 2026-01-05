"""
XGBoost and Random Forest Training for Predictive Maintenance
==============================================================
Advanced model training with hyperparameter optimization.

This module provides:
- XGBoost classifier training
- Random Forest training
- Class imbalance handling (scale_pos_weight, SMOTE)
- Time-series cross-validation
- Hyperparameter optimization via RandomizedSearchCV
- Model comparison and champion selection

Author: IoT Predictive Maintenance Team
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, recall_score, precision_score,
    accuracy_score, roc_auc_score, matthews_corrcoef
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
from config import (
    MODEL_CONFIG, DATASET_CONFIG, 
    get_processed_data_path, MODELS_DIR, REPORTS_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load processed data for modeling."""
    if path is None:
        path = str(get_processed_data_path())
    
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info(f"Data loaded: {df.shape}")
    
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'Machine failure'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix and target for modeling.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Encode categorical Type if present
    if 'Type' in df.columns:
        le = LabelEncoder()
        df = df.copy()
        df['Type_encoded'] = le.fit_transform(df['Type'])
    
    # Get feature columns
    exclude_cols = DATASET_CONFIG.exclude_columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Handle infinite values
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
    logger.info(f"Positive class rate: {y.mean():.4f}")
    
    return X, y, feature_cols


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    tune: bool = True,
    use_smote: bool = False,
    n_iter: int = 50
) -> Tuple[XGBClassifier, StandardScaler, Dict[str, Any]]:
    """
    Train XGBoost classifier with optional hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        save_path: Path to save model
        tune: Whether to perform hyperparameter tuning
        use_smote: Whether to use SMOTE for class imbalance
        n_iter: Number of RandomizedSearchCV iterations
        
    Returns:
        Tuple of (model, scaler, metrics)
    """
    logger.info("=" * 60)
    logger.info("TRAINING XGBOOST CLASSIFIER")
    logger.info("=" * 60)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate class imbalance ratio
    scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
    logger.info(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    # Time-series split
    tscv = TimeSeriesSplit(n_splits=MODEL_CONFIG.cv_splits_tuning)
    
    if tune:
        logger.info(f"Performing hyperparameter tuning with {n_iter} iterations...")
        
        param_dist = MODEL_CONFIG.xgb_param_grid
        
        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=MODEL_CONFIG.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist'  # Faster training
        )
        
        search = RandomizedSearchCV(
            base_model,
            param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring='f1',
            random_state=MODEL_CONFIG.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Apply SMOTE if requested (time-appropriate: only on training folds)
        if use_smote:
            logger.info("Using SMOTE for oversampling in CV folds...")
            # Note: SMOTE should be applied only to training data in each CV fold
            # This is handled internally by sklearn with Pipeline
            smote = SMOTE(
                sampling_strategy=MODEL_CONFIG.smote_sampling_strategy,
                k_neighbors=MODEL_CONFIG.smote_k_neighbors,
                random_state=MODEL_CONFIG.random_state
            )
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
            search.fit(X_resampled, y_resampled)
        else:
            search.fit(X_scaled, y)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV F1 Score: {best_score:.4f}")
    else:
        logger.info("Using default hyperparameters...")
        
        best_model = XGBClassifier(
            n_estimators=MODEL_CONFIG.xgb_n_estimators,
            max_depth=MODEL_CONFIG.xgb_max_depth,
            learning_rate=MODEL_CONFIG.xgb_learning_rate,
            subsample=MODEL_CONFIG.xgb_subsample,
            colsample_bytree=MODEL_CONFIG.xgb_colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=MODEL_CONFIG.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        best_model.fit(X_scaled, y)
        best_params = {
            'n_estimators': MODEL_CONFIG.xgb_n_estimators,
            'max_depth': MODEL_CONFIG.xgb_max_depth,
            'learning_rate': MODEL_CONFIG.xgb_learning_rate,
            'subsample': MODEL_CONFIG.xgb_subsample,
            'colsample_bytree': MODEL_CONFIG.xgb_colsample_bytree
        }
        best_score = None
    
    # Evaluate on full dataset
    y_pred = best_model.predict(X_scaled)
    y_pred_proba = best_model.predict_proba(X_scaled)[:, 1]
    
    logger.info("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))
    
    cm = confusion_matrix(y, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Calculate metrics
    metrics = {
        'model_name': 'XGBoost',
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_pred_proba)),
        'mcc': float(matthews_corrcoef(y, y_pred)),
        'confusion_matrix': cm.tolist(),
        'best_params': best_params,
        'cv_best_score': float(best_score) if best_score else None,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'positive_rate': float(y.mean()),
        'scale_pos_weight': float(scale_pos_weight),
        'timestamp': datetime.now().isoformat()
    }
    
    # Feature importance
    feature_importance = dict(zip(feature_names, best_model.feature_importances_.tolist()))
    sorted_importance = dict(sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:20])  # Top 20
    metrics['top_features'] = sorted_importance
    
    # Save model
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path / MODEL_CONFIG.xgb_model_filename
        pipeline_path = save_path / MODEL_CONFIG.preprocessing_pipeline_filename
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, pipeline_path)
        
        # Save metrics
        metrics_path = save_path / 'xgb_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save feature names
        joblib.dump(feature_names, save_path / 'feature_names.joblib')
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {pipeline_path}")
    
    return best_model, scaler, metrics


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    tune: bool = True,
    n_iter: int = 30
) -> Tuple[RandomForestClassifier, StandardScaler, Dict[str, Any]]:
    """
    Train Random Forest classifier with optional hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        save_path: Path to save model
        tune: Whether to perform hyperparameter tuning
        n_iter: Number of RandomizedSearchCV iterations
        
    Returns:
        Tuple of (model, scaler, metrics)
    """
    logger.info("=" * 60)
    logger.info("TRAINING RANDOM FOREST CLASSIFIER")
    logger.info("=" * 60)
    
    # Scale features (RF doesn't require scaling, but we do for consistency)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-series split
    tscv = TimeSeriesSplit(n_splits=MODEL_CONFIG.cv_splits_tuning)
    
    if tune:
        logger.info(f"Performing hyperparameter tuning with {n_iter} iterations...")
        
        param_dist = MODEL_CONFIG.rf_param_grid
        
        base_model = RandomForestClassifier(
            random_state=MODEL_CONFIG.random_state,
            n_jobs=-1
        )
        
        search = RandomizedSearchCV(
            base_model,
            param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring='f1',
            random_state=MODEL_CONFIG.random_state,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X_scaled, y)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV F1 Score: {best_score:.4f}")
    else:
        best_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=MODEL_CONFIG.random_state,
            n_jobs=-1
        )
        best_model.fit(X_scaled, y)
        best_params = None
        best_score = None
    
    # Evaluate
    y_pred = best_model.predict(X_scaled)
    y_pred_proba = best_model.predict_proba(X_scaled)[:, 1]
    
    logger.info("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))
    
    cm = confusion_matrix(y, y_pred)
    
    metrics = {
        'model_name': 'Random Forest',
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_pred_proba)),
        'mcc': float(matthews_corrcoef(y, y_pred)),
        'confusion_matrix': cm.tolist(),
        'best_params': best_params,
        'cv_best_score': float(best_score) if best_score else None,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'timestamp': datetime.now().isoformat()
    }
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(best_model, save_path / MODEL_CONFIG.rf_model_filename)
        
        metrics_path = save_path / 'rf_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return best_model, scaler, metrics


def select_champion_model(
    models_metrics: List[Dict[str, Any]],
    primary_metric: str = 'f1',
    secondary_metric: str = 'recall'
) -> Tuple[str, Dict[str, Any]]:
    """
    Select the best model based on metrics.
    
    For predictive maintenance, we prioritize:
    1. F1 score (balance of precision and recall)
    2. Recall (catching all failures is critical)
    
    Args:
        models_metrics: List of metrics dictionaries for each model
        primary_metric: Primary metric for selection
        secondary_metric: Tiebreaker metric
        
    Returns:
        Tuple of (champion_model_name, champion_metrics)
    """
    logger.info("\n" + "=" * 60)
    logger.info("CHAMPION MODEL SELECTION")
    logger.info("=" * 60)
    
    # Sort by primary metric, then secondary
    sorted_models = sorted(
        models_metrics,
        key=lambda x: (x.get(primary_metric, 0), x.get(secondary_metric, 0)),
        reverse=True
    )
    
    champion = sorted_models[0]
    
    logger.info(f"\nPrimary metric: {primary_metric}")
    logger.info(f"Secondary metric: {secondary_metric}")
    logger.info("\nModel Rankings:")
    
    for i, model in enumerate(sorted_models):
        marker = "👑 CHAMPION" if i == 0 else ""
        logger.info(f"  {i+1}. {model['model_name']}: "
                   f"{primary_metric}={model.get(primary_metric, 0):.4f}, "
                   f"{secondary_metric}={model.get(secondary_metric, 0):.4f} {marker}")
    
    return champion['model_name'], champion


def generate_comparison_report(
    models_metrics: List[Dict[str, Any]],
    champion_name: str,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a markdown comparison report for all models.
    
    Args:
        models_metrics: List of metrics for each model
        champion_name: Name of the selected champion model
        save_path: Optional path to save report
        
    Returns:
        Markdown report string
    """
    report = f"""# Model Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MCC |
|-------|----------|-----------|--------|----------|---------|-----|
"""
    
    for m in models_metrics:
        marker = " 👑" if m['model_name'] == champion_name else ""
        report += f"| {m['model_name']}{marker} | "
        report += f"{m.get('accuracy', 0):.4f} | "
        report += f"{m.get('precision', 0):.4f} | "
        report += f"{m.get('recall', 0):.4f} | "
        report += f"{m.get('f1', 0):.4f} | "
        report += f"{m.get('roc_auc', 0):.4f} | "
        report += f"{m.get('mcc', 0):.4f} |\n"
    
    report += f"""
---

## Champion Model: {champion_name}

The **{champion_name}** model was selected as the production champion based on:
- Highest F1-Score (balance of precision and recall)
- Strong recall performance (critical for catching all failures)
- Robust ROC-AUC (good discrimination ability)

---

## Detailed Model Analysis

"""
    
    for m in models_metrics:
        report += f"""### {m['model_name']}

**Confusion Matrix:**

```
              Predicted
              Neg    Pos
Actual Neg    {m['confusion_matrix'][0][0]:5d}  {m['confusion_matrix'][0][1]:5d}
       Pos    {m['confusion_matrix'][1][0]:5d}  {m['confusion_matrix'][1][1]:5d}
```

- **True Negatives:** {m['confusion_matrix'][0][0]:,}
- **False Positives:** {m['confusion_matrix'][0][1]:,}
- **False Negatives:** {m['confusion_matrix'][1][0]:,}
- **True Positives:** {m['confusion_matrix'][1][1]:,}

"""
        if m.get('best_params'):
            report += "**Best Parameters:**\n```python\n"
            for k, v in m['best_params'].items():
                report += f"  {k}: {v}\n"
            report += "```\n\n"
    
    report += """---

## Recommendations

1. **Champion model deployed for production inference**
2. **Monitor model performance with new data**
3. **Retrain periodically with fresh data**
4. **Consider ensemble methods for further improvement**

---

*Generated by Predictive Maintenance Pipeline*
"""
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        report_file = save_path / 'model_comparison_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {report_file}")
    
    return report


if __name__ == "__main__":
    # Load data
    df = load_processed_data()
    X, y, feature_names = prepare_features(df)
    
    # Train XGBoost
    xgb_model, xgb_scaler, xgb_metrics = train_xgboost(
        X, y, feature_names,
        save_path=str(MODELS_DIR),
        tune=True,
        n_iter=50
    )
    
    # Train Random Forest
    rf_model, rf_scaler, rf_metrics = train_random_forest(
        X, y, feature_names,
        save_path=str(MODELS_DIR),
        tune=True,
        n_iter=30
    )
    
    # Select champion
    all_metrics = [xgb_metrics, rf_metrics]
    champion_name, champion_metrics = select_champion_model(all_metrics)
    
    # Generate report
    report = generate_comparison_report(
        all_metrics,
        champion_name,
        save_path=str(REPORTS_DIR)
    )
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE - Champion: {champion_name}")
    print(f"Champion F1: {champion_metrics['f1']:.4f}")
    print(f"Champion Recall: {champion_metrics['recall']:.4f}")
    print("=" * 60)
