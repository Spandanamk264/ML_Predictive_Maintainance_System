"""
Baseline Model: Logistic Regression for Predictive Maintenance
==============================================================
Establishes a baseline performance using Logistic Regression.

This module provides:
- Simple baseline model for comparison
- Time-series cross-validation
- Comprehensive metrics evaluation
- Model persistence

Author: IoT Predictive Maintenance Team
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, recall_score, precision_score,
    accuracy_score, roc_auc_score
)
import joblib
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
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
    """
    Load processed data for modeling.
    
    Args:
        path: Optional path to data file
        
    Returns:
        Loaded DataFrame
    """
    if path is None:
        path = str(get_processed_data_path())
    
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info(f"Data loaded: {df.shape}")
    
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'Machine failure'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target for modeling.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (X, y)
    """
    # Get feature columns (exclude target and non-feature columns)
    exclude_cols = DATASET_CONFIG.exclude_columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Handle infinite values
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
    logger.info(f"Positive class rate: {y.mean():.4f}")
    
    return X, y


def train_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    save_path: Optional[str] = None,
    n_splits: int = 5
) -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
    """
    Train and evaluate baseline Logistic Regression model.
    
    Uses time-series cross-validation to prevent data leakage.
    
    Args:
        X: Feature matrix
        y: Target vector
        save_path: Optional path to save model
        n_splits: Number of CV splits
        
    Returns:
        Tuple of (model, scaler, metrics_dict)
    """
    logger.info("=" * 60)
    logger.info("TRAINING BASELINE MODEL (Logistic Regression)")
    logger.info("=" * 60)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_metrics = {
        'f1': [], 'precision': [], 'recall': [], 
        'accuracy': [], 'roc_auc': []
    }
    
    logger.info(f"Running {n_splits}-fold time-series cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=MODEL_CONFIG.baseline_max_iter,
            random_state=MODEL_CONFIG.random_state
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        cv_metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        cv_metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        cv_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        
        try:
            cv_metrics['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
        except ValueError:
            cv_metrics['roc_auc'].append(0.0)
        
        logger.info(f"Fold {fold + 1}: F1={cv_metrics['f1'][-1]:.4f}, "
                   f"Recall={cv_metrics['recall'][-1]:.4f}")
    
    # Summary
    logger.info("-" * 40)
    logger.info("Cross-Validation Results:")
    for metric, values in cv_metrics.items():
        logger.info(f"  {metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")
    
    # Train final model on all data
    logger.info("\nTraining final model on full dataset...")
    final_model = LogisticRegression(
        class_weight='balanced',
        max_iter=MODEL_CONFIG.baseline_max_iter,
        random_state=MODEL_CONFIG.random_state
    )
    final_model.fit(X_scaled, y)
    
    # Final evaluation (on training data - for comparison purposes)
    y_pred_final = final_model.predict(X_scaled)
    
    logger.info("\nFinal Model Performance (Training Set):")
    print(classification_report(y, y_pred_final, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred_final)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Compile metrics
    metrics = {
        'model_name': 'Logistic Regression (Baseline)',
        'cv_scores': cv_metrics,
        'cv_mean_f1': float(np.mean(cv_metrics['f1'])),
        'cv_std_f1': float(np.std(cv_metrics['f1'])),
        'cv_mean_recall': float(np.mean(cv_metrics['recall'])),
        'cv_mean_precision': float(np.mean(cv_metrics['precision'])),
        'cv_mean_roc_auc': float(np.mean(cv_metrics['roc_auc'])),
        'confusion_matrix': cm.tolist(),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'positive_rate': float(y.mean()),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save model and scaler
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path / MODEL_CONFIG.baseline_model_filename
        scaler_path = save_path / MODEL_CONFIG.scaler_filename
        
        joblib.dump(final_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save metrics
        metrics_path = save_path / 'baseline_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Metrics saved to {metrics_path}")
    
    return final_model, scaler, metrics


def evaluate_baseline(
    model: LogisticRegression,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate baseline model on test data.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        metrics['roc_auc'] = 0.0
    
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    
    return metrics


def generate_baseline_report(metrics: Dict[str, Any]) -> str:
    """
    Generate a markdown report for baseline model.
    
    Args:
        metrics: Dictionary of metrics from training
        
    Returns:
        Markdown formatted report string
    """
    report = f"""# Baseline Model Report

## Model: {metrics['model_name']}

**Generated:** {metrics['timestamp']}

---

## Dataset Information

| Metric | Value |
|--------|-------|
| Total Samples | {metrics['n_samples']:,} |
| Features | {metrics['n_features']} |
| Positive Rate | {metrics['positive_rate']:.4f} ({metrics['positive_rate']*100:.2f}%) |

---

## Cross-Validation Results

| Metric | Mean | Std |
|--------|------|-----|
| F1-Score | {metrics['cv_mean_f1']:.4f} | ±{metrics['cv_std_f1']:.4f} |
| Recall | {metrics['cv_mean_recall']:.4f} | - |
| Precision | {metrics['cv_mean_precision']:.4f} | - |
| ROC-AUC | {metrics['cv_mean_roc_auc']:.4f} | - |

---

## Confusion Matrix

```
              Predicted
              Neg    Pos
Actual Neg    {metrics['confusion_matrix'][0][0]:5d}  {metrics['confusion_matrix'][0][1]:5d}
       Pos    {metrics['confusion_matrix'][1][0]:5d}  {metrics['confusion_matrix'][1][1]:5d}
```

---

## Analysis

The baseline Logistic Regression model achieves a mean F1-score of **{metrics['cv_mean_f1']:.4f}** 
on time-series cross-validation. This establishes the baseline performance for comparison 
with more advanced models.

### Key Observations

- Linear model captures basic patterns in sensor data
- Class imbalance handled using balanced class weights
- Time-series CV prevents data leakage from future to past

### Recommendations

1. Compare with tree-based models (Random Forest, XGBoost)
2. Consider additional feature engineering
3. Tune decision threshold for optimal recall

---

*Report generated automatically by Predictive Maintenance Pipeline*
"""
    return report


if __name__ == "__main__":
    # Load data
    df = load_processed_data()
    X, y = prepare_features(df)
    
    # Train baseline
    model, scaler, metrics = train_baseline(
        X, y, 
        save_path=str(MODELS_DIR)
    )
    
    # Generate and save report
    report = generate_baseline_report(metrics)
    report_path = REPORTS_DIR / 'baseline_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print(f"Mean CV F1: {metrics['cv_mean_f1']:.4f}")
    print("=" * 60)
