"""
Advanced Model Comparison & Evaluation
Compares multiple models with comprehensive metrics
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, recall_score, precision_score, 
    roc_auc_score, accuracy_score, matthews_corrcoef
)
import joblib
import os
import json
from datetime import datetime

def load_processed_data(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def prepare_features(df, target_col='Machine failure'):
    # Encode 'Type' if present
    if 'Type' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['Type_encoded'] = le.fit_transform(df['Type'])
    
    feature_cols = [c for c in df.columns if c not in [target_col, 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
    
    # Handle infinite values
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    return X, y

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive model evaluation"""
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_test, y_pred),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])
    
    # Specificity
    metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
    
    return metrics, model

def compare_all_models(X, y, save_path=None):
    """Compare multiple advanced models"""
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-series split (80-20)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Calculate class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Failure rate in train: {y_train.mean():.4f}")
    print(f"Failure rate in test: {y_test.mean():.4f}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}\n")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }
    
    results = []
    trained_models = {}
    
    print("=" * 80)
    print("TRAINING AND EVALUATING ALL MODELS")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n{name}...")
        try:
            metrics, trained_model = evaluate_model(model, X_train, X_test, y_train, y_test, name)
            results.append(metrics)
            trained_models[name] = trained_model
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in metrics else "  ROC-AUC:   N/A")
            print(f"  MCC:       {metrics['mcc']:.4f}")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    print("\n" + "=" * 80)
    print("FINAL COMPARISON (Sorted by F1-Score)")
    print("=" * 80)
    print(results_df[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'mcc']].to_string(index=False))
    
    # Save results
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save comparison CSV
        results_df.to_csv(os.path.join(save_path, 'model_comparison.csv'), index=False)
        
        # Save best model
        best_model_name = results_df.iloc[0]['model_name']
        best_model = trained_models[best_model_name]
        joblib.dump(best_model, os.path.join(save_path, 'best_model.joblib'))
        joblib.dump(scaler, os.path.join(save_path, 'best_scaler.joblib'))
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'best_model': best_model_name,
            'best_f1_score': float(results_df.iloc[0]['f1_score']),
            'dataset_info': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'num_features': X.shape[1],
                'failure_rate_train': float(y_train.mean()),
                'failure_rate_test': float(y_test.mean())
            },
            'all_results': results
        }
        
        with open(os.path.join(save_path, 'evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Results saved to {save_path}")
        print(f"✓ Best model: {best_model_name} (F1: {results_df.iloc[0]['f1_score']:.4f})")
    
    return results_df, trained_models, scaler

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_data.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'models')
    
    print("Loading data...")
    df = load_processed_data(DATA_PATH)
    X, y = prepare_features(df)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Failure rate: {y.mean():.4f}\n")
    
    results_df, models, scaler = compare_all_models(X, y, MODEL_PATH)
