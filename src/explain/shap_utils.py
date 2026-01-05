"""
SHAP Explainability Module for Predictive Maintenance
======================================================
Comprehensive model interpretability using SHAP values.

This module provides:
- SHAP value computation for tree-based models
- Global feature importance (summary plots, bar plots)
- Local explanations (force plots, decision plots)
- Domain validation layer for SHAP patterns
- Human-readable textual explanations
- Export utilities for plots and artifacts

Author: IoT Predictive Maintenance Team
"""
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server deployment
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging
import json

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EXPLAINABILITY_CONFIG, MODELS_DIR, REPORTS_DIR, 
    SHAP_PLOTS_DIR, ARTIFACTS_DIR, DATASET_CONFIG,
    get_processed_data_path
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainability for predictive maintenance.
    
    Provides global and local explanations with domain validation.
    
    Attributes:
        model: Trained model (tree-based)
        X: Feature matrix
        feature_names: List of feature names
        shap_values: Computed SHAP values
        expected_value: Base value from explainer
    """
    
    def __init__(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained tree-based model
            X: Feature matrix for explanation
            feature_names: List of feature names
        """
        self.model = model
        self.X = X
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        logger.info("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(model)
        
        logger.info("Computing SHAP values...")
        self._compute_shap_values()
        
    def _compute_shap_values(self):
        """Compute SHAP values for the dataset."""
        shap_values = self.explainer.shap_values(self.X)
        
        # Handle both binary and multi-class outputs
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Binary classification - use values for positive class
            self.shap_values = shap_values[1]
            self.expected_value = self.explainer.expected_value[1]
        else:
            self.shap_values = shap_values
            self.expected_value = self.explainer.expected_value
        
        logger.info(f"SHAP values computed: shape {self.shap_values.shape}")
    
    def summary_plot(
        self,
        save_path: Optional[str] = None,
        max_display: int = 20
    ) -> str:
        """
        Generate global SHAP summary plot.
        
        Shows feature importance with impact direction.
        
        Args:
            save_path: Directory to save plot
            max_display: Maximum features to display
            
        Returns:
            Path to saved plot
        """
        logger.info("Generating SHAP summary plot...")
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            self.shap_values,
            self.X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Summary Plot - Feature Impact on Failure Prediction', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plot_path = save_path / 'shap_summary.png'
            plt.savefig(plot_path, dpi=EXPLAINABILITY_CONFIG.plot_dpi, bbox_inches='tight')
            logger.info(f"Summary plot saved to {plot_path}")
            plt.close()
            return str(plot_path)
        
        plt.close()
        return ""
    
    def bar_plot(
        self,
        save_path: Optional[str] = None,
        max_display: int = 15
    ) -> str:
        """
        Generate global feature importance bar plot.
        
        Args:
            save_path: Directory to save plot
            max_display: Maximum features to display
            
        Returns:
            Path to saved plot
        """
        logger.info("Generating SHAP bar plot...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.X,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance - Mean Absolute Impact', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plot_path = save_path / 'shap_bar.png'
            plt.savefig(plot_path, dpi=EXPLAINABILITY_CONFIG.plot_dpi, bbox_inches='tight')
            logger.info(f"Bar plot saved to {plot_path}")
            plt.close()
            return str(plot_path)
        
        plt.close()
        return ""
    
    def decision_plot(
        self,
        indices: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        max_samples: int = 50
    ) -> str:
        """
        Generate decision plot for multiple predictions.
        
        Shows how features contribute to each prediction.
        
        Args:
            indices: Specific sample indices to plot
            save_path: Directory to save plot
            max_samples: Maximum samples to display
            
        Returns:
            Path to saved plot
        """
        logger.info("Generating SHAP decision plot...")
        
        if indices is None:
            indices = list(range(min(max_samples, len(self.X))))
        
        plt.figure(figsize=(12, 10))
        shap.decision_plot(
            self.expected_value,
            self.shap_values[indices],
            feature_names=self.feature_names,
            show=False
        )
        plt.title('SHAP Decision Plot - Prediction Path', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plot_path = save_path / 'shap_decision.png'
            plt.savefig(plot_path, dpi=EXPLAINABILITY_CONFIG.plot_dpi, bbox_inches='tight')
            logger.info(f"Decision plot saved to {plot_path}")
            plt.close()
            return str(plot_path)
        
        plt.close()
        return ""
    
    def waterfall_plot(
        self,
        sample_idx: int,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate waterfall plot for a single prediction.
        
        Args:
            sample_idx: Index of sample to explain
            save_path: Directory to save plot
            
        Returns:
            Path to saved plot
        """
        logger.info(f"Generating waterfall plot for sample {sample_idx}...")
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.expected_value,
            data=self.X[sample_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall - Sample {sample_idx}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plot_path = save_path / f'shap_waterfall_{sample_idx}.png'
            plt.savefig(plot_path, dpi=EXPLAINABILITY_CONFIG.plot_dpi, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {plot_path}")
            plt.close()
            return str(plot_path)
        
        plt.close()
        return ""
    
    def force_plot_html(
        self,
        sample_idx: int,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate interactive force plot as HTML.
        
        Args:
            sample_idx: Index of sample to explain
            save_path: Directory to save HTML
            
        Returns:
            Path to saved HTML file
        """
        logger.info(f"Generating force plot for sample {sample_idx}...")
        
        shap.initjs()
        force_plot = shap.force_plot(
            self.expected_value,
            self.shap_values[sample_idx],
            self.X[sample_idx],
            feature_names=self.feature_names
        )
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            html_path = save_path / f'force_plot_{sample_idx}.html'
            shap.save_html(str(html_path), force_plot)
            logger.info(f"Force plot saved to {html_path}")
            return str(html_path)
        
        return ""
    
    def get_feature_importance(
        self,
        top_n: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get ranked feature importances based on mean absolute SHAP values.
        
        Args:
            top_n: Number of top features to return (None for all)
            
        Returns:
            Dictionary of feature names to importance values
        """
        # Mean absolute SHAP value per feature
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create sorted dictionary
        importance_dict = dict(zip(self.feature_names, importance))
        sorted_importance = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        if top_n:
            sorted_importance = dict(list(sorted_importance.items())[:top_n])
        
        return sorted_importance
    
    def get_local_explanation(
        self,
        sample_idx: int,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Get local explanation for a single prediction.
        
        Args:
            sample_idx: Index of sample to explain
            top_n: Number of top features to include
            
        Returns:
            Dictionary with explanation details
        """
        shap_vals = self.shap_values[sample_idx]
        feature_vals = self.X[sample_idx]
        
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        
        contributions = []
        for i in sorted_idx[:top_n]:
            contributions.append({
                'feature': self.feature_names[i],
                'value': float(feature_vals[i]),
                'shap_value': float(shap_vals[i]),
                'direction': 'increases' if shap_vals[i] > 0 else 'decreases'
            })
        
        total_effect = float(np.sum(shap_vals))
        predicted_value = self.expected_value + total_effect
        
        return {
            'sample_idx': sample_idx,
            'base_value': float(self.expected_value),
            'predicted_value': float(predicted_value),
            'total_effect': total_effect,
            'top_contributions': contributions
        }
    
    def get_textual_explanation(
        self,
        sample_idx: int,
        prediction_prob: float,
        top_n: int = 3
    ) -> str:
        """
        Generate human-readable explanation for a prediction.
        
        Args:
            sample_idx: Index of sample to explain
            prediction_prob: Predicted failure probability
            top_n: Number of top factors to mention
            
        Returns:
            Human-readable explanation string
        """
        local_exp = self.get_local_explanation(sample_idx, top_n)
        
        risk_level = "HIGH" if prediction_prob > 0.5 else "MEDIUM" if prediction_prob > 0.3 else "LOW"
        
        explanation = f"**Risk Level: {risk_level}** (Failure Probability: {prediction_prob:.1%})\n\n"
        explanation += "**Key Factors:**\n"
        
        for contrib in local_exp['top_contributions']:
            direction = "↑ increases" if contrib['direction'] == 'increases' else "↓ decreases"
            explanation += f"- **{contrib['feature']}** = {contrib['value']:.2f} → {direction} failure risk by {abs(contrib['shap_value']):.3f}\n"
        
        explanation += f"\n**Technical Note:** Base failure rate is {local_exp['base_value']:.3f}, "
        explanation += f"combined factors shift this by {local_exp['total_effect']:.3f}.\n"
        
        return explanation


class DomainValidator:
    """
    Validate SHAP patterns against domain knowledge.
    
    Checks whether model behavior aligns with manufacturing intuition.
    """
    
    # Expected relationships (positive = higher values increase failure risk)
    EXPECTED_DIRECTIONS = EXPLAINABILITY_CONFIG.expected_shap_directions
    
    @classmethod
    def validate_shap_patterns(
        cls,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Validate SHAP patterns against domain expectations.
        
        Args:
            shap_values: SHAP values matrix
            feature_values: Original feature values
            feature_names: List of feature names
            
        Returns:
            Validation report dictionary
        """
        logger.info("Validating SHAP patterns against domain knowledge...")
        
        anomalies = []
        validations = []
        
        for i, feature in enumerate(feature_names):
            # Check if this feature has an expected direction
            for pattern, expected_dir in cls.EXPECTED_DIRECTIONS.items():
                # Skip derivative features for simple directional checks as they can be complex
                if any(x in feature for x in ['_lag', '_roll', '_ema', '_roc']):
                    continue
                    
                if pattern in feature:
                    # Compute correlation between feature values and SHAP values
                    correlation = np.corrcoef(feature_values[:, i], shap_values[:, i])[0, 1]
                    
                    if np.isnan(correlation):
                        continue
                    
                    # Check if direction matches expectation
                    actual_dir = 1 if correlation > 0 else -1
                    
                    if expected_dir != 0:  # If we have a directional expectation
                        matches = (actual_dir == expected_dir)
                        
                        result = {
                            'feature': feature,
                            'expected_direction': 'positive' if expected_dir > 0 else 'negative',
                            'actual_correlation': float(correlation),
                            'matches_expectation': matches
                        }
                        
                        if matches:
                            validations.append(result)
                        else:
                            # Check if the deviation is significant
                            if abs(correlation) > EXPLAINABILITY_CONFIG.shap_anomaly_threshold:
                                result['severity'] = 'HIGH' if abs(correlation) > 0.5 else 'MEDIUM'
                                anomalies.append(result)
        
        # Summary
        report = {
            'total_features_checked': len(validations) + len(anomalies),
            'features_aligned': len(validations),
            'anomalies_detected': len(anomalies),
            'anomaly_rate': len(anomalies) / (len(validations) + len(anomalies) + 0.001),
            'validations': validations,
            'anomalies': anomalies,
            'timestamp': datetime.now().isoformat()
        }
        
        if anomalies:
            logger.warning(f"Found {len(anomalies)} potential SHAP anomalies")
            for a in anomalies:
                logger.warning(f"  - {a['feature']}: expected {a['expected_direction']}, "
                             f"actual correlation {a['actual_correlation']:.3f}")
        else:
            logger.info("All checked patterns align with domain expectations ✓")
        
        return report
    
    @classmethod
    def generate_validation_report(cls, report: Dict[str, Any]) -> str:
        """
        Generate markdown report for domain validation.
        
        Args:
            report: Validation report dictionary
            
        Returns:
            Markdown formatted report
        """
        md = """# SHAP Domain Validation Report

**Generated:** {}

---

## Summary

| Metric | Value |
|--------|-------|
| Features Checked | {} |
| Aligned with Domain | {} |
| Anomalies Detected | {} |
| Anomaly Rate | {:.1%} |

---

## Validation Status: {}

""".format(
            report['timestamp'],
            report['total_features_checked'],
            report['features_aligned'],
            report['anomalies_detected'],
            report['anomaly_rate'],
            "✅ PASSED" if report['anomaly_rate'] < 0.2 else "⚠️ REVIEW NEEDED"
        )
        
        if report['anomalies']:
            md += "## Anomalies Requiring Investigation\n\n"
            md += "| Feature | Expected | Actual Correlation | Severity |\n"
            md += "|---------|----------|-------------------|----------|\n"
            
            for a in report['anomalies']:
                md += f"| {a['feature']} | {a['expected_direction']} | "
                md += f"{a['actual_correlation']:.3f} | {a.get('severity', 'LOW')} |\n"
            
            md += "\n**Recommended Actions:**\n"
            md += "1. Review feature engineering for anomalous features\n"
            md += "2. Check for data quality issues\n"
            md += "3. Consult domain experts for validation\n\n"
        
        if report['validations']:
            md += "## Validated Patterns\n\n"
            md += "| Feature | Expected | Correlation | Status |\n"
            md += "|---------|----------|-------------|--------|\n"
            
            for v in report['validations'][:10]:  # Show top 10
                md += f"| {v['feature']} | {v['expected_direction']} | "
                md += f"{v['actual_correlation']:.3f} | ✅ |\n"
        
        md += "\n---\n\n*Generated by Predictive Maintenance XAI Pipeline*\n"
        
        return md


def generate_all_explanations(
    model_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate all SHAP explanations and plots.
    
    Args:
        model_path: Path to model directory
        data_path: Path to processed data
        output_path: Path for output plots
        
    Returns:
        Dictionary of generated artifact paths
    """
    logger.info("=" * 60)
    logger.info("GENERATING SHAP EXPLANATIONS")
    logger.info("=" * 60)
    
    # Set defaults
    if model_path is None:
        model_path = str(MODELS_DIR)
    if data_path is None:
        data_path = str(get_processed_data_path())
    if output_path is None:
        output_path = str(SHAP_PLOTS_DIR)
    
    # Load model and scaler
    model = joblib.load(os.path.join(model_path, 'final_xgb_model.joblib'))
    scaler = joblib.load(os.path.join(model_path, 'preprocessing_pipeline.joblib'))
    
    # Try to load feature names
    try:
        feature_names = joblib.load(os.path.join(model_path, 'feature_names.joblib'))
    except FileNotFoundError:
        feature_names = None
    
    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Encode Type if present
    if 'Type' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['Type_encoded'] = le.fit_transform(df['Type'])
    
    # Get features
    exclude_cols = DATASET_CONFIG.exclude_columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Handle infinite values
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_clean[feature_cols].values
    X_scaled = scaler.transform(X)
    
    if feature_names is None:
        feature_names = feature_cols
    
    # Create explainer
    explainer = SHAPExplainer(model, X_scaled, feature_names)
    
    # Generate plots
    artifacts = {}
    
    artifacts['summary_plot'] = explainer.summary_plot(output_path)
    artifacts['bar_plot'] = explainer.bar_plot(output_path)
    artifacts['decision_plot'] = explainer.decision_plot(save_path=output_path)
    
    # Generate waterfall for first high-risk sample
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    high_risk_idx = np.argmax(y_pred_proba)
    artifacts['waterfall_plot'] = explainer.waterfall_plot(high_risk_idx, output_path)
    
    # Get feature importance
    importance = explainer.get_feature_importance(top_n=20)
    # Convert numpy float32 to native Python float for JSON
    importance = {k: float(v) for k, v in importance.items()}
    importance_path = Path(output_path) / 'feature_importance.json'
    with open(importance_path, 'w') as f:
        json.dump(importance, f, indent=2)
    artifacts['feature_importance'] = str(importance_path)
    
    # Domain validation
    validation_report = DomainValidator.validate_shap_patterns(
        explainer.shap_values, X_scaled, feature_names
    )
    validation_md = DomainValidator.generate_validation_report(validation_report)
    
    validation_path = Path(REPORTS_DIR) / 'shap_domain_validation.md'
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(validation_path, 'w', encoding='utf-8') as f:
        f.write(validation_md)
    artifacts['domain_validation'] = str(validation_path)
    
    # Save validation JSON
    validation_json_path = Path(output_path) / 'domain_validation.json'
    with open(validation_json_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("SHAP EXPLANATIONS COMPLETE")
    logger.info(f"Artifacts saved to: {output_path}")
    logger.info("=" * 60)
    
    return artifacts


if __name__ == "__main__":
    artifacts = generate_all_explanations()
    
    print("\nGenerated Artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
