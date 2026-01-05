# Baseline Model Report

## Model: Logistic Regression (Baseline)

**Generated:** 2025-12-24T12:01:30.322713

---

## Dataset Information

| Metric | Value |
|--------|-------|
| Total Samples | 9,833 |
| Features | 106 |
| Positive Rate | 0.0342 (3.42%) |

---

## Cross-Validation Results

| Metric | Mean | Std |
|--------|------|-----|
| F1-Score | 0.3004 | ±0.0621 |
| Recall | 0.6529 | - |
| Precision | 0.2225 | - |
| ROC-AUC | 0.8806 | - |

---

## Confusion Matrix

```
              Predicted
              Neg    Pos
Actual Neg     8349   1148
       Pos       30    306
```

---

## Analysis

The baseline Logistic Regression model achieves a mean F1-score of **0.3004** 
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
