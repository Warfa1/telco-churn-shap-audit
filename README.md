# Telco Churn Prediction: A Leakage-Free SHAP Consistency Audit

## Overview
This project predicts customer churn on the IBM Telco Customer Churn dataset using Logistic Regression and Random Forest. It addresses two research questions:

1. **RQ1:** Do Logistic Regression and Random Forest agree on SHAP feature importance rankings, and is that agreement stable across random seeds?
2. **RQ2:** Under a leakage-free pipeline, how do SMOTE, class_weight, and no imbalance handling differ in Recall, F1, and ROC-AUC — and how much inflation does pre-split SMOTE produce?

## Dataset
- **Source:** IBM Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)
- **Size:** 7,043 rows, 21 columns (7,032 after cleaning)
- **Target:** Churn (0/1)
- **Class imbalance:** 73.4% no churn / 26.6% churn

## Key Results

### RQ2: Leakage Audit
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (Majority) | 0.734 | 0.000 | 0.000 | 0.000 | 0.500 |
| LogReg (No Imbalance) | 0.804 | 0.648 | 0.572 | 0.608 | 0.836 |
| RandomForest (No Imbalance) | 0.783 | 0.616 | 0.484 | 0.542 | 0.812 |
| LogReg (Class Weight) | 0.726 | 0.490 | 0.797 | 0.607 | 0.835 |
| RandomForest (Class Weight) | 0.782 | 0.616 | 0.476 | 0.537 | 0.813 |
| LogReg (SMOTE Correct) | 0.732 | 0.497 | 0.778 | 0.607 | 0.833 |
| RandomForest (SMOTE Correct) | 0.771 | 0.570 | 0.570 | 0.570 | 0.810 |
| **LogReg (SMOTE Leaky)** | 0.788 | 0.768 | 0.825 | 0.796 | 0.864 |
| **RandomForest (SMOTE Leaky)** | 0.861 | 0.868 | 0.852 | 0.860 | 0.931 |

**Conclusion:** Pre-split SMOTE inflates Random Forest F1 from 0.570 to 0.860 (+0.29) — a clearly fake result. The baseline shows why accuracy is misleading: 73.4% accuracy but zero Recall.

### RQ1: SHAP Consistency
- **Spearman ρ (LR vs RF):** 0.585 (moderate agreement)
- **Top-10 overlap:** 7/10 features
- **Within-model stability (3 seeds):** LR = 1.000, RF = 1.000

**Conclusion:** SHAP explanations are perfectly stable within a model but only moderately consistent across different algorithm families. Relying on one model's explanation alone is insufficient.

## Files
- `Telco_Churn_SHAP_Audit.ipynb` — full Colab notebook
- `metrics_table.csv` — complete model comparison
- `shap_importance_lr.csv`, `shap_importance_rf.csv` — feature rankings
- `stability_results.txt` — stability scores
- `figures/` — all plots (ROC curves, confusion matrices, SHAP comparison)

## How to Run
1. Open `Telco_Churn_SHAP_Audit.ipynb` in Google Colab
2. Mount your Google Drive
3. Run all cells in order (~5 minutes on free CPU)

## Author
Yusuf abdinsir zarfa
