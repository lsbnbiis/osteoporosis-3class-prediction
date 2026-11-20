# Osteoporosis Three-Class Prediction

## Overview

This study provides a machine learning framework for three-class osteoporosis prediction, including both models with SNP-based genetic information and models without SNP features.

The models classify individuals into three categories:

- **Normal:** T-score > -1.0
- **Osteopenia:** -2.5 ≤ T-score ≤ -1.0
- **Osteoporosis:** T-score < -2.5

## Machine Learning Models

The following machine learning algorithms were implemented:

- Support Vector Machine (SVM)
- Random Forest (RF)
- Extreme Gradient Boosting (XGBoost)
- Histogram-based Gradient Boosting Classifier (HGBC)
- Ensemble Learning

Two modeling approaches were evaluated:

1. **Without SNP:** Prediction using clinical and demographic features.
2. **With SNP:** Prediction incorporating clinical, demographic, and SNP-based genetic information, including polygenic risk scores (PRS).

## Repository Structure

```text
osteoporosis-3class-prediction/
├── development/
│   ├── data/
│   ├── prs/
│   ├── snp2vec/
│   ├── osteoporosis_all_feature_model(3class).ipynb
│   ├── osteoporosis_select_feature_model(3class).ipynb
│   ├── osteoporosis_snp_all_feature_model(3class).ipynb
│   └── osteoporosis_snp_select_feature_model(3class).ipynb
├── model/
│   ├── sub_feature_oste_scaler.pkl
│   ├── sub_feature_nosnp_ensemble(3class).pkl
│   ├── sub_snp_feature_oste_scaler.pkl
│   └── sub_snp_feature_svm(3class).pkl
├── predict_nonsnp.py
└── predict_snp.py
```

## Quick Start Prediction

>python predict_nonsnp -i nonsnp_example -o nonsnp_prediction.csv

>python predict_snp -i snp_example -o snp_prediction.csv
