# Credit Default Prediction: Gradient Boosting Ensemble & Anomaly Detection

### DSC LOGIKA UI 2026 | Team GASSS JUARA

## Project Overview

This project focuses on predicting credit default risk using machine learning techniques. It was developed as part of the DSC LOGIKA UI 2026 Data Science Competition.

We explored feature engineering, cross-validation, gradient boosting models, weighted ensemble learning, and anomaly-aware training.

## Machine Learning Models

- LightGBM V12 — 40%
- LightGBM V11 — 20%
- CatBoost V3-C — 20%
- XGBoost V1 — 20%

## Results

| Evaluation Metric | Score |
|---|---|
| Public KS | **0.55163** |
| Private KS | **0.53271** |
| Out-of-Fold KS | **0.512201** |
| Out-of-Fold AUC | **0.837744** |

## Research Experiments

We investigated several anomaly detection techniques, including Isolation Forest, FastABOD, CBLOF, COF, MCD, and One-Class SVM.

These experiments demonstrated that improvements in individual models or cross-validation scores do not always translate to improved performance on unseen data.

## Technologies

Python, Pandas, NumPy, Scikit-learn, LightGBM, CatBoost, XGBoost, and Google Colab.

## Kaggle Notebook

[View the Technical Portfolio on Kaggle](https://www.kaggle.com/code/sukmaayuhartani/credit-default-prediction-gradient-boosting-ensem)

## Reproducibility Note

The notebook documents the original model development process and includes archived training code excerpts. It does not independently reproduce the full competition training pipeline without the original dataset, engineered features, training splits, and saved model checkpoints.

## Key Takeaways

- Weighted ensembles can improve predictive performance by combining complementary models.
- Out-of-fold validation is important for reliable model evaluation.
- Anomaly-aware learning does not always improve ensemble generalization.
- Public and private leaderboard results can differ significantly.
