# AI Capability Analysis Results

This directory contains the results and outputs from the AI Capability Analysis project.

## Contents

### Model Performance Results

- `model_performance.xlsx`: Comprehensive performance metrics for all models
  - MSE, RMSE, MAE values
  - R² and adjusted R² scores
  - Cross-validation results
  - Hyperparameter settings

- `model_comparison.png`: Visual comparison of model performance
  - Bar charts comparing error metrics
  - Performance across different evaluation criteria
  - Confidence intervals for performance metrics

### Feature Importance Analysis

- `feature_importance_summary.xlsx`: Aggregated feature importance across models
  - Ranking of features by importance
  - Consistency of importance across models
  - Statistical significance of feature contributions

- Feature importance visualizations for each model:
  - Coefficient plots for linear models
  - Gini importance for tree-based models
  - Permutation importance for all models

### SHAP Value Analysis

- SHAP summary plots for each model
  - Feature impact direction and magnitude
  - Feature interaction effects
  - Individual prediction explanations

- `shap_values_summary.xlsx`: Tabulated SHAP values
  - Mean absolute SHAP values by feature
  - SHAP value distributions
  - Feature clustering by SHAP patterns

### Final Analysis Report

- `ai_capability_analysis.xlsx`: Final analysis results
  - Key findings summary
  - Statistical test results
  - Hypothesis validation outcomes
  - Recommendations based on analysis

## Interpretation Guide

The results should be interpreted in the context of the research questions outlined in the paper:

1. What are the key factors influencing AI capability in graduate students?
2. How do these factors interact and relate to each other?
3. Which models best predict AI capability based on the identified factors?
4. What are the most important predictors of AI capability?

The SEM model results address questions 1 and 2, while the ensemble model results address questions 3 and 4.