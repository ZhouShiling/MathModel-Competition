# AI Capability Analysis Models

This directory contains the model files and related artifacts for the AI Capability Analysis project.

## Directory Structure

### SEM Model

The Structural Equation Model (SEM) directory contains:

- Model specification files
- Model fit statistics
- Path coefficients
- Measurement model results
- Structural model results

### Ensemble Model 1

The first ensemble model combines linear models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- SVR

For each model, the following files are available:
- Feature importance analysis
- Prediction results
- Residual analysis
- SHAP value analysis

### Ensemble Model 2

The second ensemble model combines tree-based models:

- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

For each model, the following files are available:
- Feature importance analysis
- Prediction results
- Residual analysis
- SHAP value analysis

## Model Performance

The model performance is evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score
- Adjusted R² score

The comparative performance of all models is summarized in `model_comparison.png` and `model_performance.xlsx`.