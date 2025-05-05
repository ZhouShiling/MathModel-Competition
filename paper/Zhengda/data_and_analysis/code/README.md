# Code for AI Capability Analysis

This directory contains the Python scripts used for the AI Capability Analysis project.

## Files

1. `sem_model.py`: Implementation of Structural Equation Modeling for analyzing relationships between variables
2. `confirmatory_factor_analysis.py`: Script for confirmatory factor analysis to validate measurement models
3. `reliability_validity_presurvey.py`: Tests for reliability and validity of the pre-survey data
4. `reliability_validity_mainsurvey.py`: Tests for reliability and validity of the main survey data
5. `topic_modeling_wordcloud.py`: LDA topic modeling and word cloud generation for text analysis
6. `ensemble_algorithm_1.py`: First ensemble algorithm implementation (linear models)
7. `ensemble_algorithm_2.py`: Second ensemble algorithm implementation (tree-based models)

## Usage

Each script can be run independently, but they should generally be executed in the following order:

```bash
# Data preprocessing and validation
python reliability_validity_presurvey.py
python reliability_validity_mainsurvey.py

# Model building
python confirmatory_factor_analysis.py
python sem_model.py

# Text analysis
python topic_modeling_wordcloud.py

# Predictive modeling
python ensemble_algorithm_1.py
python ensemble_algorithm_2.py
```

## Implementation Details

### SEM Model

The Structural Equation Model analyzes the relationships between latent variables related to AI capability. The model includes:

- Measurement model: Connecting observed variables to latent constructs
- Structural model: Defining relationships between latent constructs

### Ensemble Algorithms

Two ensemble approaches were implemented:

1. **Ensemble Algorithm 1**: Combines linear models
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - ElasticNet
   - SVR

2. **Ensemble Algorithm 2**: Combines tree-based models
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM

Both ensemble methods use model stacking with cross-validation to improve prediction accuracy.