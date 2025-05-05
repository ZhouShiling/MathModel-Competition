# AI Capability Analysis - Data and Analysis

This directory contains the data, code, models, and results for the AI Capability Analysis project (TJJM20250309033491).

## Directory Structure

- **code/**: Python scripts for data analysis, model training, and evaluation
  - SEM model implementation
  - Reliability and validity tests
  - Topic modeling and word cloud generation
  - Ensemble algorithm implementations

- **data/**: Raw and processed data
  - Survey data (questionnaire responses)
  - Preprocessed datasets

- **models/**: Model files and related artifacts
  - SEM model results
  - Ensemble model 1 (multiple regression models)
  - Ensemble model 2 (tree-based models)

- **visualizations/**: Visualizations and figures
  - Word clouds
  - Data distribution plots
  - Correlation analyses
  - Workflow diagrams

- **results/**: Analysis results and outputs
  - Model performance metrics
  - Feature importance analyses
  - SHAP value analyses
  - Comparative model evaluations

## Key Files

- `code/sem_model.py`: Implementation of Structural Equation Modeling
- `code/factor_analysis.py`: Confirmatory factor analysis
- `code/reliability_validity.py`: Tests for reliability and validity
- `code/ensemble_algorithm_1.py`: First ensemble algorithm implementation
- `code/ensemble_algorithm_2.py`: Second ensemble algorithm implementation
- `data/survey_data.xlsx`: Raw survey data from questionnaires
- `visualizations/research_workflow.png`: Overall research workflow diagram
- `visualizations/survey_workflow.png`: Survey methodology workflow
- `results/model_comparison.png`: Comparison of different model performances
- `results/ai_capability_analysis.xlsx`: Final analysis results

## Usage

The code in this directory follows the research workflow described in the paper. To reproduce the results:

1. Start with the data preprocessing scripts
2. Run the reliability and validity tests
3. Execute the SEM model analysis
4. Run the ensemble algorithms
5. Generate visualizations and result summaries

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - xgboost
  - lightgbm
  - shap
  - semopy (for SEM modeling)
  - wordcloud