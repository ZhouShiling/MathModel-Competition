# AI Capability Analysis Data

This directory contains the data files used in the AI Capability Analysis project.

## Data Files

### Survey Data

- `survey_data.xlsx`: Raw data collected from the questionnaire survey
  - Contains responses from participants
  - Includes demographic information, AI knowledge assessment, and capability self-evaluation
  - Structured according to the survey design described in the paper

### Processed Data

- `preprocessed_data.csv`: Cleaned and preprocessed survey data
  - Missing values handled
  - Outliers addressed
  - Feature engineering applied
  - Ready for model input

- `factor_analysis_data.csv`: Data prepared for factor analysis
  - Standardized variables
  - Correlation matrix
  - KMO and Bartlett's test results

- `sem_model_data.csv`: Data prepared for Structural Equation Modeling
  - Latent variables defined
  - Measurement model variables organized
  - Structural model variables prepared

- `ensemble_model_data.csv`: Data prepared for ensemble modeling
  - Features and target variables separated
  - Train-test split information
  - Cross-validation folds defined

## Data Dictionary

The `data_dictionary.xlsx` file provides detailed information about each variable:

- Variable name
- Description
- Data type
- Measurement scale
- Possible values
- Source
- Transformation applied (if any)

## Data Collection Methodology

The data was collected through a structured questionnaire administered to graduate students. The survey design followed these steps:

1. Literature review to identify key dimensions of AI capability
2. Expert consultation to validate the survey items
3. Pre-survey testing for reliability and validity
4. Main survey implementation
5. Data cleaning and preprocessing

Details of the survey methodology are provided in the paper and visualized in `survey_workflow.png`.