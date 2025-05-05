"""
Sample code for AI Capability Analysis

This file contains sample code for the AI Capability Analysis project.
It demonstrates the basic structure of the code used in the project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
def load_data(file_path):
    """Load data from Excel file"""
    data = pd.read_excel(file_path)
    return data

# Preprocess data
def preprocess_data(data):
    """Preprocess data for analysis"""
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Feature engineering
    data['feature_ratio'] = data['feature1'] / data['feature2']
    
    return data

# Train model
def train_model(X_train, y_train):
    """Train a random forest model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mse, r2

# Main function
def main():
    # Load data
    data = load_data("问卷数据.xlsx")
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Split data
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    mse, r2 = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
if __name__ == "__main__":
    main()