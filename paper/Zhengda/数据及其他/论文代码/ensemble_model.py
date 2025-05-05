"""
Ensemble Model for AI Capability Analysis

This file contains the implementation of the ensemble model used in the AI Capability Analysis project.
It combines multiple machine learning models to improve prediction accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

class EnsembleModel:
    """Ensemble model that combines multiple base models"""
    
    def __init__(self):
        """Initialize the ensemble model with base models"""
        self.base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge(alpha=1.0)),
            ('lasso', Lasso(alpha=0.1)),
            ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ]
        self.meta_model = LinearRegression()
        self.base_model_predictions = None
        
    def fit(self, X, y, n_folds=5):
        """Fit the ensemble model using k-fold cross-validation"""
        # Initialize array to store meta features
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        # K-fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # For each base model
        for i, (name, model) in enumerate(self.base_models):
            print(f"Training {name} model...")
            
            # For each fold
            for train_idx, val_idx in kf.split(X):
                # Split data
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict on validation set
                meta_features[val_idx, i] = model.predict(X_val)
            
            # Retrain on full dataset
            model.fit(X, y)
        
        # Train meta model
        print("Training meta model...")
        self.meta_model.fit(meta_features, y)
        
        # Save base model predictions for later use
        self.base_model_predictions = meta_features
        
        return self
    
    def predict(self, X):
        """Make predictions with the ensemble model"""
        # Initialize array to store predictions from base models
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        # For each base model
        for i, (name, model) in enumerate(self.base_models):
            # Make predictions
            base_predictions[:, i] = model.predict(X)
        
        # Make final prediction using meta model
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions
    
    def evaluate(self, X, y):
        """Evaluate the ensemble model"""
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Print results
        print(f"Ensemble Model Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Evaluate individual base models
        print("\nBase Model Performance:")
        for i, (name, _) in enumerate(self.base_models):
            base_mse = mean_squared_error(y, self.base_model_predictions[:, i])
            base_r2 = r2_score(y, self.base_model_predictions[:, i])
            print(f"{name}: MSE = {base_mse:.4f}, R² = {base_r2:.4f}")
        
        return mse, r2