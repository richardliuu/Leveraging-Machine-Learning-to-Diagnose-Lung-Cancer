"""
XGBoost Surrogate Model for Lung Cancer MLP Analysis

This module implements an XGBoost regressor that serves as an interpretable 
surrogate model for the Multi-Layer Perceptron (MLP) implemented in 
class_based_models/lung_cancer_mlp.py. The primary purpose is to analyze and 

understand the black-box behavior of the deep learning model through gradient
boosted tree predictions as a comparison to the decision tree surrogate.

Surrogate Model Approach:
The XGBoost regressor is trained to mimic the predictions of the trained MLP model,
providing interpretable insights into how the neural network makes decisions.
This ensemble approach allows clinicians and researchers to understand the 
decision-making process that would otherwise be opaque in the MLP while 
potentially offering better fidelity than single decision trees.

Key Features:
- Patient-grouped cross-validation to prevent data leakage
- XGBoost regressor trained on MLP predictions rather than true labels
- SHAP analysis for feature importance and decision path visualization
- Model fidelity assessment to ensure surrogate accuracy
- Comprehensive tree ensemble visualization and rule extraction
- Reproducible results with fixed random seeds
- Built-in regularization and overfitting prevention

Surrogate Model Benefits:
1. Interpretability: Provides feature importance scores and interaction effects
2. Validation: Helps verify that MLP decisions are medically reasonable
3. Feature Analysis: Identifies key biomarkers driving predictions
4. Clinical Translation: Offers explainable AI for medical decision support
5. Model Debugging: Reveals potential biases or errors in MLP behavior
6. Higher Fidelity: Ensemble approach may better approximate MLP behavior

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- xgboost: Gradient boosting implementation
- scikit-learn: Evaluation metrics and preprocessing
- matplotlib: Plotting and visualization
- shap: Model interpretability and feature importance analysis

Data Requirements:
The surrogate model expects 'data/surrogate_data.csv' containing:
- Feature columns: Same features used to train the original MLP
- predicted_label: MLP predictions to be mimicked by the surrogate
- true_label: Original ground truth labels for validation
- patient_id: Patient identifiers for grouped cross-validation
- segment: Data segment identifiers

Usage:
    python models/xgboost.py

The script trains XGBoost surrogates using 4-fold cross-validation,
evaluates surrogate fidelity to the MLP, and generates interpretability
visualizations including feature importance plots and SHAP analysis.

Author: Richard Liu
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd 
import numpy as np 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap 

"""
Reproducibility Configuration

Setting deterministic seeds across all random number generators to ensure 
reproducible surrogate model training and evaluation results.
"""

import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

class DataHandling:
    def __init__(self):
        self.data = None

        # Data integrity tracking
        self.feature_cols = None
        self.duplicates = None
        self.patient_labels = None
        self.groups = None
        self.inconsistent_patients = None
        
        # Training/test split attributes
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.num_classes = None
        self.train_patients = None
        self.test_patients = None

    def load_data(self):
        self.data = pd.read_csv("data\less_dense\ld_surrogate_data.csv")
        
        # Extract features (same as original MLP input)
        self.X = self.data.drop(columns=["segment", "true_label", "patient_id", "predicted_label", "mlp_c1_prob"])
        
        # Target is MLP predictions (not ground truth)
        self.y = self.data['mlp_c1_prob']
        
        # Preserve feature names for interpretability
        self.feature_cols = self.X.columns.tolist()
        
        # Data Leakage Prevention Setup
        self.groups = self.data['patient_id']
        self.duplicates = self.data.duplicated(subset=self.feature_cols)
        self.patient_labels = self.data.groupby('patient_id')['true_label'].nunique()
        self.inconsistent_patients = self.patient_labels[self.patient_labels > 1]

    def split(self, X, y, data, train_idx, test_idx):
        self.X_train = X.iloc[train_idx]
        self.y_train = y.iloc[train_idx]
        self.X_test = X.iloc[test_idx]
        self.y_test = y.iloc[test_idx]

        # Track patient groups for validation
        self.train_patients = set(data.iloc[train_idx]['patient_id'])
        self.test_patients = set(data.iloc[test_idx]['patient_id'])

        self.num_classes = 2
    
    def validation_split(self):
        # Create stratified validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=0.15, 
            random_state=SEED
        )


def pipeline(handler):
    gkf = GroupKFold(n_splits=4)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        handler.split(handler.X, handler.y, handler.data, train_idx, test_idx)
        handler.validation_split()

        model = XGBRegressor(
            n_estimators=5000,
            learning_rate=0.05,
            max_depth=30,
            subsample=0.5,
            colsample_bytree=0.3,
            random_state=42,
            gamma=0,
            reg_alpha=1,
            reg_lambda=1,
            eval_metric="rmse",
            early_stopping_rounds=10
        )

        model.fit(
            handler.X_train,
            handler.y_train,
            eval_set=[(handler.X_val, handler.y_val)],
            verbose=False
        )

        accuracy = model.score(handler.X_test, handler.y_test)
        y_pred_raw = model.predict(handler.X_test)
        # Apply probability bounding to ensure predictions are in [0, 1] range
        y_pred = np.clip(y_pred_raw, 0.0, 1.0)
        
        # Validation: Check if any predictions were out of bounds
        out_of_bounds = np.sum((y_pred_raw < 0) | (y_pred_raw > 1))
        if out_of_bounds > 0:
            print(f"Warning: {out_of_bounds} predictions were clipped to [0,1] range")
            print(f"Raw prediction range: [{y_pred_raw.min():.4f}, {y_pred_raw.max():.4f}]")
        
        print(np.unique(y_pred, return_counts=True))

        print(f"Fold {fold + 1} R² Score:", accuracy)
        print("Number of Trees:", model.n_estimators)
        print("Best Iteration:", model.best_iteration if hasattr(model, 'best_iteration') else 'N/A')
        
        # Feature importance analysis
        feature_importance = model.feature_importances_
        feature_names = handler.feature_cols
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))

    return model

class SHAPAnalysis:
    def __init__(self, model):
        self.model = model

    def run_shap(model, handler):
        # Convert validation data to DataFrame with feature names
        X_val_df = pd.DataFrame(handler.X_val, columns=handler.feature_cols)
        print(f"Validation data shape: {X_val_df.shape}")

        # Initialize SHAP explainer for XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_df)
        print(f"SHAP values shape: {np.array(shap_values).shape}")

        # Generate SHAP summary plot for regression output
        # Shows features that increase/decrease predicted probability
        shap.summary_plot(shap_values, X_val_df)

class FidelityCheck:

    def __init__(self, surrogate_model, data_handler):
        self.fidelity = None
        self.surrogate = surrogate_model
        self.handler = data_handler

    def print_predictions_comparison(self, mlp_preds, surrogate_preds, num_samples=100):
        print(f"\nComparing first {num_samples} predictions (MLP vs Surrogate):\n")
        print(f"{'Index':>5} | {'MLP Prediction':>15} | {'Surrogate Prediction':>20} | {'Difference':>10}")
        print("-" * 60)
        for i in range(min(num_samples, len(mlp_preds))):
            mlp_val = mlp_preds[i]
            sur_val = surrogate_preds[i]
            diff = abs(mlp_val - sur_val)
            print(f"{i:5d} | {mlp_val:15.4f} | {sur_val:20.4f} | {diff:10.4f}")

        mse = mean_squared_error(mlp_preds, surrogate_preds)
        mae = mean_absolute_error(mlp_preds, surrogate_preds)
        r2 = r2_score(mlp_preds, surrogate_preds)

        print("\nFidelity Metrics:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²:  {r2:.4f}\n")

    def comparison(self):
        try:
            # Import MLP module only when needed to avoid import conflicts
            import class_based_models.lung_cancer_mlp
            
            # Load and initialize MLP model
            mlp_model = class_based_models.lung_cancer_mlp.LungCancerMLP(
                num_classes=self.handler.num_classes,
                input_dim=self.handler.X_val.shape[1]
            )
            
            # Generate MLP predictions on validation data
            mlp_preds = mlp_model.predict(self.handler.X_val)
            print("MLP predictions shape:", mlp_preds.shape)
            
            # Generate surrogate predictions on validation data with probability bounding
            surrogate_preds_raw = self.surrogate.predict(self.handler.X_val)
            surrogate_preds = np.clip(surrogate_preds_raw, 0.0, 1.0)
            
            # Validation: Check if any predictions were out of bounds
            out_of_bounds = np.sum((surrogate_preds_raw < 0) | (surrogate_preds_raw > 1))
            if out_of_bounds > 0:
                print(f"Validation Warning: {out_of_bounds} predictions were clipped to [0,1] range")
                print(f"Raw validation prediction range: [{surrogate_preds_raw.min():.4f}, {surrogate_preds_raw.max():.4f}]")
            
            print("Surrogate predictions shape:", surrogate_preds.shape)
            
            # Extract class 1 probabilities from MLP predictions if multi-class output
            if len(mlp_preds.shape) > 1 and mlp_preds.shape[1] > 1:
                mlp_class1_probs = mlp_preds[:, 1]  # Extract class 1 probabilities
            else:
                mlp_class1_probs = mlp_preds
                
            print("MLP class 1 probabilities shape:", mlp_class1_probs.shape)
            
            # Print detailed prediction comparison and fidelity metrics
            self.print_predictions_comparison(mlp_class1_probs, surrogate_preds, num_samples=100)
            
            # Set fidelity as R² score (ranges from -∞ to 1, where 1 is perfect)
            self.fidelity = r2_score(mlp_class1_probs, surrogate_preds)
                
        except Exception as e:
            print(f"Fidelity check failed: {e}")
            print("Unable to compare with MLP model - using surrogate validation accuracy instead")
            self.fidelity = self.surrogate.score(self.handler.X_val, self.handler.y_val)
            print(f"Surrogate validation accuracy: {self.fidelity:.2%}")


if __name__ == "__main__":
    handler = DataHandling()
    handler.load_data()
    surrogate_model = pipeline(handler)

    check = FidelityCheck(surrogate_model, handler)
    check.comparison()

    analyze = SHAPAnalysis(surrogate_model, handler)
    analyze.run_shap(surrogate_model, handler)