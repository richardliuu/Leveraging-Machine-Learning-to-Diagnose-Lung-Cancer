"""
Decision Tree Surrogate Model for Lung Cancer Random Forest Analysis

This module implements a decision tree regressor that serves as an interpretable 
surrogate model for the trained Random Forest. The surrogate is trained on the 
RF probabilities for class 1 and provides insights into feature importance 
and decision paths using SHAP analysis.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shap
import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import random

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# -----------------------------
# Load saved Random Forest
# -----------------------------
rf_model = joblib.load("models/rf_model.pkl")
print("Random Forest loaded successfully")

# -----------------------------
# Data Handling
# -----------------------------
class DataHandling:
    def __init__(self):
        self.data = None
        self.feature_cols = None
        self.groups = None
        self.X = None
        self.y = None

    def load_data(self, path="data/rf_surrogate_data.csv"):
        self.data = pd.read_csv(path)

        # Features: drop non-feature columns
        self.X = self.data.drop(columns=["chunk", "true_label", "patient_id", "predicted_label", "c1_prob"])
        self.feature_cols = self.X.columns.tolist()

        # Surrogate target: RF probabilities for class 1
        self.y = rf_model.predict_proba(self.X)[:, 1]

        # Patient grouping for GroupKFold
        self.groups = self.data['patient_id']

# -----------------------------
# Patient-safe validation split
# -----------------------------
def patient_safe_val_split(X_train, y_train, groups_train, val_size=0.15):
    """
    Create a patient-safe validation split from training data using GroupKFold.
    """
    n_splits = int(1 / val_size) if val_size < 0.5 else 5
    inner_gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in inner_gkf.split(X_train, y_train, groups_train):
        return (
            X_train.iloc[train_idx], X_train.iloc[val_idx],
            y_train[train_idx], y_train[val_idx]
        )

# -----------------------------
# SHAP Analysis
# -----------------------------
class SHAPAnalysis:
    @staticmethod
    def run_shap(model, X_val, feature_cols, fold=None):
        X_val_df = pd.DataFrame(X_val, columns=feature_cols)
        if fold is not None:
            print(f"\nRunning SHAP for fold {fold}, validation shape: {X_val_df.shape}")
        else:
            print(f"\nRunning SHAP, validation shape: {X_val_df.shape}")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_df)
        print(f"SHAP values shape: {np.array(shap_values).shape}")

        shap.summary_plot(shap_values, X_val_df)

# -----------------------------
# Surrogate Pipeline
# -----------------------------
def pipeline(handler, n_splits=4):
    outer_gkf = GroupKFold(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(outer_gkf.split(handler.X, handler.y, handler.groups), 1):
        print(f"\n--- Fold {fold} ---")

        # Outer split
        X_train, X_test = handler.X.iloc[train_idx], handler.X.iloc[test_idx]
        y_train, y_test = handler.y[train_idx], handler.y[test_idx]
        groups_train, groups_test = handler.groups.iloc[train_idx], handler.groups.iloc[test_idx]

        # Inner validation split (patient-safe)
        X_train, X_val, y_train, y_val = patient_safe_val_split(X_train, y_train, groups_train, val_size=0.15)

        # Train surrogate on train set
        surrogate_model = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=12,
            min_samples_leaf=6,
            min_samples_split=6,
            max_leaf_nodes=50,
            random_state=SEED
        )
        surrogate_model.fit(X_train, y_train)

        # Predictions
        y_val_pred = np.clip(surrogate_model.predict(X_val), 0.0, 1.0)
        y_test_pred = np.clip(surrogate_model.predict(X_test), 0.0, 1.0)

        # Fidelity: surrogate vs RF
        rf_probs_val = rf_model.predict_proba(X_val)[:, 1]
        rf_probs_test = rf_model.predict_proba(X_test)[:, 1]

        val_mse = mean_squared_error(rf_probs_val, y_val_pred)
        val_mae = mean_absolute_error(rf_probs_val, y_val_pred)
        val_r2 = r2_score(rf_probs_val, y_val_pred)

        test_mse = mean_squared_error(rf_probs_test, y_test_pred)
        test_mae = mean_absolute_error(rf_probs_test, y_test_pred)
        test_r2 = r2_score(rf_probs_test, y_test_pred)

        results.append({
            "fold": fold,
            "val_mse": val_mse, "val_mae": val_mae, "val_r2": val_r2,
            "test_mse": test_mse, "test_mae": test_mae, "test_r2": test_r2
        })

        print(f"Validation -> MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        print(f"Test       -> MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        print("Node Count:", surrogate_model.tree_.node_count, "Max Depth:", surrogate_model.tree_.max_depth)

        # SHAP on validation set of current fold
        SHAPAnalysis.run_shap(surrogate_model, X_val, handler.feature_cols, fold=fold)

    return surrogate_model, results

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    handler = DataHandling()
    handler.load_data()

    surrogate_model, results = pipeline(handler, n_splits=4)

    print("\n=== Summary Across Folds ===")
    results_df = pd.DataFrame(results)
    print(results_df.mean(numeric_only=True).round(4))
