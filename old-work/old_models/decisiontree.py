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
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from imblearn.combine import SMOTEENN
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
# Data Preprocessing
# -----------------------------
class DataPreprocess:
    def split(self, X, y, groups, train_idx, test_idx):
        self.X_train, self.X_test = X.iloc[train_idx], X.iloc[test_idx]
        self.y_train, self.y_test = y[train_idx], y[test_idx]

        self.train_patients = set(groups.iloc[train_idx])
        self.test_patients = set(groups.iloc[test_idx])

    def validation_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.15, random_state=SEED
        )

# -----------------------------
# Surrogate Pipeline
# -----------------------------
def pipeline(handler):
    gkf = GroupKFold(n_splits=4)
    surrogate_model = None

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        dp = DataPreprocess()
        dp.split(handler.X, handler.y, handler.groups, train_idx, test_idx)
        dp.validation_split()

        surrogate_model = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=10,
            min_samples_leaf=3,
            min_samples_split=20,
            max_leaf_nodes=15,
            random_state=SEED
        )
        surrogate_model.fit(dp.X_train, dp.y_train)

        # Predictions on test set
        y_pred_raw = surrogate_model.predict(dp.X_test)
        y_pred = np.clip(y_pred_raw, 0.0, 1.0)

        # Surrogate fidelity vs RF
        rf_probs_test = rf_model.predict_proba(dp.X_test)[:, 1]
        mse = mean_squared_error(rf_probs_test, y_pred)
        mae = mean_absolute_error(rf_probs_test, y_pred)
        r2 = r2_score(rf_probs_test, y_pred)

        print(f"\nFold {fold+1} -> Surrogate Fidelity:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        print("Node Count:", surrogate_model.tree_.node_count, "Max Depth:", surrogate_model.tree_.max_depth)

        # Optional: Export tree visualization
        # print(export_graphviz(surrogate_model, feature_names=handler.feature_cols))

    return surrogate_model, dp.X_val

# -----------------------------
# SHAP Analysis
# -----------------------------
class SHAPAnalysis:
    @staticmethod
    def run_shap(model, X_val, feature_cols):
        X_val_df = pd.DataFrame(X_val, columns=feature_cols)
        print(f"Validation data shape: {X_val_df.shape}")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_df)
        print(f"SHAP values shape: {np.array(shap_values).shape}")

        shap.summary_plot(shap_values, X_val_df)

# -----------------------------
# Fidelity Check
# -----------------------------
class FidelityCheck:
    def __init__(self, surrogate_model, rf_model, X_val):
        self.surrogate = surrogate_model
        self.rf_model = rf_model
        self.X_val = X_val

    def run(self):
        rf_probs = self.rf_model.predict_proba(self.X_val)[:, 1]
        surrogate_preds = np.clip(self.surrogate.predict(self.X_val), 0, 1)

        mse = mean_squared_error(rf_probs, surrogate_preds)
        mae = mean_absolute_error(rf_probs, surrogate_preds)
        r2 = r2_score(rf_probs, surrogate_preds)

        print("\nFidelity on validation set:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    handler = DataHandling()
    handler.load_data()

    surrogate_model, X_val = pipeline(handler)

    # Run fidelity check
    fidelity = FidelityCheck(surrogate_model, rf_model, X_val)
    fidelity.run()

    # Run SHAP analysis
    SHAPAnalysis.run_shap(surrogate_model, X_val, handler.feature_cols)
