"""
Decision Tree Surrogate Model with Fold Diagnostics for Lung Cancer Random Forest
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shap
import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
    def patient_safe_val_split(self, X_train, y_train, groups_train, val_size=0.15):
        n_splits = int(1 / val_size)
        gkf_inner = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf_inner.split(X_train, y_train, groups_train):
            return X_train.iloc[train_idx], X_train.iloc[val_idx], y_train[train_idx], y_train[val_idx]

# -----------------------------
# Surrogate Pipeline with Diagnostics
# -----------------------------
def surrogate_pipeline(handler, max_depth=10, problematic_r2_threshold=0.6):
    outer_gkf = GroupKFold(n_splits=4)
    dp = DataPreprocess()

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(outer_gkf.split(handler.X, handler.y, handler.groups), 1):
        print(f"\n--- Fold {fold} ---")

        # Split train/test
        X_train, X_test = handler.X.iloc[train_idx], handler.X.iloc[test_idx]
        y_train, y_test = handler.y[train_idx], handler.y[test_idx]
        groups_train = handler.groups.iloc[train_idx]

        # Patient-safe validation split
        X_train_inner, X_val, y_train_inner, y_val = dp.patient_safe_val_split(X_train, y_train, groups_train)

        # Train surrogate
        surrogate = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=max_depth,
            min_samples_leaf=6,
            min_samples_split=20,
            max_leaf_nodes=50,
            random_state=SEED
        )
        surrogate.fit(X_train_inner, y_train_inner)

        # Predictions
        y_val_pred = np.clip(surrogate.predict(X_val), 0, 1)
        y_test_pred = np.clip(surrogate.predict(X_test), 0, 1)
        rf_probs_val = rf_model.predict_proba(X_val)[:, 1]
        rf_probs_test = rf_model.predict_proba(X_test)[:, 1]

        # Compute metrics
        val_r2 = r2_score(rf_probs_val, y_val_pred)
        test_r2 = r2_score(rf_probs_test, y_test_pred)
        val_mse = mean_squared_error(rf_probs_val, y_val_pred)
        test_mse = mean_squared_error(rf_probs_test, y_test_pred)

        print(f"Validation -> MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
        print(f"Test       -> MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        print("Node Count:", surrogate.tree_.node_count, "Max Depth:", surrogate.tree_.max_depth)

        fold_results.append({
            "fold": fold, "surrogate": surrogate,
            "val_r2": val_r2, "test_r2": test_r2,
            "val_mse": val_mse, "test_mse": test_mse
        })

        # Diagnose problematic folds
        if test_r2 < problematic_r2_threshold:
            print(f"Fold {fold} flagged as problematic (R² < {problematic_r2_threshold})")

            # 1️⃣ Feature distributions
            for col in X_train.columns[:4]:  # show first 5 features as example
                plt.figure(figsize=(6,3))
                sns.kdeplot(X_train[col], label="Train", fill=True)
                sns.kdeplot(X_test[col], label="Test", fill=True)
                plt.title(f"Fold {fold} - Feature: {col}")
                plt.xlabel(col)
                plt.ylabel("Density")
                plt.legend()
                plt.show()

            # 2️⃣ RF probabilities
            plt.figure(figsize=(6,3))
            sns.kdeplot(rf_probs_val, label="Validation RF probs", fill=True)
            sns.kdeplot(rf_probs_test, label="Test RF probs", fill=True)
            plt.title(f"Fold {fold} - RF Probability Distribution")
            plt.xlabel("Probability for class 1")
            plt.ylabel("Density")
            plt.legend()
            plt.show()

            # 3️⃣ Extreme predictions
            extreme_mask = (rf_probs_test < 0.05) | (rf_probs_test > 0.95)
            print(f"Number of extreme RF probabilities in test set: {extreme_mask.sum()} / {len(rf_probs_test)}")

    return fold_results

# -----------------------------
# SHAP Analysis
# -----------------------------
class SHAPAnalysis:
    @staticmethod
    def run_shap(model, X_val, feature_cols):
        X_val_df = pd.DataFrame(X_val, columns=feature_cols)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_df)
        shap.summary_plot(shap_values, X_val_df)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    handler = DataHandling()
    handler.load_data()

    # Run surrogate pipeline with diagnostics
    fold_results = surrogate_pipeline(handler, max_depth=10, problematic_r2_threshold=0.6)

    # Optional: Run SHAP for the last fold surrogate
    last_fold_surrogate = fold_results[-1]["surrogate"]
    SHAPAnalysis.run_shap(last_fold_surrogate, handler.X.iloc[:100], handler.feature_cols)  # example subset
