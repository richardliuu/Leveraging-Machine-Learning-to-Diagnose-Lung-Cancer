"""
Decision Tree Surrogate Model WITHOUT Calibration
Direct surrogate approximation of Random Forest probabilities
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shap
import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import random
import matplotlib.pyplot as plt

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
        self.true_labels = None

    def load_data(self, path="data/rf2_surrogate_data.csv"):
        """Load original data"""
        self.data = pd.read_csv(path)
        
        # Features: drop non-feature columns (labels and metadata)
        self.X = self.data.drop(columns=['chunk', 'patient_id', 'c1_prob', 'predicted_label', 'true_label'])
        self.feature_cols = self.X.columns.tolist()
        
        # True labels 
        self.true_labels = self.data['c1_prob']
        
        # Patient grouping for GroupKFold
        self.groups = self.data['patient_id']

# -----------------------------
# Data Preprocessing
# -----------------------------
class DataPreprocess:
    def patient_safe_val_split(self, X_train, groups_train, val_size=0.15):
        """
        Split training data into train/validation preserving patient groups.
        """
        n_splits = int(1 / val_size)
        gkf_inner = GroupKFold(n_splits=n_splits)
        
        dummy_y = np.zeros(len(X_train))
        
        for train_idx, val_idx in gkf_inner.split(X_train, dummy_y, groups_train):
            return train_idx, val_idx

# -----------------------------
# Surrogate Pipeline WITHOUT Calibration
# -----------------------------
def surrogate_pipeline_uncalibrated(handler, max_depth=5):
    outer_gkf = GroupKFold(n_splits=4)
    dp = DataPreprocess()

    fold_results = []
    extreme_prob_stats = []

    print("\n=== Training Surrogate Model WITHOUT Calibration ===\n")

    for fold, (train_idx, test_idx) in enumerate(outer_gkf.split(handler.X, handler.true_labels, handler.groups), 1):
        print(f"--- Fold {fold} ---")

        # Split train/test features and groups
        X_train_fold, X_test_fold = handler.X.iloc[train_idx], handler.X.iloc[test_idx]
        groups_train = handler.groups.iloc[train_idx]
        
        # Get UNCALIBRATED RF predictions (raw probabilities)
        y_train_fold = rf_model.predict_proba(X_train_fold)[:, 1]
        y_test_fold = rf_model.predict_proba(X_test_fold)[:, 1]
        
        # Count extreme probabilities
        train_extreme = np.sum((y_train_fold < 0.05) | (y_train_fold > 0.95))
        test_extreme = np.sum((y_test_fold < 0.05) | (y_test_fold > 0.95))
        
        print(f"Extreme probs - Train: {train_extreme}/{len(y_train_fold)} ({100*train_extreme/len(y_train_fold):.1f}%)")
        print(f"Extreme probs - Test: {test_extreme}/{len(y_test_fold)} ({100*test_extreme/len(y_test_fold):.1f}%)")

        # Patient-safe validation split
        train_inner_idx, val_idx = dp.patient_safe_val_split(X_train_fold, groups_train)
        
        X_train_inner = X_train_fold.iloc[train_inner_idx]
        X_val = X_train_fold.iloc[val_idx]
        
        # Generate UNCALIBRATED RF predictions for train/val
        y_train_inner = rf_model.predict_proba(X_train_inner)[:, 1]
        y_val = rf_model.predict_proba(X_val)[:, 1]

        # Train surrogate on UNCALIBRATED probabilities
        surrogate = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=6,
            max_features=0.6,
            min_samples_leaf=10,
            min_samples_split=25,
            max_leaf_nodes=None,
            random_state=SEED
        )
        surrogate.fit(X_train_inner, y_train_inner)

        # Surrogate predictions
        y_val_pred = np.clip(surrogate.predict(X_val), 0, 1)
        y_test_pred = np.clip(surrogate.predict(X_test_fold), 0, 1)

        # Compute metrics
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test_fold, y_test_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        test_mse = mean_squared_error(y_test_fold, y_test_pred)

        print(f"Validation -> MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
        print(f"Test       -> MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        print(f"Node Count: {surrogate.tree_.node_count}, Max Depth: {surrogate.tree_.max_depth}")
        #print(export_graphviz(surrogate, feature_names=handler.feature_cols))

        fold_results.append({
            "fold": fold, 
            "surrogate": surrogate,
            "val_r2": val_r2, 
            "test_r2": test_r2,
            "val_mse": val_mse, 
            "test_mse": test_mse,
            "node_count": surrogate.tree_.node_count,
            "max_depth": surrogate.tree_.max_depth
        })
        
        extreme_prob_stats.append({
            "fold": fold,
            "train_extreme": train_extreme,
            "test_extreme": test_extreme,
            "train_size": len(y_train_fold),
            "test_size": len(y_test_fold)
        })

    return fold_results, extreme_prob_stats

# -----------------------------
# Analyze Results
# -----------------------------
def analyze_results(fold_results, extreme_stats):
    print("\n=== SUMMARY STATISTICS ===\n")
    
    # R² Statistics
    val_r2s = [f["val_r2"] for f in fold_results]
    test_r2s = [f["test_r2"] for f in fold_results]
    
    print("Validation R²:")
    print(f"  Mean: {np.mean(val_r2s):.4f} ± {np.std(val_r2s):.4f}")
    print(f"  Range: [{min(val_r2s):.4f}, {max(val_r2s):.4f}]")
    
    print("\nTest R²:")
    print(f"  Mean: {np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}")
    print(f"  Range: [{min(test_r2s):.4f}, {max(test_r2s):.4f}]")
    
    # Identify problematic folds
    problematic_threshold = 0.7
    problematic_folds = [f["fold"] for f in fold_results if f["test_r2"] < problematic_threshold]
    
    if problematic_folds:
        print(f"\nProblematic folds (R² < {problematic_threshold}): {problematic_folds}")
        for fold_num in problematic_folds:
            fold_data = fold_results[fold_num - 1]
            extreme_data = extreme_stats[fold_num - 1]
            print(f"  Fold {fold_num}:")
            print(f"    Test R²: {fold_data['test_r2']:.4f}")
            print(f"    Extreme probs: {extreme_data['test_extreme']}/{extreme_data['test_size']} "
                  f"({100*extreme_data['test_extreme']/extreme_data['test_size']:.1f}%)")
    else:
        print(f"\nNo problematic folds (all R² >= {problematic_threshold})")
    
    # Tree complexity
    print("\nTree Complexity:")
    print(f"  Average nodes: {np.mean([f['node_count'] for f in fold_results]):.1f}")
    print(f"  Average depth: {np.mean([f['max_depth'] for f in fold_results]):.1f}")
    
    # Compare with calibrated results
    print("\n=== COMPARISON WITH CALIBRATED RESULTS ===")
    print("Uncalibrated (this run):")
    print(f"  Average Test R²: {np.mean(test_r2s):.4f}")
    print("\nCalibrated (from previous run):")
    print("  Beta calibration Test R²: 0.9402")
    print("  Isotonic calibration Test R²: 0.9235")
    print("  Platt calibration Test R²: 0.9241")
    print(f"\nDifference: {0.9402 - np.mean(test_r2s):.4f} improvement with calibration")

# -----------------------------
# Visualize Probability Distributions
# -----------------------------
def visualize_distributions(handler, fold_results):
    """Visualize probability distributions for each fold"""
    outer_gkf = GroupKFold(n_splits=4)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for fold, (train_idx, test_idx) in enumerate(outer_gkf.split(handler.X, handler.true_labels, handler.groups), 1):
        X_test_fold = handler.X.iloc[test_idx]
        
        # RF probabilities
        rf_probs = rf_model.predict_proba(X_test_fold)[:, 1]
        
        # Surrogate probabilities
        surrogate = fold_results[fold-1]['surrogate']
        surrogate_probs = np.clip(surrogate.predict(X_test_fold), 0, 1)
        
        # Plot RF distribution
        ax1 = axes[0, fold-1]
        ax1.hist(rf_probs, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title(f'Fold {fold} - RF Probs')
        ax1.set_xlabel('Probability')
        ax1.set_ylabel('Count')
        ax1.axvline(0.05, color='r', linestyle='--', alpha=0.5)
        ax1.axvline(0.95, color='r', linestyle='--', alpha=0.5)
        
        # Plot Surrogate distribution
        ax2 = axes[1, fold-1]
        ax2.hist(surrogate_probs, bins=30, alpha=0.7, edgecolor='black', color='green')
        ax2.set_title(f'Fold {fold} - Surrogate Probs (R²={fold_results[fold-1]["test_r2"]:.3f})')
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('Count')
        ax2.axvline(0.05, color='r', linestyle='--', alpha=0.5)
        ax2.axvline(0.95, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('uncalibrated_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nProbability distributions saved to 'uncalibrated_distributions.png'")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    handler = DataHandling()
    handler.load_data()

    # Run uncalibrated surrogate pipeline
    fold_results, extreme_stats = surrogate_pipeline_uncalibrated(handler, max_depth=10)
    
    # Analyze results
    analyze_results(fold_results, extreme_stats)
    
    # Visualize distributions
    visualize_distributions(handler, fold_results)
    
    # Save best model
    best_fold = max(fold_results, key=lambda x: x["test_r2"])
    print(f"\nSaving best surrogate model (Fold {best_fold['fold']} with R²={best_fold['test_r2']:.4f})")
    joblib.dump(best_fold['surrogate'], 'models/surrogate.pkl')
    print("Model saved to 'models/surrogate.pkl'")