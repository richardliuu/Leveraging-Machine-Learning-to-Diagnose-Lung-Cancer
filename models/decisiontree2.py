"""
Improved Decision Tree Surrogate Model for Lung Cancer MLP Analysis

This module implements an optimized decision tree regressor that serves as an interpretable 
surrogate model for the Multi-Layer Perceptron (MLP) implemented in 
class_based_models/lung_cancer_mlp.py. This version addresses fidelity issues by:

- Relaxing tree constraints for better MLP approximation
- Adding consistent preprocessing (scaling) as used in MLP training
- Increasing validation split size for more reliable fidelity assessment
- Using optimal splitting strategy for better decision boundary fitting

Key Improvements over decisiontree.py:
1. Increased max_leaf_nodes from 15 to 100 for more complex decision boundaries
2. Removed max_depth constraint to allow deeper trees when needed
3. Reduced min_samples_split from 20 to 5 for more granular splits
4. Added StandardScaler preprocessing consistent with MLP training
5. Increased validation split from 15% to 25% for better fidelity assessment
6. Using splitter='best' for optimal split selection

Author: Richard Liu (Improved version)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shap
import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTEENN
import class_based_models.lung_cancer_mlp 

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
        self.scaler = StandardScaler()  # Add scaler for consistency with MLP

        # Data integrity tracking
        self.feature_cols = None
        self.duplicates = None
        self.patient_labels = None
        self.groups = None
        self.inconsistent_patients = None

    def load_data(self):
        self.data = pd.read_csv("data/surrogate_data.csv")
        
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

class DataPreprocess:
    def __init__(self):
        # Dataset attributes
        self.X = None
        self.y = None
        self.X_val = None
        self.y_val = None
        self.num_classes = None
        self.scaler = StandardScaler()  # Add scaler for preprocessing

    def split(self, X, y, data, train_idx, test_idx):
        self.X_train = X.iloc[train_idx]
        self.y_train = y.iloc[train_idx]
        self.X_test = X.iloc[test_idx]
        self.y_test = y.iloc[test_idx]

        # Track patient groups for validation
        self.train_patients = set(data.iloc[train_idx]['patient_id'])
        self.test_patients = set(data.iloc[test_idx]['patient_id'])

        self.num_classes = 2
    
    def transform(self):
        """Apply consistent preprocessing as used in MLP training"""
        # Scale features using StandardScaler (consistent with MLP)
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        print(f"Applied StandardScaler - Train mean: {self.X_train.mean().mean():.4f}, std: {self.X_train.std().mean():.4f}")
    
    def validation_split(self):
        # Create validation split with increased size for better fidelity assessment
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=0.25,  # Increased from 0.15 to 0.25
            random_state=SEED
        )
        
        print(f"Validation split - Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")

def pipeline(handler):
    gkf = GroupKFold(n_splits=4)
    all_fidelity_scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        print(f"\n=== Fold {fold + 1} ===")
        # Create preprocessor instance to handle data splitting and transformation
        preprocessor = DataPreprocess()
        preprocessor.split(handler.X, handler.y, handler.data, train_idx, test_idx)
        preprocessor.transform()
        preprocessor.validation_split()

        # Optimized DecisionTreeRegressor parameters for better fidelity
        model = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=None,           # Removed constraint - let tree grow as needed
            min_samples_leaf=3,       # Keep reasonable minimum
            min_samples_split=5,      # Reduced from 20 to 5 for more granular splits
            max_leaf_nodes=100,       # Increased from 15 to 100 for more complex boundaries
            splitter='best',          # Use best splits for optimal approximation
            random_state=SEED
        )
        
        model.fit(preprocessor.X_train, preprocessor.y_train, sample_weight=None)

        # Test set evaluation
        accuracy = model.score(preprocessor.X_test, preprocessor.y_test, sample_weight=None)
        y_pred_raw = model.predict(preprocessor.X_test)
        
        # Apply probability bounding to ensure predictions are in [0, 1] range
        y_pred = np.clip(y_pred_raw, 0.0, 1.0)
        
        # Validation: Check if any predictions were out of bounds
        out_of_bounds = np.sum((y_pred_raw < 0) | (y_pred_raw > 1))
        if out_of_bounds > 0:
            print(f"Warning: {out_of_bounds} predictions were clipped to [0,1] range")
            print(f"Raw prediction range: [{y_pred_raw.min():.4f}, {y_pred_raw.max():.4f}]")
        else:
            print("All predictions within [0,1] range - good fit!")
        
        print(f"Test R²: {accuracy:.4f}")
        print("Node Count:", model.tree_.node_count)
        print("Tree Depth:", model.tree_.max_depth)
        
        # Calculate fidelity on validation set
        val_pred_raw = model.predict(preprocessor.X_val)
        val_pred = np.clip(val_pred_raw, 0.0, 1.0)
        
        # Calculate regression metrics for validation
        val_mse = mean_squared_error(preprocessor.y_val, val_pred)
        val_mae = mean_absolute_error(preprocessor.y_val, val_pred)
        val_r2 = r2_score(preprocessor.y_val, val_pred)
        
        print(f"Validation MSE: {val_mse:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        
        all_fidelity_scores.append(val_r2)

    # Print overall fidelity statistics
    print(f"\n=== Overall Fidelity Results ===")
    print(f"Mean R²: {np.mean(all_fidelity_scores):.4f} ± {np.std(all_fidelity_scores):.4f}")
    print(f"Min R²: {np.min(all_fidelity_scores):.4f}")
    print(f"Max R²: {np.max(all_fidelity_scores):.4f}")

    return model, preprocessor  # Return both model and preprocessor

class SHAPAnalysis:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def run_shap(model, preprocessor, handler):
        # Convert validation data to DataFrame with feature names
        X_val_df = pd.DataFrame(preprocessor.X_val, columns=handler.feature_cols)
        print(f"Validation data shape: {X_val_df.shape}")

        # Initialize SHAP explainer for decision tree
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_df)
        print(f"SHAP values shape: {np.array(shap_values).shape}")

        # Generate SHAP summary plot for probability regression
        shap.summary_plot(shap_values, X_val_df)

class FidelityCheck:

    def __init__(self, surrogate_model, preprocessor):
        self.fidelity = None
        self.surrogate = surrogate_model
        self.preprocessor = preprocessor

    def comparison(self):
        try:
            # Load and initialize MLP model
            mlp_model = class_based_models.lung_cancer_mlp.LungCancerMLP(
                num_classes=self.preprocessor.num_classes,
                input_dim=self.preprocessor.X_val.shape[1]
            )
            
            # Generate MLP predictions on validation data
            mlp_preds = mlp_model.predict(self.preprocessor.X_val)
            print("MLP predictions shape:", mlp_preds.shape)
            
            # Generate surrogate predictions on validation data with probability bounding
            surrogate_preds_raw = self.surrogate.predict(self.preprocessor.X_val)
            surrogate_preds = np.clip(surrogate_preds_raw, 0.0, 1.0)
            
            # Validation: Check if any predictions were out of bounds
            out_of_bounds = np.sum((surrogate_preds_raw < 0) | (surrogate_preds_raw > 1))
            if out_of_bounds > 0:
                print(f"Validation Warning: {out_of_bounds} predictions were clipped to [0,1] range")
                print(f"Raw validation prediction range: [{surrogate_preds_raw.min():.4f}, {surrogate_preds_raw.max():.4f}]")
            else:
                print("All validation predictions within bounds!")
            
            print("Surrogate predictions shape:", surrogate_preds.shape)
            
            # Extract class 1 probabilities from MLP predictions if multi-class output
            if len(mlp_preds.shape) > 1 and mlp_preds.shape[1] > 1:
                mlp_class1_probs = mlp_preds[:, 1]  # Extract class 1 probabilities
            else:
                mlp_class1_probs = mlp_preds.flatten()
                
            print("MLP class 1 probabilities shape:", mlp_class1_probs.shape)
            
            # Calculate regression metrics for probability comparison
            self.fidelity_mse = mean_squared_error(mlp_class1_probs, surrogate_preds)
            self.fidelity_mae = mean_absolute_error(mlp_class1_probs, surrogate_preds)
            self.fidelity_r2 = r2_score(mlp_class1_probs, surrogate_preds)
            
            print(f"\n=== MLP vs Surrogate Fidelity ===")
            print(f"Fidelity MSE: {self.fidelity_mse:.4f}")
            print(f"Fidelity MAE: {self.fidelity_mae:.4f}")
            print(f"Fidelity R²: {self.fidelity_r2:.4f}")
            
            # Additional analysis
            correlation = np.corrcoef(mlp_class1_probs, surrogate_preds)[0, 1]
            print(f"Pearson Correlation: {correlation:.4f}")
            
            # Set fidelity as R² score (ranges from -∞ to 1, where 1 is perfect)
            self.fidelity = self.fidelity_r2
                
        except Exception as e:
            print(f"Fidelity check failed: {e}")
            print("Unable to compare with MLP model - using surrogate validation accuracy instead")
            self.fidelity = self.surrogate.score(self.preprocessor.X_val, self.preprocessor.y_val)
            print(f"Surrogate validation R²: {self.fidelity:.4f}")

if __name__ == "__main__":
    print("Running Improved Decision Tree Surrogate Model...")
    print("Key improvements: Relaxed constraints, added scaling, increased validation size\n")
    
    handler = DataHandling()
    handler.load_data()
    surrogate_model, preprocessor = pipeline(handler)

    check = FidelityCheck(surrogate_model, preprocessor)
    check.comparison()

    print(f"\nFinal fidelity score: {check.fidelity:.4f}")
    
    # Uncomment to run SHAP analysis
    # analyze = SHAPAnalysis(surrogate_model)
    # analyze.run_shap(surrogate_model, preprocessor, handler)