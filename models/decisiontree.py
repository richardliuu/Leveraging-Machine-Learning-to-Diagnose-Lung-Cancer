"""
Decision Tree Surrogate Model for Lung Cancer MLP Analysis

This module implements a decision tree classifier that serves as an interpretable 
surrogate model for the Multi-Layer Perceptron (MLP) implemented in 
class_based_models/lung_cancer_mlp.py. The primary purpose is to analyze and 
understand the black-box behavior of the deep learning model through transparent 
decision tree predictions.

Surrogate Model Approach:
The decision tree is trained to mimic the predictions of the trained MLP model,
providing interpretable insights into how the neural network makes decisions.
This white-box approach allows clinicians and researchers to understand the 
decision-making process that would otherwise be opaque in the MLP.

Key Features:
- Patient-grouped cross-validation to prevent data leakage
- Decision tree trained on MLP predictions rather than true labels
- SHAP analysis for feature importance and decision path visualization
- Model fidelity assessment to ensure surrogate accuracy
- Comprehensive tree visualization and rule extraction
- SMOTEENN resampling for balanced surrogate training
- Reproducible results with fixed random seeds

Surrogate Model Benefits:
1. Interpretability: Provides clear decision rules and feature thresholds
2. Validation: Helps verify that MLP decisions are medically reasonable
3. Feature Analysis: Identifies key biomarkers driving predictions
4. Clinical Translation: Offers explainable AI for medical decision support
5. Model Debugging: Reveals potential biases or errors in MLP behavior

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- scikit-learn: Decision tree implementation and evaluation metrics
- matplotlib: Tree visualization and plotting
- imbalanced-learn: Handling class imbalance with SMOTEENN
- shap: Model interpretability and feature importance analysis

Data Requirements:
The surrogate model expects 'data/surrogate_data.csv' containing:
- Feature columns: Same features used to train the original MLP
- predicted_label: MLP predictions to be mimicked by the surrogate
- true_label: Original ground truth labels for validation
- patient_id: Patient identifiers for grouped cross-validation
- segment: Data segment identifiers

Usage:
    python models/decisiontree.py

The script trains decision tree surrogates using 4-fold cross-validation,
evaluates surrogate fidelity to the MLP, and generates interpretability
visualizations including tree plots and SHAP analysis.

Author: Richard Liu
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

        # Apply SMOTEENN resampling to training data
        """
        self.X_train, self.y_train = SMOTEENN(
            sampling_strategy='auto', 
            random_state=SEED
        ).fit_resample(self.X_train, self.y_train)
        """

def pipeline(handler):
    gkf = GroupKFold(n_splits=4)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        handler.split(handler.X, handler.y, handler.data, train_idx, test_idx)
        handler.transform()
        handler.validation_split()

        model = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=10,
            min_samples_leaf=3,
            min_samples_split=20,
            max_leaf_nodes=15,
            splitter='best',
            random_state=SEED
        )
        
        model.fit(handler.X_train, handler.y_train, sample_weight=None)

        #prob = model.predict_proba(handler.X_test)
        #print(prob)

        accuracy = model.score(handler.X_test, handler.y_test, sample_weight=None)
        y_pred_raw = model.predict(handler.X_test)
        # Apply probability bounding to ensure predictions are in [0, 1] range
        y_pred = np.clip(y_pred_raw, 0.0, 1.0)
        
        # Validation: Check if any predictions were out of bounds
        out_of_bounds = np.sum((y_pred_raw < 0) | (y_pred_raw > 1))
        if out_of_bounds > 0:
            print(f"Warning: {out_of_bounds} predictions were clipped to [0,1] range")
            print(f"Raw prediction range: [{y_pred_raw.min():.4f}, {y_pred_raw.max():.4f}]")
        
        print(np.unique(y_pred, return_counts=True))

        #c_matrix = confusion_matrix(handler.y_test, y_pred)

        #print(classification_report(handler.y_test, y_pred, target_names=[str(cls) for cls in handler.encoder.classes_]))
        print(accuracy)
        #print(c_matrix)
        print("Node Size", model.tree_.node_count)
        print("Max Depth", model.tree_.max_depth)
        #plot_tree(model, feature_names=handler.feature_cols, class_names=[str(cls) for cls in handler.encoder.classes_], filled=True)

        """ Uncomment if graphviz code is required to visualize the tree """
        print(export_graphviz(model, feature_names=handler.feature_cols))

    return model

class SHAPAnalysis:
    def __init__(self, model):
        self.model = model

    def run_shap(model):
        # Convert validation data to DataFrame with feature names
        X_val_df = pd.DataFrame(handler.X_val, columns=handler.feature_cols)
        print(f"Validation data shape: {X_val_df.shape}")

        # Initialize SHAP explainer for decision tree
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_df)
        print(f"SHAP values shape: {np.array(shap_values).shape}")

        # Generate SHAP summary plot for class 1 (positive cancer stage)
        # Focuses on features that increase probability of higher cancer stage
        shap.summary_plot(shap_values, X_val_df)

class FidelityCheck:

    def __init__(self, surrogate_model, data_handler):
        self.fidelity = None
        self.surrogate = surrogate_model
        self.handler = data_handler

    def comparison(self):

        try:
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
            
            # Calculate regression metrics for probability comparison
            self.fidelity_mse = mean_squared_error(mlp_class1_probs, surrogate_preds)
            self.fidelity_mae = mean_absolute_error(mlp_class1_probs, surrogate_preds)
            self.fidelity_r2 = r2_score(mlp_class1_probs, surrogate_preds)
            
            print(f"Fidelity MSE: {self.fidelity_mse:.4f}")
            print(f"Fidelity MAE: {self.fidelity_mae:.4f}")
            print(f"Fidelity R²: {self.fidelity_r2:.4f}")
            
            # Set fidelity as R² score (ranges from -∞ to 1, where 1 is perfect)
            self.fidelity = self.fidelity_r2
                
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

    analyze = SHAPAnalysis()
    analyze.run_shap()

