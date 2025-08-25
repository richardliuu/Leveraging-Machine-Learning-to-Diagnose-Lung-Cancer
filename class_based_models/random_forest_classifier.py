"""
Random Forest Classification for Lung Cancer Stage Classification

This module implements a Random Forest approach for lung cancer stage classification
using scikit-learn's RandomForestClassifier. The model employs rigorous cross-validation
techniques to prevent data leakage and ensure robust performance evaluation.

Key Features:
- Patient-grouped cross-validation to prevent data leakage
- Data integrity checks for duplicate samples and label consistency
- Balanced class weighting to handle class imbalance
- Comprehensive performance evaluation with ROC-AUC metrics
- Feature importance analysis for model interpretability
- Reproducible results with fixed random seeds
- Model persistence with joblib

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- scikit-learn: Machine learning utilities and Random Forest implementation
- matplotlib: Plotting and visualization
- joblib: Model serialization and persistence
- See requirements.txt for all dependencies required

Usage:
    python random_forest_classifier.py

The script expects a CSV file at 'data/jitter_shimmerlog.csv' containing
lung cancer data with features, patient IDs, and cancer stage labels.

Author: Adapted from models/randfor.py
"""

import pandas as pd
import numpy as np
import random
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Seed setting for reproducibility
os.environ['PYTHONHASHSEED'] = '42'
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

"""
The DataHandling Class handles and transforms the Random Forest performance data into training data. 

The instantiated variables for training and testing are managed through class functions.
Data Splits are handled in split().

This class encapsulates all data preprocessing steps including:
- Data integrity checks for duplicate samples and label consistency
- Feature selection and column dropping
- Patient-grouped cross-validation splits
- Class distribution analysis

Attributes:
    data (str): Path to the input CSV file
    
    Storage for cross-validation results:
    - reports: Classification reports from each fold
    - conf_matrices: Confusion matrices from each fold
    - roc_aucs: ROC-AUC scores from each fold
    - fold_details: Fold-specific statistics and metadata
    - predictions: Model predictions from each fold
"""

class DataHandling:
    def __init__(self):
        self.data = "data/jitter_shimmerlog.csv"
        
        # Storage for cross-validation results
        self.reports = []
        self.conf_matrices = []
        self.roc_aucs = []
        self.fold_details = []
        self.predictions = []
        
        # Data attributes
        self.X = None
        self.y = None
        self.groups = None
        self.feature_cols = None
        
        # Current fold data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_patients = None
        self.test_patients = None
        
    def load_data(self):
        """
        Load and perform comprehensive data integrity checks on the lung cancer dataset.
        
        This method loads the dataset, extracts features and labels, and validates data quality
        by checking for duplicate samples and ensuring patient-label consistency across all samples.
        It also provides a summary of the class distribution.
        
        Returns:
            bool: True if data passes all integrity checks (no duplicates and consistent 
                  patient labels), False otherwise.
        
        Checks Performed:
            1. Duplicate Detection: Identifies samples with identical feature values
            2. Patient-Label Consistency: Ensures each patient has consistent cancer stage labels
            3. Class Distribution: Reports overall distribution of cancer stages
        
        Side Effects:
            - Prints data loading statistics and warnings for any integrity issues found
            - Sets class attributes: X, y, feature_cols, groups
        
        Data Structure Expected:
            CSV file with columns: 'chunk', 'cancer_stage', 'patient_id', 'filename', 
            'rolloff', 'bandwidth', 'skew', 'zcr', 'rms', plus feature columns
        
        Example:
            >>> handler = DataHandling()
            >>> is_clean = handler.load_data()
            >>> if is_clean:
            ...     print("Data passed all integrity checks")
        """
        print("Loading dataset...")
        df = pd.read_csv(self.data)
        print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")
        
        # Prepare features and labels - drop metadata columns
        self.X = df.drop(columns=['chunk', 'cancer_stage', 'patient_id', 'filename', 
                                 'rolloff', 'bandwidth', "skew", "zcr", 'rms'])
        self.y = df['cancer_stage']
        self.groups = df['patient_id']
        self.feature_cols = self.X.columns.tolist()
        
        print(f"Total samples: {len(df)}")
        print(f"Total patients: {df['patient_id'].nunique()}")
        print(f"Features: {self.X.shape[1]}")
        
        # Perform data integrity verification
        return self._verify_data_integrity(df)
    
    def _verify_data_integrity(self, df):
        """
        Verify data integrity by checking for duplicates and inconsistent patient labels.
        
        Args:
            df (pd.DataFrame): The loaded dataset
            
        Returns:
            bool: True if data passes all checks, False otherwise
        
        Private method that performs:
            - Duplicate sample detection across feature columns
            - Patient label consistency verification
            - Class distribution analysis and reporting
        """
        print("\n=== DATA INTEGRITY CHECKS ===")
        
        # Check for duplicate samples
        duplicates = df.duplicated(subset=self.feature_cols)
        print(f"Duplicate feature rows: {duplicates.sum()}")
        
        if duplicates.sum() > 0:
            print("WARNING: Duplicate samples found!")
            dup_rows = df[duplicates]
            print(f"Example duplicate patients: {dup_rows['patient_id'].unique()[:5]}")
        else:
            print("No duplicate feature rows found")
        
        # Check for inconsistent patient labels
        patient_labels = df.groupby('patient_id')['cancer_stage'].nunique()
        inconsistent_patients = patient_labels[patient_labels > 1]
        if len(inconsistent_patients) > 0:
            print(f"WARNING: {len(inconsistent_patients)} patients have inconsistent labels!")
            print("Inconsistent patients:", inconsistent_patients.index.tolist()[:10])
        else:
            print("All patients have consistent labels")
        
        # Display class distribution
        print(f"\nOverall class distribution:")
        class_counts = df['cancer_stage'].value_counts()
        print(class_counts)
        print(f"Class ratio: {class_counts.iloc[0]/class_counts.iloc[1]:.2f}:1")
        
        return duplicates.sum() == 0 and len(inconsistent_patients) == 0
    
    def split(self, df, train_idx, test_idx):
        """
        Split the dataset into training and testing sets for the current fold.
        
        Args:
            df (pd.DataFrame): The complete dataset
            train_idx (np.ndarray): Indices for training samples
            test_idx (np.ndarray): Indices for testing samples
        
        Side Effects:
            Sets the following class attributes:
            - X_train, X_test: Training and testing feature matrices
            - y_train, y_test: Training and testing labels
            - train_patients, test_patients: Sets of patient IDs for each split
        
        This method handles the data splitting for GroupKFold cross-validation,
        ensuring that patient grouping is maintained.
        """
        self.X_train, self.X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
        self.y_train, self.y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
        
        # Track patients in each split to verify no leakage
        self.train_patients = set(df.iloc[train_idx]['patient_id'])
        self.test_patients = set(df.iloc[test_idx]['patient_id'])


class RandomForestModel:
    """
    Random Forest classifier for lung cancer stage classification.
    
    This class implements a Random Forest model with configurable hyperparameters
    optimized for medical classification tasks. The model includes balanced class
    weights to handle potential class imbalance and provides feature importance analysis.
    
    Attributes:
        model (RandomForestClassifier): The scikit-learn Random Forest model
        feature_cols (list): Names of the feature columns for interpretability
    
    Model Configuration:
        - criterion: "log_loss" for probabilistic splits
        - n_estimators: 200 trees for stable predictions
        - max_depth: 5 to prevent overfitting on medical data
        - min_samples_split: 12 to ensure robust splits
        - min_samples_leaf: 3 for stable leaf predictions
        - class_weight: 'balanced' to handle class imbalance
        - random_state: Fixed seed for reproducibility
        - n_jobs: -1 to use all available processors
    """
    
    def __init__(self):
        """
        Initialize the Random Forest model with optimized hyperparameters.
        
        The hyperparameters are tuned for medical classification tasks where:
        - Preventing overfitting is crucial (controlled depth and samples)
        - Class imbalance may exist (balanced weights)
        - Interpretability is important (moderate number of trees)
        - Reproducibility is required (fixed random state)
        """
        self.model = RandomForestClassifier(
            criterion="log_loss",
            n_estimators=200,
            max_depth=5,
            max_features=None,
            min_samples_split=12,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
        )
        self.feature_cols = None
    
    def train(self, X_train, y_train, feature_cols):
        """
        Train the Random Forest model on the provided training data.
        
        Args:
            X_train (pd.DataFrame): Training feature matrix
            y_train (pd.Series): Training labels
            feature_cols (list): Names of feature columns for interpretability
        
        Side Effects:
            - Fits the Random Forest model to the training data
            - Stores feature column names for later interpretation
        
        The training process uses all specified hyperparameters and leverages
        scikit-learn's efficient Random Forest implementation with parallel processing.
        """
        self.feature_cols = feature_cols
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """
        Generate class predictions for test data.
        
        Args:
            X_test (pd.DataFrame): Test feature matrix
            
        Returns:
            np.ndarray: Predicted class labels
        
        Uses the trained Random Forest to make discrete class predictions
        based on majority voting across all trees in the ensemble.
        """
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """
        Generate class probability predictions for test data.
        
        Args:
            X_test (pd.DataFrame): Test feature matrix
            
        Returns:
            np.ndarray: Probability predictions for each class
        
        Returns the probability estimates for each class, averaged across
        all trees in the Random Forest ensemble.
        """
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance on test data with comprehensive metrics.
        
        This method computes multiple performance metrics to provide a thorough
        assessment of model performance for medical classification tasks.
        
        Args:
            X_test (pd.DataFrame): Test feature matrix
            y_test (pd.Series): True test labels
            
        Returns:
            tuple: A 3-tuple containing:
                - report (dict): Comprehensive classification report with precision,
                  recall, F1-score, and support for each class
                - c_matrix (np.ndarray): Confusion matrix showing prediction accuracy
                - auc (float): ROC-AUC score for binary classification performance
        
        Metrics Computed:
            1. Classification Report:
                - Per-class precision, recall, F1-score
                - Macro and weighted averages
                - Support (number of samples per class)
            
            2. Confusion Matrix:
                - True vs predicted class counts
                - Diagonal elements show correct classifications
                - Off-diagonal elements show misclassifications
            
            3. ROC-AUC Score:
                - Area under the ROC curve for binary classification
                - Measures model's ability to distinguish between classes
                - Values closer to 1.0 indicate better discriminative performance
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Generate comprehensive classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        c_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC-AUC score with error handling
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = np.nan
        
        return report, c_matrix, auc
    
    def get_feature_importance(self):
        """
        Get feature importance scores from the trained Random Forest.
        
        Returns:
            pd.DataFrame: DataFrame with features and their importance scores,
                         sorted by importance in descending order
        
        Feature Importance:
            - Based on mean decrease in impurity across all trees
            - Higher values indicate more important features for classification
            - Useful for understanding which biomarkers drive predictions
            - Can guide feature selection for future model iterations
        
        Note:
            Requires the model to be trained and feature_cols to be set.
        """
        if self.feature_cols is not None:
            importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None


def pipeline(handler):
    """
    Execute the complete machine learning pipeline with patient-grouped cross-validation.
    
    This function implements a comprehensive ML pipeline that trains and evaluates
    a Random Forest classifier for lung cancer stage classification using rigorous
    cross-validation techniques to prevent data leakage and ensure robust performance.
    
    Args:
        handler (DataHandling): Initialized DataHandling object containing loaded
                               dataset and preprocessing configurations.
    
    Pipeline Steps (per fold):
        1. Patient-grouped data splitting to prevent leakage
        2. Patient leakage verification
        3. Model initialization and training
        4. Comprehensive evaluation with multiple metrics
        5. Results storage and performance logging
        6. Feature importance analysis (final fold only)
    
    Cross-Validation Strategy:
        - 4-fold GroupKFold ensures patients don't appear in both train/test
        - Each fold trains a fresh model to avoid bias
        - Performance metrics aggregated across all folds
        - Real-time progress reporting during evaluation
    
    Results Storage:
        Updates handler attributes with:
        - reports: Classification reports with precision/recall/F1
        - conf_matrices: Confusion matrices showing prediction accuracy
        - roc_aucs: ROC-AUC scores for model discrimination ability
        - fold_details: Fold metadata including sample counts and accuracy
        - predictions: Model predictions for each fold
    
    Performance Monitoring:
        - Patient leakage detection and termination if found
        - Real-time accuracy statistics during cross-validation
        - Confusion matrix display for each fold
        - Comprehensive summary statistics across all folds
        - Feature importance analysis from final model
    
    Example:
        >>> handler = DataHandling()
        >>> handler.load_data()
        >>> pipeline(handler)
        # Executes complete 4-fold cross-validation pipeline
    """
    print("\nRunning Random Forest Cross-Validation...")
    
    # Load the full dataset for patient tracking
    df = pd.read_csv(handler.data)
    
    # Initialize GroupKFold for patient-grouped cross-validation
    group_kfold = GroupKFold(n_splits=4)
    
    # Execute cross-validation
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(handler.X, handler.y, handler.groups)):
        print(f"\n{'='*50}\nFOLD {fold+1}/4\n{'='*50}")
        
        # Split data for current fold
        handler.split(df, train_idx, test_idx)
        
        # Critical: Verify no patient leakage between train and test sets
        overlap = handler.train_patients.intersection(handler.test_patients)
        if overlap:
            print(f"CRITICAL: Patient leakage detected! {overlap}")
            return None
        
        print(f"Train: {len(handler.train_patients)} patients, {len(handler.X_train)} samples")
        print(f"Test:  {len(handler.test_patients)} patients, {len(handler.X_test)} samples")
        
        # Initialize and train Random Forest model
        model = RandomForestModel()
        model.train(handler.X_train, handler.y_train, handler.feature_cols)
        
        # Evaluate model performance
        report, c_matrix, auc = model.evaluate(handler.X_test, handler.y_test)
        
        # Store predictions for analysis
        y_pred = model.predict(handler.X_test)
        handler.predictions.append(y_pred)
        
        # Store evaluation results
        handler.reports.append(report)
        handler.conf_matrices.append(c_matrix)
        handler.roc_aucs.append(auc)
        handler.fold_details.append({
            'fold': fold + 1,
            'train_patients': len(handler.train_patients),
            'test_patients': len(handler.test_patients),
            'train_samples': len(handler.X_train),
            'test_samples': len(handler.X_test),
            'accuracy': report['accuracy']
        })
        
        # Display fold results
        print(f"\nFold {fold+1} Results:")
        print(classification_report(handler.y_test, y_pred))
        print("Confusion Matrix:")
        print(c_matrix)
        print(f"ROC AUC Score: {auc:.4f}")
        
        # Store final model for feature importance analysis
        if fold == 3:  # Last fold
            final_model = model
    
    # ==================== SUMMARY STATISTICS ====================
    print(f"\n{'='*60}\nCROSS-VALIDATION SUMMARY\n{'='*60}")
    
    # Calculate accuracy statistics
    accuracies = [r['accuracy'] for r in handler.reports]
    print("Per-fold results:")
    for i, (acc, details) in enumerate(zip(accuracies, handler.fold_details)):
        print(f"Fold {i+1}: {acc:.4f} accuracy "
              f"({details['test_patients']} patients, {details['test_samples']} samples)")
    
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"\nOverall Performance:")
    print(f"Mean Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Min Accuracy: {min(accuracies):.4f}")
    print(f"Max Accuracy: {max(accuracies):.4f}")
    
    # Calculate class-wise F1 scores
    class_0_f1 = [r['0']['f1-score'] for r in handler.reports]
    class_1_f1 = [r['1']['f1-score'] for r in handler.reports]
    print(f"\nClass-wise F1-scores:")
    print(f"Class 0: {np.mean(class_0_f1):.4f} ± {np.std(class_0_f1):.4f}")
    print(f"Class 1: {np.mean(class_1_f1):.4f} ± {np.std(class_1_f1):.4f}")
    
    # Plot F1 scores per fold
    plt.figure(figsize=(10, 6))
    folds = list(range(1, len(class_0_f1) + 1))
    
    plt.plot(folds, class_0_f1, 'o-', label='Class 0 F1-score', linewidth=2, markersize=8)
    plt.plot(folds, class_1_f1, 's-', label='Class 1 F1-score', linewidth=2, markersize=8)
    
    plt.xlabel('Fold')
    plt.ylabel('F1-score')
    plt.title('F1-score per Fold for Random Forest Classifier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(folds)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    # Display average confusion matrix
    avg_conf_matrix = np.mean(handler.conf_matrices, axis=0)
    print(f"\nAverage Confusion Matrix:")
    print(np.round(avg_conf_matrix).astype(int))
    
    # Display feature importance from final model
    print("\nFeature Importance (Top 10):")
    importance_df = final_model.get_feature_importance()
    if importance_df is not None:
        print(importance_df.head(10))
    
    # Optionally save the final model
    # joblib.dump(final_model.model, "models/rf_model.pkl")
    # print("Random Forest model saved as rf_model.pkl")
    
    return handler


def main():
    """
    Main execution function for the Random Forest classification pipeline.
    
    This function orchestrates the complete workflow:
    1. Initialize data handling
    2. Load and validate dataset
    3. Execute cross-validation pipeline
    4. Display comprehensive results
    
    The function serves as the entry point for running the Random Forest
    classification analysis on lung cancer staging data.
    """
    print("Random Forest Classification for Lung Cancer Stage Prediction")
    print("=" * 65)
    
    # Initialize data handler
    handler = DataHandling()
    
    # Step 1: Load and validate data
    print("Step 1: Data Loading and Integrity Checks")
    is_clean = handler.load_data()
    
    if not is_clean:
        print("\nDATA QUALITY ISSUES DETECTED! Results may be unreliable")
    
    # Step 2: Execute pipeline
    print("\nStep 2: Cross-Validation Pipeline")
    results = pipeline(handler)
    
    if results is None:
        print("\nPipeline failed due to patient leakage!")
        return
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()