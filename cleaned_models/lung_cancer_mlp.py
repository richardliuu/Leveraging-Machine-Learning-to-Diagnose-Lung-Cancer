"""
Lung Cancer Classification using Multi-Layer Perceptron (MLP)

This module implements a deep learning approach for lung cancer stage classification
using a Multi-Layer Perceptron neural network with TensorFlow/Keras. The model
employs rigorous cross-validation techniques to prevent data leakage and ensure
robust performance evaluation.

Key Features:
- Patient-grouped cross-validation to prevent data leakage
- Data integrity checks for duplicate samples and label consistency
- SMOTEENN resampling to handle class imbalance
- Deep MLP architecture with batch normalization and dropout
- Comprehensive performance evaluation with ROC-AUC metrics
- Reproducible results with fixed random seeds

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- tensorflow: Deep learning framework
- matplotlib: Plotting and visualization
- scikit-learn: Machine learning utilities
- imbalanced-learn: Handling imbalanced datasets

Usage:
    python lung_cancer_mlp.py

The script expects a CSV file at 'data/binary_features_log.csv' containing
lung cancer data with features, patient IDs, and cancer stage labels.

Author: Science2 Project
"""

import pandas as pd
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import random
import os
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from imblearn.combine import SMOTEENN


# Configure deterministic behavior for reproducible results
os.environ['PYTHONHASHSEED'] = '42'  # Fixed hash seed for Python
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Enable TensorFlow deterministic operations

# Set random seeds for all libraries to ensure reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
# Limit parallelism to ensure deterministic behavior
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def data_integrity(df: pd.DataFrame) -> bool:
    """
    Perform comprehensive data integrity checks on the lung cancer dataset.
    
    This function validates the dataset by checking for duplicate samples and
    ensuring patient-label consistency across all samples. It also provides
    a summary of the patient-level class distribution.
    
    Args:
        df (pd.DataFrame): Input dataset containing features, patient IDs, and labels.
                          Expected columns: 'segment', 'cancer_stage', 'patient_id', 
                          plus feature columns.
    
    Returns:
        bool: True if data passes all integrity checks (no duplicates and consistent 
              patient labels), False otherwise.
    
    Checks Performed:
        1. Duplicate Detection: Identifies samples with identical feature values
        2. Patient-Label Consistency: Ensures each patient has consistent cancer stage labels
        3. Class Distribution: Reports patient-level distribution of cancer stages
    
    Side Effects:
        - Prints warnings for any integrity issues found
        - Displays patient-level class distribution
        - For inconsistent patients, shows example of label conflicts
    
    Example:
        >>> df = pd.read_csv('lung_cancer_data.csv')
        >>> is_clean = data_integrity(df)
        >>> if is_clean:
        ...     print("Data passed all integrity checks")
    """
    # Check for duplicate samples across all feature columns
    feature_cols = df.drop(columns=['segment', 'cancer_stage', 'patient_id']).columns
    duplicates = df.duplicated(subset=feature_cols)
    
    # Verify each patient has consistent cancer stage labels across all samples
    patient_labels = df.groupby('patient_id')['cancer_stage'].nunique()
    inconsistent_patients = patient_labels[patient_labels > 1]  # Patients with >1 unique label
    
    if len(inconsistent_patients) > 0:
        print(f"WARNING: {len(inconsistent_patients)} patients have inconsistent labels!")
        print("Inconsistent patients:", inconsistent_patients.index.tolist()[:10])
        # Show details for first inconsistent patient
        if len(inconsistent_patients) > 0:
            first_patient = inconsistent_patients.index[0]
            patient_data = df[df['patient_id'] == first_patient][['patient_id', 'cancer_stage']]
            print(f"Example - Patient {first_patient}:")
            print(patient_data['cancer_stage'].value_counts())
    
    # Calculate patient-level class distribution using mode (most frequent label per patient)
    patient_df = df.groupby('patient_id').agg({
        'cancer_stage': lambda x: x.mode()[0]  # Take most frequent label for each patient
    }).reset_index()
    print(f"\nPatient-level class distribution:")
    patient_class_counts = patient_df['cancer_stage'].value_counts()
    print(patient_class_counts)
    
    return duplicates.sum() == 0 and len(inconsistent_patients) == 0

def lung_cancer_model(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[Dict[str, Any]], List[Dict[str, List[float]]], List[float]]:
    """
    Train and evaluate a Multi-Layer Perceptron for lung cancer stage classification.
    
    This function implements a comprehensive machine learning pipeline including:
    - Patient-grouped k-fold cross-validation to prevent data leakage
    - Data preprocessing with standardization and label encoding
    - Class imbalance handling using SMOTEENN resampling
    - Deep neural network training with regularization techniques
    - Performance evaluation with multiple metrics
    
    Args:
        df (pd.DataFrame): Input dataset with features, patient IDs, and cancer stage labels.
                          Must contain columns: 'segment', 'cancer_stage', 'patient_id'
                          plus numerical feature columns.
    
    Returns:
        tuple: A 5-tuple containing:
            - all_reports (list): Classification reports for each fold
            - all_conf_matrices (list): Confusion matrices for each fold  
            - fold_details (list): Detailed statistics for each fold
            - all_histories (list): Training histories for each fold
            - all_roc_aucs (list): ROC-AUC scores for each fold
    
    Model Architecture:
        - Input layer matching number of features
        - 4 hidden layers: 512, 256, 128, 64 neurons
        - ReLU activation with batch normalization and dropout
        - Softmax output layer for multi-class classification
        - Adam optimizer with categorical crossentropy loss
    
    Cross-Validation Strategy:
        - 4-fold GroupKFold to ensure patients don't appear in both train/test
        - SMOTEENN resampling applied only to training data
        - Early stopping and learning rate reduction callbacks
        - Validation split (10%) from resampled training data
    
    Global Variables Modified:
        - history: Training history from the last fold
        - y_train_final: Final training labels from the last fold
    
    Example:
        >>> df = pd.read_csv('lung_cancer_data.csv')
        >>> reports, matrices, details, histories, aucs = lung_cancer_model(df)
        >>> print(f"Average accuracy: {np.mean([r['accuracy'] for r in reports]):.4f}")
    """
    global history 
    global y_train_final
    
    # Prepare features and labels for machine learning
    X = df.drop(columns=['segment', 'cancer_stage', 'patient_id'])  # Feature matrix
    y = df['cancer_stage']  # Target labels
    groups = df['patient_id']  # Patient IDs for grouped cross-validation

    # Use GroupKFold to ensure patients don't appear in both train and test sets
    group_kfold = GroupKFold(n_splits=4)
    
    all_reports = []
    all_conf_matrices = []
    fold_details = []
    all_histories = []
    all_roc_aucs = [] 
    
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
        
        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        
        # Verify no patient leakage between train and test sets
        train_patients = set(df.iloc[train_idx]['patient_id'])
        test_patients = set(df.iloc[test_idx]['patient_id'])
        assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage detected!"

        # Fit preprocessing ONLY on training data to prevent data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)  # Fit on train, transform train
        X_test_scaled = scaler.transform(X_test_fold)  # Only transform test (no fitting)
        
        # Encode categorical labels to numerical values (fit only on training data)
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_fold)  # Fit on train labels
        y_test_encoded = le.transform(y_test_fold)  # Transform test labels
        
        num_classes = len(le.classes_)
        
        # Apply SMOTEENN resampling only to training data to handle class imbalance
        # SMOTEENN combines SMOTE (oversampling minority) with ENN (undersampling majority)
        smote = SMOTEENN(random_state=SEED)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)
        
        # Convert integer labels to one-hot encoded categorical format for neural network
        y_train_cat = to_categorical(y_train_resampled, num_classes=num_classes)
        y_test_cat = to_categorical(y_test_encoded, num_classes=num_classes)
        
        # Create validation split from resampled training data (10% for validation)
        # Stratify to maintain class distribution in train/validation splits
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train_resampled, y_train_cat, test_size=0.1, 
            stratify=y_train_resampled, random_state=SEED
        )
        
        # Build deep MLP model (fresh model for each fold to avoid bias)
        # Architecture: Progressive reduction in layer sizes with regularization
        model = Sequential([
            Input(shape=(X_train_final.shape[1],)),  # Input layer matching feature count
            
            # First hidden layer: 512 neurons with regularization
            Dense(512, activation='relu'),
            BatchNormalization(),  # Normalize inputs to next layer
            Dropout(0.4),  # Prevent overfitting by randomly dropping 40% of neurons
            
            # Second hidden layer: 256 neurons
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third hidden layer: 128 neurons
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Fourth hidden layer: 64 neurons
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer: Number of classes with softmax for probability distribution
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Stop when validation loss stops improving
            patience=5,  # Wait 5 epochs before stopping
            restore_best_weights=True  # Restore weights from best epoch
            )
        
        # Learning rate reduction when validation loss plateaus
        lr_schedule = ReduceLROnPlateau(
            monitor='val_loss',  # Monitor validation loss
            factor=0.1,  # Reduce learning rate by factor of 10
            patience=3,  # Wait 3 epochs before reducing
            verbose=1  # Print when learning rate is reduced
            )
        
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val_final, y_val_final),
            epochs=50,
            batch_size=16,
            callbacks=[early_stopping, lr_schedule],
            verbose=1
        )
        
        # Make predictions on test set
        y_pred_prob = model.predict(X_test_scaled, verbose=0)  # Get class probabilities
        y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions
        
        # Generate comprehensive classification report with precision, recall, F1-score
        target_names = [str(cls) for cls in le.classes_]  # Use original class names
        report = classification_report(y_test_encoded, y_pred, target_names=target_names, output_dict=True)
        
        print(f"\nFold {fold+1} Results:")
        print(classification_report(y_test_encoded, y_pred, target_names=target_names))
        
        c_matrix = confusion_matrix(y_test_encoded, y_pred)
        print("Confusion Matrix:")
        print(c_matrix)

        try:
            # Calculate ROC-AUC score for binary classification
            y_true = np.argmax(y_test_cat, axis=1)  # Convert one-hot back to labels
            y_score = y_pred_prob[:, 1]  # Use probability of positive class

            auc = roc_auc_score(y_true, y_score)
            print(f"ROC AUC Score: {auc:.4f}")
        except Exception as e:
            # Handle cases where ROC-AUC cannot be computed (e.g., single class in test set)
            print("ROC AUC could not be computed:", str(e))
            auc = np.nan

        all_roc_aucs.append(auc)
        all_reports.append(report)
        all_conf_matrices.append(c_matrix)
        all_histories.append(history.history)
        
        fold_details.append({
            'fold': fold + 1,
            'train_patients': len(train_patients),
            'test_patients': len(test_patients),
            'train_samples': len(X_train_fold),
            'test_samples': len(X_test_fold),
            'accuracy': report['accuracy'],
            'epochs_trained': len(history.history['loss'])
        })
    
    return all_reports, all_conf_matrices, fold_details, all_histories, all_roc_aucs

def summarize_results(all_reports: List[Dict[str, Any]], all_conf_matrices: List[np.ndarray], fold_details: List[Dict[str, Any]], all_histories: List[Dict[str, List[float]]], all_roc_aucs: List[float]) -> None:
    """
    Generate comprehensive summary and visualizations of cross-validation results.
    
    This function processes the results from all cross-validation folds to provide:
    - Statistical summaries of model performance across folds
    - Visualizations of training/validation curves for each fold
    - Class-wise performance metrics (F1-scores)
    - ROC-AUC analysis with confidence intervals
    - Average confusion matrix across all folds
    
    Args:
        all_reports (list): List of classification report dictionaries from each fold
        all_conf_matrices (list): List of confusion matrices (numpy arrays) from each fold
        fold_details (list): List of dictionaries containing fold-specific statistics
        all_histories (list): List of training history dictionaries from each fold
        all_roc_aucs (list): List of ROC-AUC scores from each fold
    
    Returns:
        None: Function performs analysis and displays results via print statements
              and matplotlib plots.
    
    Visualizations Created:
        1. Training/validation loss and accuracy curves for each fold
        2. ROC-AUC scores across folds with mean and standard deviation
    
    Metrics Reported:
        - Per-fold accuracy and training statistics
        - Overall mean ± standard deviation of accuracies
        - Min/max accuracy across folds
        - Class-wise F1-scores with statistics
        - Average confusion matrix
        - ROC-AUC performance with confidence intervals
    
    Example:
        >>> reports, matrices, details, histories, aucs = lung_cancer_model(df)
        >>> summarize_results(reports, matrices, details, histories, aucs)
        # Displays comprehensive performance analysis
    """
    
    # Per-fold summary
    accuracies = [report['accuracy'] for report in all_reports]
    epochs_trained = [fold['epochs_trained'] for fold in fold_details]
    
    print("Per-fold results:")
    for i, (acc, details) in enumerate(zip(accuracies, fold_details)):
        print(f"Fold {i+1}: {acc:.4f} accuracy "
              f"({details['test_patients']} patients, {details['test_samples']} samples, "
              f"{details['epochs_trained']} epochs)")

    for i, history in enumerate(all_histories):
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Fold {i+1} Performance", fontsize=14)
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Train Acc')
        plt.plot(history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"\nOverall Performance:")
    print(f"Mean Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Min Accuracy:  {min(accuracies):.4f}")
    print(f"Max Accuracy:  {max(accuracies):.4f}")
    
    class_0_f1 = [report['0']['f1-score'] for report in all_reports]
    class_1_f1 = [report['1']['f1-score'] for report in all_reports]
    
    print(f"\nClass-wise F1-scores:")
    print(f"Class 0: {np.mean(class_0_f1):.4f} ± {np.std(class_0_f1):.4f}")
    print(f"Class 1: {np.mean(class_1_f1):.4f} ± {np.std(class_1_f1):.4f}")
    
    avg_conf_matrix = np.mean(all_conf_matrices, axis=0)
    print(f"\nAverage Confusion Matrix:")
    print(np.round(avg_conf_matrix).astype(int))
    
    # ROC AUC Line Plot
    if all_roc_aucs:
        auc_scores = [score for score in all_roc_aucs if not np.isnan(score)]
        if auc_scores:
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)

            print(f"\nMean ROC AUC (macro): {mean_auc:.4f} ± {std_auc:.4f}")

            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(auc_scores) + 1), auc_scores, marker='o', linestyle='-',
                     color='blue', label='ROC AUC per Fold')
            plt.axhline(mean_auc, color='red', linestyle='--', label=f'Mean AUC = {mean_auc:.4f}')
            plt.fill_between(range(1, len(auc_scores) + 1),
                             [mean_auc - std_auc] * len(auc_scores),
                             [mean_auc + std_auc] * len(auc_scores),
                             color='red', alpha=0.2, label='±1 STD')
            plt.xticks(range(1, len(auc_scores) + 1))
            plt.xlabel("Fold")
            plt.ylabel("ROC AUC (macro)")
            plt.title("ROC AUC per Fold (Macro)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Load the lung cancer dataset
    df = pd.read_csv("data/binary_features_log.csv")
    
    # Perform data integrity checks before modeling
    is_clean = data_integrity(df)
    
    # Train and evaluate the MLP model using cross-validation
    results = lung_cancer_model(df)
    
    # Unpack results from cross-validation
    all_reports, all_conf_matrices, fold_details, all_histories, all_roc_aucs = results
    
    # Generate comprehensive summary and visualizations of results
    summarize_results(all_reports, all_conf_matrices, fold_details, all_histories, all_roc_aucs)    