"""
Lung Cancer Classification using Convolutional Neural Network (CNN)

This module implements a deep learning approach for lung cancer stage classification
using a Convolutional Neural Network with TensorFlow/Keras. The model processes
MFCC (Mel-Frequency Cepstral Coefficients) features extracted from audio data and
employs rigorous cross-validation techniques to prevent data leakage and ensure
robust performance evaluation.

Key Features:
- Patient-grouped cross-validation to prevent data leakage
- Data integrity checks for duplicate samples and label consistency
- MFCC feature preprocessing with padding and normalization
- CNN architecture optimized for 2D spectral feature analysis
- Comprehensive performance evaluation with ROC-AUC metrics
- Reproducible results with fixed random seeds
- Performance comparison baseline for MLP architecture selection

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- tensorflow: Deep learning framework
- scikit-learn: Machine learning utilities
- imbalanced-learn: Handling imbalanced datasets
- See requirements.txt for all dependencies required

Usage:
    python lung_cancer_cnn.py

The script expects a NumPy file at 'data/binary_mfccs.npy' containing
lung cancer MFCC data with features, patient IDs, and cancer stage labels.

Author: Richard Liu
"""

"""
NOTE to self

This CNN model serves as a performance comparison baseline for architectural 
decisions in the project. After evaluation, the MLP architecture was selected
as the primary model due to better performance characteristics and computational
efficiency for the given dataset.
"""

import pandas as pd
import numpy as np
import tensorflow as tf 
import random
import os
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from imblearn.combine import SMOTEENN

# Seed setting
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

"""
The DataHandling Class handles and transforms the CNN performance data into training data for this model. 

The instantiated variables for training and validation are transformed through the class functions using LabelEncoder.
Data Splits are handled in validation_split() with patient-grouped cross-validation.

This class encapsulates all data preprocessing steps including:
- MFCC data loading and integrity checks for duplicate samples and label consistency
- MFCC feature preprocessing with padding and normalization
- Patient-grouped cross-validation splits
- Label encoding for categorical targets
- Validation set creation with group-aware splitting

Attributes:
    encoder (LabelEncoder): Converts categorical labels to numerical format
    smote (SMOTEENN): Combined over/under-sampling technique for class balance (unused in current implementation)
    data (str): Path to the input NumPy file containing MFCC data
    
    Storage for cross-validation results:
    - history: Training histories from each fold
    - auc: ROC-AUC scores from each fold
    - report: Classification reports from each fold
    - c_matrix: Confusion matrices from each fold
    - details: Fold-specific statistics and metadata
    
    Data processing attributes:
    - X, y, groups: Main feature matrix, labels, and patient grouping
    - Various train/test/validation splits and encodings
    - Patient ID tracking for group-aware splitting
"""

class DataHandling:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.smote = SMOTEENN()
        self.data = r"data/binary_mfccs.npy"

        self.history = []
        self.auc = []
        self.report = []
        self.c_matrix = []
        self.details = []

        self.X = None
        self.y = None
        self.groups = None
        self.y_test_fold = None 
        self.X_test_fold = None 
        self.y_encoded = None 
        self.X_train_fold = None 
        self.X_test_fold = None 
        self.y_train_fold_int = None
        self.y_test_fold_int = None 
        self.train_patients = None 
        self.val_patients = None
        self.test_patients = None 

        self.groups_train = None
        self.groups_test = None 

        self.num_classes = None
        self.patient_labels = None 
        self.inconsistent_patients = None 

        self.rows = None 
        self.train_idx = None
        self.test_idx = None 

    def load_data(self):
        """
        Load and perform comprehensive data integrity checks on the lung cancer MFCC dataset.
        
        This method loads the MFCC feature dataset from a NumPy file and validates data quality
        by checking for patient-label consistency across all samples. Each sample in the dataset
        contains MFCC features, cancer stage labels, and patient IDs.
        
        Returns:
            bool: True if data passes all integrity checks (consistent patient labels), 
                  False otherwise.
        
        Checks Performed:
            1. Patient-Label Consistency: Ensures each patient has consistent cancer stage labels
               across all their audio samples
        
        Side Effects:
            - Prints warnings for any integrity issues found
            - Sets class attributes: rows, patient_labels, inconsistent_patients
        
        Data Structure Expected:
            NumPy array containing tuples of (mfcc_features, cancer_stage, patient_id)
        
        Example:
            >>> handler = DataHandling()
            >>> is_clean = handler.load_data()
            >>> if is_clean:
            ...     print("Data passed all integrity checks")
        """
        data_array = np.load(self.data, allow_pickle=True)
        self.rows = [{'patient_id': pid, 'cancer_stage': label} for _, label, pid in data_array]
        data = pd.DataFrame(self.rows)
        self.patient_labels = data.groupby('patient_id')['cancer_stage'].nunique()
        self.inconsistent_patients = self.patient_labels[self.patient_labels > 1]

        return len(self.inconsistent_patients) == 0
    
    def transform(self, data_array):
        """
        Transform raw MFCC data into CNN-ready format with preprocessing and normalization.
        
        This method processes the raw MFCC features to create standardized input suitable
        for convolutional neural network training. It handles variable-length sequences
        through padding and applies normalization for stable training.
        
        Args:
            data_array (np.ndarray): Raw data array containing (mfcc, label, patient_id) tuples
        
        Preprocessing Steps:
            1. MFCC Padding/Truncation:
               - Standardizes all MFCC sequences to exactly 60 time frames
               - Truncates sequences longer than 60 frames
               - Zero-pads sequences shorter than 60 frames
            
            2. Feature Normalization:
               - Applies global min-max normalization across all features
               - Scales values to prevent gradient explosion/vanishing
               - Maintains relative feature relationships
            
            3. CNN Input Formatting:
               - Adds channel dimension for Conv2D layers (grayscale: 1 channel)
               - Final shape: (samples, 60, 13, 1) for standard MFCC features
        
        Side Effects:
            - Sets class attributes: X (features), y (labels), groups (patient IDs)
            - X is formatted as 4D tensor ready for CNN input
            - y and groups remain as 1D arrays for compatibility
        
        Technical Details:
            - MFCC features typically have shape (time_frames, 13_coefficients)
            - Padding ensures consistent input dimensions across variable-length audio
            - Normalization prevents numerical instability in deep networks
        
        Example:
            >>> handler = DataHandling()
            >>> data = np.load('data/binary_mfccs.npy', allow_pickle=True)
            >>> handler.transform(data)
            >>> print(f"Processed features shape: {handler.X.shape}")
        """
        X, y, groups = [], [], []

        for mfcc, label, patient_id in data_array:
            mfcc = mfcc[:60] if mfcc.shape[0] >= 60 else np.pad(mfcc, ((0, 60 - mfcc.shape[0]), (0, 0)))
            X.append(mfcc)
            y.append(label)
            groups.append(patient_id)     

        self.X = np.array(X)[..., np.newaxis]  
        self.X = self.X / np.max(np.abs(self.X))

        self.y = np.array(y)
        self.groups = np.array(groups)

    def data_split(self, encoder, train_idx, test_idx):

        self.train_idx = train_idx
        self.test_idx = test_idx

        self.y_encoded = encoder.fit_transform(self.y)
        self.X_train_fold = self.X[self.train_idx]
        self.X_test_fold = self.X[self.test_idx]
        self.y_train_fold_int = self.y_encoded[self.train_idx]
        self.y_test_fold_int = self.y_encoded[self.test_idx]
        self.groups_train = self.groups[self.train_idx]
        self.groups_test = self.groups[self.test_idx]

        self.y_train_fold = to_categorical(self.y_train_fold_int, num_classes=self.num_classes)
        self.y_test_fold = to_categorical(self.y_test_fold_int, num_classes=self.num_classes)

    def validation_split(self):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        val_train_idx, val_idx = next(gss.split(self.X_train_fold, self.y_train_fold_int, groups=self.groups_train))

        self.X_train_final = self.X_train_fold[val_train_idx]
        self.X_val_final = self.X_train_fold[val_idx]
        self.y_train_final = self.y_train_fold[val_train_idx]
        self.y_val_final = self.y_train_fold[val_idx]

        self.train_patients = set(self.groups_train[val_train_idx])
        self.val_patients = set(self.groups_train[val_idx])
        self.test_patients = set(self.groups_test)

    def get_data(self):
        return self.y_train_fold, self.y_test_fold, self.train_patients, self.val_patients, self.test_patients

class LungCancerCNN:
    """
    Convolutional Neural Network for lung cancer stage classification from MFCC features.
    
    This class implements a CNN architecture specifically designed for processing
    2D spectral features (MFCC) extracted from audio data. The model uses convolutional
    layers to capture local patterns in the frequency-time domain, followed by dense
    layers for final classification.
    
    Attributes:
        X_train_final (np.ndarray): Training feature data used for input shape determination
        num_classes (int): Number of target classes for classification
        model (Sequential): Compiled Keras CNN model ready for training
        target_names (list): Class names for evaluation (set during training)
    
    Architecture Features:
        - Two convolutional blocks with increasing filter complexity (32 -> 64)
        - MaxPooling for spatial down-sampling and feature compression
        - Dropout layers for regularization and overfitting prevention
        - Dense layers for final feature combination and classification
        - Softmax activation for multi-class probability output
    
    Model Comparison Notes:
        This CNN serves as a performance baseline for architectural comparison.
        While CNNs excel at capturing spatial patterns in 2D data, the project
        ultimately selected MLP architecture for better performance on this
        specific lung cancer classification task.
    """
    def __init__(self, X_train_final, num_classes):
        self.X_train_final = X_train_final
        self.num_classes = num_classes 
        self.model = self._buildmodel()
        self.target_names = None

    def _buildmodel(self):
        """
        Build and compile the Convolutional Neural Network architecture.
        
        Creates a CNN specifically designed for MFCC feature analysis with:
        - Two convolutional blocks for hierarchical feature extraction
        - Progressive filter increase (32 -> 64) to capture increasingly complex patterns
        - MaxPooling for spatial dimensionality reduction
        - Dropout regularization to prevent overfitting
        - Dense layers for final classification
        
        Model Architecture:
            - Conv2D Layer 1: 32 filters, 3x3 kernel, ReLU activation
            - MaxPooling2D: 2x2 pooling for spatial reduction
            - Dropout: 30% for regularization
            - Conv2D Layer 2: 64 filters, 3x3 kernel, ReLU activation  
            - MaxPooling2D: 2x2 pooling for further reduction
            - Dropout: 30% for regularization
            - Flatten: Convert 2D feature maps to 1D
            - Dense: 128 neurons, ReLU activation
            - Dropout: 30% for final regularization
            - Output: num_classes neurons with softmax activation
        
        Input Shape:
            - (60, 13, 1): 60 time frames, 13 MFCC coefficients, 1 channel
            - Designed for standard MFCC spectral analysis
        
        Regularization Techniques:
            - Dropout (30%): Prevents overfitting by randomly dropping neurons
            - MaxPooling: Reduces spatial dimensions and computational complexity
        
        Compilation Settings:
            - Optimizer: Adam (adaptive learning rate)
            - Loss: Categorical crossentropy for multi-class classification
            - Metrics: Accuracy for performance monitoring
        
        Returns:
            Sequential: Compiled Keras CNN model ready for training
        
        Architecture Rationale:
            - Two conv blocks capture both local and global spectral patterns
            - Filter progression (32->64) allows hierarchical feature learning
            - Moderate depth prevents overfitting on medical datasets
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(60, 13, 1)),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
           
            Dropout(0.3),
            Flatten(),

            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        
        return model
    
    def train(self, X_train_final, y_train_final, X_val_final, y_val_final, epochs=50, batch_size=16):
        """
        Train the CNN model with advanced regularization callbacks.
        
        This method implements comprehensive training with multiple regularization
        techniques specifically tuned for CNN architectures to prevent overfitting
        and ensure optimal model performance on spectral feature data.
        
        Args:
            X_train_final (np.ndarray): Training MFCC feature tensor (N, 60, 13, 1)
            y_train_final (np.ndarray): Training labels (one-hot encoded)
            X_val_final (np.ndarray): Validation MFCC feature tensor
            y_val_final (np.ndarray): Validation labels (one-hot encoded)
            epochs (int, optional): Maximum number of training epochs. Defaults to 50.
            batch_size (int, optional): Training batch size. Defaults to 16.
        
        Returns:
            History: Keras training history object containing loss and metrics
                    for each epoch across training and validation sets.
        
        Regularization Techniques:
            1. Early Stopping:
                - Monitors validation loss to prevent overfitting
                - Stops training when validation loss stops improving
                - Patience of 5 epochs before stopping
                - Restores best weights from optimal epoch
            
            2. Learning Rate Reduction:
                - Monitors validation loss for plateaus
                - Reduces learning rate by factor of 0.1 when validation loss plateaus
                - Patience of 3 epochs before reduction
                - Helps fine-tune CNN features in later training stages
        
        Training Configuration:
            - Validation monitoring for both callbacks
            - Verbose output for learning rate changes
            - Batch-wise training optimized for CNN memory usage
        
        CNN-Specific Considerations:
            - Smaller batch size (16) helps with memory constraints from 2D convolutions
            - Early stopping crucial for CNNs which can overfit quickly on small datasets
            - Learning rate scheduling helps fine-tune learned convolutional filters
        
        Example:
            >>> model = LungCancerCNN(X_train.shape[1:], num_classes=2)
            >>> history = model.train(X_train, y_train, X_val, y_val)
            >>> print(f"Training completed in {len(history.history['loss'])} epochs")
        """
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        lr_schedule = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.1, 
            patience=3, 
            verbose=1
        )

        history = self.model.fit(
            X_train_final, y_train_final,
            validation_data=[X_val_final, y_val_final],
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stopping, lr_schedule], verbose=1
        )

        return history 

    def evaluate(self, y_test, preds, encoder):
        """
        Evaluate the trained CNN model on test data with comprehensive metrics.
        
        This method computes multiple performance metrics to provide a thorough
        assessment of CNN model performance on spectral feature classification,
        including precision, recall, F1-score, confusion matrix, and ROC-AUC scores.
        
        Args:
            y_test (np.ndarray): Test labels (integer encoded)
            preds (np.ndarray): Model predictions (unused - recalculated internally)
            encoder (LabelEncoder): Fitted label encoder for class name mapping
        
        Returns:
            tuple: A 3-tuple containing:
                - report (dict): Comprehensive classification report with precision,
                  recall, F1-score, and support for each class
                - c_matrix (np.ndarray): Confusion matrix showing prediction accuracy
                - auc (float): ROC-AUC score using one-vs-rest multi-class approach
        
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
                - One-vs-rest approach for multi-class problems
                - Measures CNN's ability to distinguish between spectral patterns
                - Values closer to 1.0 indicate better feature discrimination
        
        Process:
            1. Generate fresh predictions using trained CNN
            2. Convert softmax probabilities to class predictions
            3. Compute classification metrics using original class names
            4. Calculate ROC-AUC with multi-class handling
        
        CNN-Specific Notes:
            - Predictions generated from learned convolutional filters
            - Performance reflects CNN's ability to capture spectral patterns
            - Comparison baseline for architectural decision-making
        
        Example:
            >>> report, matrix, auc = model.evaluate(y_test, None, encoder)
            >>> print(f"CNN Test Accuracy: {report['accuracy']:.4f}")
            >>> print(f"CNN ROC-AUC Score: {auc:.4f}")
        """
        preds = np.argmax(self.model.predict(self.X_test_fold), axis=1)

        report = classification_report(
            y_test, preds, 
            target_names=[str(cls) for cls in encoder.classes_],
            output_dict=True 
        )

        c_matrix = confusion_matrix(y_test, preds)

        auc = roc_auc_score(
            to_categorical(y_test, num_classes=self.num_classes),
            to_categorical(preds, num_classes=self.num_classes),
            multi_class='ovr'
        )

        return report, c_matrix, auc 

def pipeline(handler):
    """
    Execute the complete CNN machine learning pipeline with patient-grouped cross-validation.
    
    This function implements a comprehensive ML pipeline that trains and evaluates
    a Convolutional Neural Network for lung cancer stage classification using rigorous
    cross-validation techniques to prevent data leakage and ensure robust performance.
    The pipeline serves as a performance comparison baseline for architectural decisions.
    
    Args:
        handler (DataHandling): Initialized DataHandling object containing loaded
                               MFCC dataset and preprocessing configurations.
    
    Pipeline Steps (per fold):
        1. Patient-grouped data splitting to prevent leakage
        2. MFCC feature preprocessing and normalization
        3. Label encoding and categorical conversion for neural network
        4. Validation set creation with group-aware splitting
        5. CNN model initialization and training with callbacks
        6. Comprehensive evaluation with multiple metrics
        7. Results storage and performance logging
    
    Cross-Validation Strategy:
        - 5-fold GroupKFold ensures patients don't appear in both train/test
        - Each fold trains a fresh CNN to avoid bias
        - Performance metrics aggregated across all folds
        - Real-time progress reporting during training
    
    Results Storage:
        Updates handler attributes with:
        - reports: Classification reports with precision/recall/F1
        - conf_matrices: Confusion matrices showing prediction accuracy
        - roc_aucs: ROC-AUC scores for model discrimination ability
        - history: Training histories for loss/accuracy curves
        - details: Fold metadata including sample counts and epochs
    
    Performance Monitoring:
        - Real-time accuracy statistics during training
        - Class-wise F1-score analysis
        - Confusion matrix display for each fold
        - Mean and standard deviation reporting
    
    Architectural Comparison Notes:
        This CNN pipeline generates performance metrics used to compare
        against MLP architecture. Results inform the final architectural
        decision for the project's primary model selection.
    
    Example:
        >>> handler = DataHandling()
        >>> handler.load_data()
        >>> pipeline(handler)
        # Executes complete 4-fold CNN cross-validation pipeline
    """
    gkf = GroupKFold(n_splits=4)

    for fold, (train_idx, test_idx) in (gkf.split(handler.X, handler.y, handler.groups)):
        handler.data_split(train_idx, test_idx)
        handler.transform()
        handler.validation_split()

        model = LungCancerCNN(
        )

        history = model.train(
            handler.X_train, handler.y_train, handler.X_val, handler.y_val
        )

        report, c_matrix, auc = model.evaluate(
            handler.X_test_scaled, handler.y_test_encoded, handler.encoder
        )

        handler.reports.append(report)
        handler.conf_matrices.append(c_matrix)
        handler.roc_aucs.append(auc)
        handler.history.append(history.history)
        handler.details.append({
            "fold": fold + 1,
            "train_samples": len(handler.X_train_fold),
            "test_samples": len(handler.X_test_fold),
            "accuracy": report['accuracy'],
            "epochs_trained": len(history.history['loss']),
        })

        # Logging 
        accuracies = [report['accuracy'] for report in handler.reports]
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print(f"\nOverall Performance:")
        print(f"Mean Accuracy: {avg_accuracy:.4f} +- {std_accuracy:.4f}")
        print(f"Min Accuracy:  {min(accuracies):.4f}")
        print(f"Max Accuracy:  {max(accuracies):.4f}")
        
        class_0_f1 = [report['0']['f1-score'] for report in handler.reports]
        class_1_f1 = [report['1']['f1-score'] for report in handler.reports]
        
        print(f"\nClass-wise F1-scores:")
        print(f"Class 0: {np.mean(class_0_f1):.4f} +- {np.std(class_0_f1):.4f}")
        print(f"Class 1: {np.mean(class_1_f1):.4f} +- {np.std(class_1_f1):.4f}")

        print(c_matrix)

handler = DataHandling()
# Requires a positional argument (data_array) but its not defined 
handler.load_data()
pipeline(handler)
