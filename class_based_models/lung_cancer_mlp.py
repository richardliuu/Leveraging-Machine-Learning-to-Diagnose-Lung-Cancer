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
- SHAP feature values for model analysis

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- tensorflow: Deep learning framework
- matplotlib: Plotting and visualization
- scikit-learn: Machine learning utilities
- imbalanced-learn: Handling imbalanced datasets
- shap: Model interpretability and feature importance analysis
- See requirements.txt for all dependencies required (should clean it up)

Usage:
    python lung_cancer_mlp.py

The script expects a CSV file at 'data/binary_features_log.csv' containing
lung cancer data with features, patient IDs, and cancer stage labels.

Author: Richard Liu
"""

"""
NOTE to self

Make SHAP analysis another class to make the fidelity check easier to run
"""

import shap
import pandas as pd
import numpy as np
import tensorflow as tf 
import random
import os
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input 
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
NOTE to self
1. Tune hyper parameters

2. Add params to SMOTE if needed  

3. Clean up code

4. Should probably use Macro metrics to combat class imbalances 
"""

"""
The DataHandling Class handles and transforms the MLP performance data into training data for this surrogate model. 

The instantiated variables for training and validation are transformed through the class functions using LabelEncoder and StandardScaler.
Data Splits are handled in validation_split().

This class encapsulates all data preprocessing steps including:
- Data integrity checks for duplicate samples and label consistency
- Feature scaling and label encoding
- SMOTEENN resampling for class imbalance handling
- Patient-grouped cross-validation splits
- Validation set creation with stratification

Attributes:
    scaler (StandardScaler): Scales features to have zero mean and unit variance
    encoder (LabelEncoder): Converts categorical labels to numerical format
    smote (SMOTEENN): Combined over/under-sampling technique for class balance
    data (str): Path to the input CSV file
    
    Storage for cross-validation results:
    - reports: Classification reports from each fold
    - conf_matrices: Confusion matrices from each fold
    - details: Fold-specific statistics and metadata
    - history: Training histories from each fold
    - roc_aucs: ROC-AUC scores from each fold
    - predictions: Model predictions from each fold
"""

class DataHandling:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.smote = SMOTEENN(random_state=SEED)
        self.data = r"data/binary_features_log.csv"

        self.reports = []
        self.conf_matrices = []
        self.details = []
        self.history = []
        self.roc_aucs = []
        self.predictions = []

        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        self.num_classes = None
        self.feature_cols = None
        self.groups = None
        self.duplicates = None
        self.patient_labels = None
        self.inconsistent_patients = None
        self.patient_data = None

        self.train_patients = None
        self.test_patients = None
        
    def load_data(self):
        """
        Load and perform comprehensive data integrity checks on the lung cancer dataset.
        
        This method loads the dataset, extracts features and labels, and validates data quality
        by checking for duplicate samples and ensuring patient-label consistency across all samples.
        It also provides a summary of the patient-level class distribution.
        
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
            - Sets class attributes: X, y, feature_cols, groups, duplicates, patient_labels,
              inconsistent_patients, patient_data
        
        Data Structure Expected:
            CSV file with columns: 'segment', 'cancer_stage', 'patient_id', plus feature columns
        
        Example:
            >>> handler = DataHandling()
            >>> is_clean = handler.load_data()
            >>> if is_clean:
            ...     print("Data passed all integrity checks")
        """
        data = pd.read_csv("data/train_data.csv")
        self.X = data.drop(columns=['segment', 'cancer_stage', 'patient_id'])
        self.y = data['cancer_stage']

        # Allows SHAP to collect the feature names for analysis 
        self.feature_cols = self.X.columns.tolist()

        self.groups = data['patient_id']
        # Check for duplicate samples across all feature columns
        self.duplicates = data.duplicated(subset=self.feature_cols)
        # Verify each patient has consistent cancer stage labels across all samples
        self.patient_labels = data.groupby('patient_id')['cancer_stage'].nunique()
        self.inconsistent_patients = self.patient_labels[self.patient_labels > 1]  # Patients with >1 unique label
        
        if len(self.inconsistent_patients) > 0:
            print(f"WARNING: {len(self.inconsistent_patients)} patients have inconsistent labels!")
            print("Inconsistent patients:", self.inconsistent_patients.index.tolist()[:10])
            # Show details for first inconsistent patient
            if len(self.inconsistent_patients) > 0:
                first_patient = self.inconsistent_patients.index[0]
                patient_data = data[data['patient_id'] == first_patient][['patient_id', 'cancer_stage']]
                print(f"Example - Patient {first_patient}:")
                print(patient_data['cancer_stage'].value_counts())
        
        # Calculate patient-level class distribution using mode (most frequent label per patient)
        self.patient_data = data.groupby('patient_id').agg({
            'cancer_stage': lambda x: x.mode()[0]  # Take most frequent label for each patient
        }).reset_index()

        print(f"\nPatient-level class distribution:")
        patient_class_counts = self.patient_data['cancer_stage'].value_counts()
        print(patient_class_counts)
        
        return self.duplicates.sum() == 0 and len(self.inconsistent_patients) == 0

    def split(self, X, y, data, train_idx, test_idx):
        self.X_train = X.iloc[train_idx]
        self.y_train = y.iloc[train_idx]
        self.X_test = X.iloc[test_idx]
        self.y_test = y.iloc[test_idx]

        self.train_patients = set(data.iloc[train_idx]['patient_id'])
        self.test_patients = set(data.iloc[test_idx]['patient_id'])

    def transform(self):
        # Scaled data 
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Encoded
        self.y_train = self.encoder.fit_transform(self.y_train)
        self.y_test = self.encoder.transform(self.y_test)
        self.num_classes = len(self.encoder.classes_)
        self.X_train, self.y_train = self.smote.fit_resample(self.X_train, self.y_train)

    def put_to_categorical(self):
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
    
    # Takes in resampled X_train, categorical y_train and resampled y_train for stratify
    def validation_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
        self.X_train, 
        self.y_train, 
        test_size=0.1, 
        stratify=self.y_train, 
        random_state=SEED
        )

class LungCancerMLP:
    """
    Multi-Layer Perceptron for lung cancer stage classification.
    
    This class implements a deep neural network with progressive layer size reduction
    and comprehensive regularization techniques to prevent overfitting. The architecture
    is designed specifically for binary classification of cancer stages.
    
    Attributes:
        input_dim (int): Number of input features
        num_classes (int): Number of target classes
        model (Sequential): Compiled Keras model ready for training
    
    Architecture Features:
        - Progressive reduction in layer sizes (256 -> 128 -> 64)
        - ReLU activation with batch normalization and dropout
        - Sigmoid output activation for binary classification
        - Adam optimizer with categorical crossentropy loss
    """
    def __init__(self, input_dim, num_classes):
        """
        Initialize the MLP model with specified dimensions.
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of target classes for classification
        """
        self.input_dim = input_dim
        self.num_classes = num_classes 
        self.model = self._buildmodel()

    def _buildmodel(self):
        """
        Build and compile the Multi-Layer Perceptron architecture.
        
        Creates a deep neural network with:
        - Input layer matching the number of features
        - 3 hidden layers with progressive size reduction (256, 128, 64 neurons)
        - ReLU activation with batch normalization and dropout for regularization
        - Sigmoid output layer for classification
        
        Model Architecture:
            - Layer 1: 256 neurons + BatchNorm + 30% Dropout
            - Layer 2: 128 neurons + BatchNorm + 30% Dropout  
            - Layer 3: 64 neurons + BatchNorm + 30% Dropout
            - Output: num_classes neurons with sigmoid activation
        
        Regularization Techniques:
            - Batch Normalization: Normalizes inputs to each layer
            - Dropout (30%): Randomly drops neurons during training to prevent overfitting
        
        Compilation Settings:
            - Optimizer: Adam (adaptive learning rate)
            - Loss: Categorical crossentropy for multi-class classification
            - Metrics: Accuracy for performance monitoring
        
        Returns:
            Sequential: Compiled Keras model ready for training
        
        Note:
            Model density kept relatively low to avoid overcomplexity and overfitting
            on medical datasets which tend to be smaller.
        """
        model = Sequential([
            Input(shape=(self.input_dim,)),    
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(self.num_classes, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        """
        Train the MLP model with advanced regularization callbacks.
        
        This method implements comprehensive training with multiple regularization
        techniques to prevent overfitting and ensure optimal model performance.
        
        Args:
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training labels (one-hot encoded)
            X_val (np.ndarray): Validation feature matrix  
            y_val (np.ndarray): Validation labels (one-hot encoded)
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
                - Helps fine-tune model in later stages of training
        
        Training Configuration:
            - Validation monitoring for both callbacks
            - Verbose output for learning rate changes
            - Batch-wise training for memory efficiency
        
        Example:
            >>> model = LungCancerMLP(input_dim=100, num_classes=2)
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
            X_train, y_train, 
            validation_data = (X_val, y_val), 
            epochs = epochs, 
            batch_size = batch_size, 
            callbacks = [early_stopping, lr_schedule], 
            verbose = 1 
        )

        return history 
        
    def evaluate(self, X_test, y_test, encoder):
        """
        Evaluate the trained model on test data with comprehensive metrics.
        
        This method computes multiple performance metrics to provide a thorough
        assessment of model performance, including precision, recall, F1-score,
        confusion matrix, and ROC-AUC scores.
        
        Args:
            X_test (np.ndarray): Test feature matrix
            y_test (np.ndarray): Test labels (one-hot encoded)
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
                - Measures model's ability to distinguish between classes
                - Values closer to 1.0 indicate better performance
        
        Process:
            1. Convert one-hot encoded labels back to class indices
            2. Generate predictions using trained model
            3. Compute classification metrics using original class names
            4. Calculate ROC-AUC with multi-class handling
        
        Example:
            >>> report, matrix, auc = model.evaluate(X_test, y_test, encoder)
            >>> print(f"Test Accuracy: {report['accuracy']:.4f}")
            >>> print(f"ROC-AUC Score: {auc:.4f}")
        """
        y_test = np.argmax(y_test, axis=1)
        preds = np.argmax(self.model.predict(X_test), axis=1)

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
    
    def predict(self, X):
        y_pred_prob = self.model.predict(X, verbose=0)
        #y_pred = np.argmax(y_pred_prob, axis=1)

        return y_pred_prob 

def pipeline(handler):
    """
    Execute the complete machine learning pipeline with patient-grouped cross-validation.
    
    This function implements a comprehensive ML pipeline that trains and evaluates
    a Multi-Layer Perceptron for lung cancer stage classification using rigorous
    cross-validation techniques to prevent data leakage and ensure robust performance.
    
    Args:
        handler (DataHandling): Initialized DataHandling object containing loaded
                               dataset and preprocessing configurations.
    
    Pipeline Steps (per fold):
        1. Patient-grouped data splitting to prevent leakage
        2. Feature scaling and label encoding
        3. SMOTEENN resampling for class balance
        4. Categorical label conversion for neural network
        5. Validation set creation with stratification
        6. Model initialization and training with callbacks
        7. Comprehensive evaluation with multiple metrics
        8. Results storage and performance logging
    
    Cross-Validation Strategy:
        - 4-fold GroupKFold ensures patients don't appear in both train/test
        - Each fold trains a fresh model to avoid bias
        - Performance metrics aggregated across all folds
        - Real-time progress reporting during training
    
    Results Storage:
        Updates handler attributes with:
        - predictions: Model predictions for each fold
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
    
    Example:
        >>> handler = DataHandling()
        >>> handler.load_data()
        >>> pipeline(handler)
        # Executes complete 4-fold cross-validation pipeline
    """
    gkf = GroupKFold(n_splits=4)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        handler.split(handler.X, handler.y, pd.read_csv(handler.data), train_idx, test_idx)
        handler.transform()
        handler.put_to_categorical()
        handler.validation_split()

        model = LungCancerMLP(
            num_classes=handler.num_classes,
            input_dim=handler.X_train.shape[1],
        )

        history = model.train(
            handler.X_train, 
            handler.y_train, 
            handler.X_val, 
            handler.y_val
        )

        report, c_matrix, auc = model.evaluate(
            handler.X_test, 
            handler.y_test, 
            handler.encoder
        )

        y_pred_prob = model.predict(handler.X_test)

        #handler.predictions.append(y_pred)
        handler.reports.append(report)
        handler.conf_matrices.append(c_matrix)
        handler.roc_aucs.append(auc)
        handler.history.append(history.history)
        handler.details.append({
            "fold": fold + 1,
            "train_samples": len(handler.X_train),
            "test_samples": len(handler.X_test),
            "accuracy": report['accuracy'],
            "epochs_trained": len(history.history['loss']),
        })

        """
        Metric Logging:
        - F1 Score: Harmonic mean of precision and recall 
        - Accuracy: Correct predictions over all predictions 
        - Confusion Matrix: Provides a visual of the model's predictions per fold through true/false positive and true/false negative

        To see more detailed logging:
        - print(report) 
        - print(auc)

        The parameters and details are found in LungCancerMLP.evaluate()
        """
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
        
        """
        SHAP (SHapley Additive exPlanations) Feature Analysis
        
        This section implements model interpretability using SHAP values to understand
        which features contribute most to the model's predictions for lung cancer staging.
        SHAP is used in combination with models/decisiontree.py to make analyse the MLP's 
        black box behaviour.
        
        SHAP Analysis Components:
        1. Feature Contribution Analysis:
           - Computes SHAP values for each feature and prediction
           - Shows positive/negative impact of each feature on model output
           - Provides local explanations for individual predictions
           - Aggregates feature importance across all validation samples
        
        2. Background Data Selection:
           - Uses 20 random samples from validation set as background distribution
           - Background represents "typical" patient profile for comparison
           - SHAP values calculated relative to this baseline expectation
        
        3. Model Integration:
           - Uses KernelExplainer for model-agnostic interpretability
           - Works with any machine learning model (MLP in this case)
           - Handles multi-class output by focusing on class of interest
        
        4. Visualization Output:
           - Summary plot showing feature importance ranking
           - Color coding: red (high feature values), blue (low feature values)
           - X-axis shows SHAP value magnitude (impact on prediction)
           - Features ordered by importance (most impactful at top)
        
        Technical Implementation:
        - Converts scaled validation data back to interpretable DataFrame format
        - Maintains feature names from original dataset for meaningful labels
        - Handles 3D SHAP output arrays for multi-class classification
        - Focuses on class 1 (positive cancer stage) for medical relevance
        
        Clinical Significance:
        - Identifies which biomarkers/features drive cancer stage predictions
        - Helps validate model decisions against medical knowledge
        - Supports clinical decision-making with explainable AI
        - Enables feature selection for future model improvements
        
        Output Interpretation:
        - Features with high positive SHAP values increase probability of higher cancer stage
        - Features with high negative SHAP values decrease probability of higher cancer stage
        - Feature value colors show whether high/low values drive the prediction
        - Summary plot provides global view of feature importance patterns
        """
        
        """
        # Commented to prevent SHAP from running when unneccessary 

        # Convert to DataFrame
        X_val_df = pd.DataFrame(handler.X_val, columns=handler.feature_cols)
        X_explain = X_val_df.iloc[:]
        X_explain_np = X_explain.to_numpy()

        # Background for SHAP
        background = X_val_df.sample(n=20, random_state=42).to_numpy()
        explainer = shap.KernelExplainer(model.model.predict, background)
        shap_values = explainer.shap_values(X_explain_np)

        # If output is a 3D array: (samples, features, classes)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            print("SHAP values shape (3D):", shap_values.shape)
            class_index = 1  
            shap_vals_to_plot = shap_values[:, :, class_index]
        else:
            shap_vals_to_plot = shap_values[1] 

        # Confirm shape match
        assert shap_vals_to_plot.shape == X_explain_np.shape, \
            f"SHAP values shape {shap_vals_to_plot.shape} != input shape {X_explain_np.shape}"

        shap.summary_plot(shap_vals_to_plot, X_explain, feature_names=handler.feature_cols)
        """
        
handler = DataHandling()
handler.load_data()
pipeline(handler)
