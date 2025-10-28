import numpy as np
import os 
import random
from sklearn.tree import RandomForestClassifier

# Seed Setting
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

"""
NOTE Insert the hyper params from the config file 
"""

class RandomForestModel:
    """
    Random Forest classifier for lung cancer stage classification.
    
    This class implements a Random Forest model with configurable hyperparameters
    optimized for medical classification tasks. The model includes balanced class
    weights to handle potential class imbalance and provides feature importance analysis.
    
    Attributes:
        model (RandomForestClassifier): The scikit-learn Random Forest model
        feature_cols (list): Names of the feature columns for interpretability
    
    Model Configuration (with aggressive regularization):
        - criterion: "log_loss" for probabilistic splits
        - n_estimators: 200 trees for stable predictions
        - max_depth: 5 to prevent overfitting (reduced from 12)
        - max_features: 0.6 to use only 60% of features for each split
        - min_samples_split: 25 to ensure robust splits (increased from 12)
        - min_samples_leaf: 10 for stable leaf predictions (increased from 3)
        - class_weight: 'balanced' to handle class imbalance
        - bootstrap: True for bootstrapping samples
        - oob_score: True for out-of-bag scoring validation
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
            max_leaf_nodes=None,
            max_features=0.3,
            min_samples_split=25,
            min_samples_leaf=10,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True,
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
