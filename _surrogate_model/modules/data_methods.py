import pandas as pd
import numpy as np
from .config import EXCLUDE_COLS, CONFIDENT_THRESHOLD, UNCONFIDENT_THRESHOLD

class LoadData:
    """
    Class designed to handle the loading and preprocessing of data from a CSV file for use in machine learning workflows.
    
    Attributes:
        data (str): Path to the CSV file containing the dataset.
        df (pd.DataFrame): DataFrame to store the loaded data.
        X_original (np.ndarray): Numpy array of feature values after excluding specified columns.
        rf_probs (np.ndarray): Array of predicted probabilities for class 1 from the random forest model.
        rf_hard_preds (np.ndarray): Array of hard predictions (predicted labels) from the random forest model.
        y_original (np.ndarray): Array of true labels.
        exclude_cols (list): List of columns to exclude from the feature set to prevent data leakage.
        feature_cols (list): List of columns used as features for modeling.
    Methods:
        __init__():
            Initializes the LoadData object and its attributes.
        load_data():
            Loads the dataset from the specified CSV file, excludes columns that could cause data leakage,
            extracts features, predicted probabilities, hard predictions, and true labels.
            Prints dataset statistics and returns the processed arrays and feature column names.
            Returns:
                tuple: (X_original, rf_probs, rf_hard_preds, y_original, feature_cols)

    """
    
    def __init__(self, data_path="data/rf_predictions.csv"):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_probs_train = None
        self.rf_probs_test = None
        self.rf_hard_preds_train = None
        self.rf_hard_preds_test = None
        self.y_original_train = None
        self.y_original_test = None
        self.feature_cols = None
        self.exclude_cols = EXCLUDE_COLS

    def load_data(self, train_idx, test_idx):
        """
        Loads data from the specified CSV file, extracts features and target columns, 
        and splits the data into training and testing sets based on provided indices.
        Parameters:
            train_idx (array-like): Indices for selecting training samples.
            test_idx (array-like): Indices for selecting testing samples.
        Returns:
            self: The instance with loaded and split data as attributes.
        Side Effects:
            - Reads the dataset from self.data_path.
            - Sets self.df to the loaded DataFrame.
            - Sets self.feature_cols to the list of feature column names.
            - Sets self.X_train, self.X_test: Feature arrays for train/test.
            - Sets self.rf_probs_train, self.rf_probs_test: Probability predictions for train/test.
            - Sets self.rf_hard_preds_train, self.rf_hard_preds_test: Hard predictions for train/test.
            - Sets self.y_original_train, self.y_original_test: True labels for train/test.
            - Prints summary statistics about the split and features.
        """

        self.df = pd.read_csv(self.data_path)
        
        # Define feature columns
        self.feature_cols = [col for col in self.df.columns if col not in self.exclude_cols]
        
        # Extract features and targets
        X = self.df[self.feature_cols].values
        rf_probs = self.df['prob_class_1'].values
        rf_hard_preds = self.df['predicted_label'].values
        y_original = self.df['true_label'].values
        
        # Split data
        self.X_train, self.X_test = X[train_idx], X[test_idx]
        self.rf_probs_train, self.rf_probs_test = rf_probs[train_idx], rf_probs[test_idx]
        self.rf_hard_preds_train, self.rf_hard_preds_test = rf_hard_preds[train_idx], rf_hard_preds[test_idx]
        self.y_original_train, self.y_original_test = y_original[train_idx], y_original[test_idx]
        
        print(f"Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Target range: [{self.rf_probs_train.min():.3f}, {self.rf_probs_train.max():.3f}]")
        
        return self

class ClusterData:
    """
    ClusterData partitions data samples into clusters based on confidence thresholds.
    This class provides methods to assign samples to "confident", "mixed", or "unconfident" clusters
    according to predicted probability scores (e.g., classifier confidence). It stores cluster assignments
    and provides utilities for extracting cluster-specific data subsets.
    Attributes
    confident_threshold : float
        Probability threshold above which samples are considered "confident".
    unconfident_threshold : float
        Probability threshold below which samples are considered "unconfident".
    clusters : dict
        Dictionary holding the most recent cluster assignments and their associated data.
    Methods
    cluster(X, rf_probs, y_original=None)
        Partition samples into "confident", "mixed", and "unconfident" clusters based on probability thresholds.
        Returns a dictionary of cluster data and prints cluster sizes.
    _create_cluster_dict(X, rf_probs, y_original, mask)
        Static method to extract and package data for a cluster using a boolean mask.
    """
        
    def __init__(self):
        """ Instantiate variables """
        super().__init__(LoadData)
        self.confident_threshold = CONFIDENT_THRESHOLD
        self.unconfident_threshold = UNCONFIDENT_THRESHOLD
        self.clusters = {}

    def cluster(self, X, rf_probs, y_original = None):
        """
        Partition samples into three clusters ("confident", "mixed", "unconfident") based on predicted probabilities.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix or collection of samples to be clustered. The function will slice subsets of X
            according to the masks computed from rf_probs.
        rf_probs : array-like of shape (n_samples,)
            Predicted probability scores (for a given class or confidence measure) used to assign each sample
            to a cluster. Values are compared against self.confident_threshold and self.unconfident_threshold.
        y_original : array-like of shape (n_samples,), optional
            Optional ground-truth labels corresponding to X. If provided, these labels are passed into the
            created cluster dictionaries for inspection or evaluation.
        Returns
        -------
        dict
            A dictionary with three keys: 'confident', 'mixed', and 'unconfident'. Each value is the
            dictionary produced by self._create_cluster_dict(...) for that cluster and typically contains
            the subset of X, corresponding rf_probs, optional y values, and any additional metadata
            produced by _create_cluster_dict.
        Side effects
        -----------
        - Sets self.clusters to the returned dictionary.
        - Prints a summary of cluster sizes to stdout.
        Notes
        -----
        - Membership rules:
            - "confident": rf_probs >= self.confident_threshold
            - "unconfident": rf_probs <= self.unconfident_threshold
            - "mixed": samples with rf_probs strictly between the two thresholds
        - The function assumes that X, rf_probs, and y_original (if provided) are aligned and have the same
          length. If lengths differ, behavior is undefined and a ValueError should be raised by callers or
          by additional validation (not implemented here).
        Raises
        ------
        ValueError
            If input arrays have mismatched lengths (recommended to validate before calling).
        """

        confident_mask = rf_probs >= self.confident_threshold
        unconfident_mask = rf_probs <= self.unconfident_threshold
        mixed_mask = ~confident_mask & ~unconfident_mask

        self.clusters = {
            'confident': self._create_cluster_dict(
                X, rf_probs, y_original, confident_mask
            ),
            'mixed': self._create_cluster_dict(
                X, rf_probs, y_original, mixed_mask
            ),
            'unconfident': self._create_cluster_dict(
                X, rf_probs, y_original, unconfident_mask
            )
        }

        print(f"\nCluster sizes:")
        for name, data in self.clusters.items():
            print(f" {name}: {len(data['X'])} samples")

        return self.clusters
    
    @staticmethod
    def _create_cluster_dict(X, rf_probs, y_original, mask):
        """
        Create a dictionary representing a cluster by selecting rows from input arrays
        using a boolean mask.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix or array whose rows correspond to samples. The mask is
            applied along the first axis to select the cluster's samples.
        rf_probs : array-like, shape (n_samples, ...) 
            Predicted probabilities (e.g., from a random forest) aligned with X. The
            mask is applied to select the corresponding probability rows/entries.
        y_original : array-like or None, shape (n_samples,)
            Optional original labels aligned with X and rf_probs. If provided, the
            mask is applied and the selected labels are returned; if None, the
            resulting dictionary contains None for 'y_original'.
        mask : array-like of bool, shape (n_samples,)
            Boolean mask indicating which rows belong to the cluster. True selects a
            row. Must be broadcastable to the first dimension of X and rf_probs.
        Returns
        -------
        dict
            A dictionary with the following keys:
            - 'X': array-like
                Rows of X selected by mask (X[mask]).
            - 'y': array-like
                Rows/entries of rf_probs selected by mask (rf_probs[mask]).
            - 'y_original': array-like or None
                Selected original labels if y_original is provided, otherwise None.
            - 'indices': numpy.ndarray of int
                Integer indices of True entries in mask (np.where(mask)[0]).
        Raises
        ------
        ValueError
            If the input arrays do not share a compatible first-dimension length such
            that boolean indexing with mask is invalid.
        Notes
        -----
        Inputs should support NumPy-style boolean indexing (e.g., numpy arrays,
        pandas DataFrame/Series). The function does not copy data beyond the slices
        returned by boolean indexing.
        """
        
        return {
            'X': X[mask],
            'y': rf_probs[mask],
            'y_original': y_original[mask] if y_original is not None else None,
            'indices': np.where(mask)[0]
        }