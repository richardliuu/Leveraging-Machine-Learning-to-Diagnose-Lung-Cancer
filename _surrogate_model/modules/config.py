"""
Model Configuration File
"""

import os 
import random 
import numpy as np 

# Set the seed for the program 
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Set paths 
DATA_PATH = "data/rf_predictions.csv"
RESULTS_DIR = "results"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# Model hyperparameters
MODEL_PARAMS = {
    "max_depth": 4,
    "min_samples_leaf": 20,
    "random_state": SEED
}

# Probability clustering thresholds
# For binary clustering around 0.5, set both to 0.5 so the
# "mixed" region is empty and only two groups remain.
CLUSTER_THRESHOLD = 0.5
CONFIDENT_THRESHOLD = CLUSTER_THRESHOLD
UNCONFIDENT_THRESHOLD = CLUSTER_THRESHOLD

# Cross Validation
N_SPLITS = 4 

# UMAP Projection Parameters
UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "random_state": SEED,
    "metric": "euclidean"
}

# Exclude columns from training data 
EXCLUDE_COLS = [
    'true_label', 
    'predicted_label', 
    'prob_class_0', 
    'prob_class_1',
    'patient_id', 
    'chunk', 
    'fold', 
    'test_idx'
]
