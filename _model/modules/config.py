"""
Module: config

"""

import os 
import random 
import numpy as np 

# Set the seed for the program 
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Paths and directories
DATA_PATH = "data/train_data.csv"
RESULTS_DIR = "results"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics/")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots/")

# Model Hyperparameters
MODEL_PARAMS = {
    "criterion": "log_loss", 
    "n_estimators": 200,
    "max_depth": 5,
    "max_leaf_nodes": None,
    "max_features": 0.3,
    "min_samples_split": 25,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "bootstrap": True,
    "oob_score": True,
    "random_state": SEED,
    "n_jobs": -1
}

# Exclude Columns
EXCLUDE_COLS = [
    'chunk', 
    'cancer_stage',
    'patient_id', 
    'filename', 
    'rolloff', 
    'bandwidth', 
    "skew", 
    "zcr", 
    'rms'
    ]

# Cross Validation
N_SPLITS = 4

