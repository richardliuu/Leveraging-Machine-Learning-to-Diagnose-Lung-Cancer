"""
Author: Richard Liu
Description:

This program contains a surrogate model for our random forest model to provide insight
into model behaviours. We analyze specific clusters of the model 

pull from README.md

"""

"""
Pipeline:

Load data and analyze two specific cluster types from the models behaviour:

- Confident predictions (60% - 100%)
- Unconfident predictions (40% - 0%)
- (Maybe uncertain predictions at 50%)

Split the clusters into datasets for training through a surrogate model 

NOTE Challenges: Running model on each dataset respectively 

Keep the surrogate model consistent which is either going to be:
- Decision tree
- Lasso
- Elastic Net
- Polynomial Regressor

Then, generate UMAP projections to visualize the data 



"""

# Import dependencies 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import umap
import joblib
import warnings
warnings.filterwarnings('ignore')

class LoadData:
    """
    This class handles the data loading for the program. This class
    requires CSV file at self.data. 
    """
    def __init__(self):
        """ Instantiate variables """
        self.data = "data/rf_predictions.csv"
        self.df = None
        self.X_original = None
        self.rf_probs = None
        self.rf_hard_preds = None
        self.y_original = None
        self.exclude_cols = None
        self.feature_cols = None

    def load_data(self):
        """
        This function loads data for the program and 
        """
        self.df = pd.read_csv(self.data)

        # Exclude columns to remove data leakage and crucial/not important information
        exclude_cols = ['true_label', 'predicted_label', 'prob_class_0', 'prob_class_1', 
                    'patient_id', 'chunk', 'fold', 'test_idx']
        
        # Include all columns that were excluded 
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        # Extract specific information for training 
        self.X_original = self.df[feature_cols].values  # Original features 
        self.rf_probabilities = self.df['prob_class_1'].values  # TARGET: Class 1 probabilities
        self.rf_hard_predictions = self.df['predicted_label'].values  # For validation 
        self.y_original = self.df['true_label'].values  # For validation 
        
        # Print out information on the dataset 
        print(f"Data shape: {self.X_original.shape}")
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(self.X_original)}")
        print(f"Target range: [{self.rf_probs.min():.3f}, {self.rf_probs.max():.3f}]")
        print(f"Target mean: {self.rf_probs.mean():.3f} +- {self.rf_probs.std():.3f}")
        
        return self.X_original, self.rf_probs, self.rf_hard_preds, self.y_original, self.feature_cols

class ClusterData:
    """
    
    """
    def __init__(self):
        """ Instantiate variables """

        # Should probably make these variables dataframes so that they can be csv datasets 
        self.confident = []
        self.mixed = []
        self.unconfident = [] 

    def cluster(self):
        """
        
        """

        
class SurrogateModel:
    """
    
    """ 
    def __init__(self):
        """ Instantiate variables """
        pass 

    def train(self):
        """
        
        """
        pass

class TrainModel:
    """
    
    """
    def __init__(self):
        super().__init__(SurrogateModel)


class EvaluateTraining:
    """
    
    """
    def __init__(self):
        """ Instantiate variables """
        super().__init__(TrainModel)
        pass

class UMAPProjection:
    """
    
    """
    def __init__(self):
        pass


class InterpretModel:
    """
    
    """
    def __init__(self):
        """ Instantiate variables """
        pass
    

# Call classes and run pipeline
if __name__ == "__main__":
    pass