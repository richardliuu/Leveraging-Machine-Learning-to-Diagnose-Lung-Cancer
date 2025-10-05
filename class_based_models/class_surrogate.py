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
    - May just need one train function but call it on 3 different datasets


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
import random
import os 
warnings.filterwarnings('ignore')

# Seed Setting for reproducable results

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

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
    This class module clusters data based on the model's 
    confidence 
    """
    def __init__(self):
        """ Instantiate variables """
        super().__init__(LoadData)
        self.confident = pd.DataFrame()
        self.mixed = pd.DataFrame()
        self.unconfident = pd.DataFrame()

    def cluster(self):
        """
        This function clusters the data and creates seperate datasets. 
        """
         
        if self.rf_probs > 0.7:
            self.confident.append(self.rf_probs)
        elif self.rf_probs > 0.4 and self.rf_probs < 0.7:
            self.mixed.append(self.rf_probs)
        elif self.rf_probs < 0.4:
            self.unconfident.append(self.rf_probs)
        else:
            pass
  
class SurrogateModel:
    """
    Base class for running our surrogate model.

    The SurrogateModel class provides an object to call the model for training
    when needed. 

    The surrogate serves as a transparent and simplified representation of 
    the random forest model used to predict lung cancer. This improves model
    explainability and interpretability through the DecisionTreeRegressor's simpler 
    rule-based structure.
    """ 
    def __init__(self):
        """ Instantiate variables """
        self.model = None
        self.model_params = None 
        self.max_depth = None

    def initialize_model(self, 
                         max_depth = 4,
                         min_samples_leaf = 20, 
                         random_state = SEED
                         ):
        """
        This function initializes the model and takes in
        the hyperparameters for the DecisionTreeRegressor.
        
        Parameters:
        ---------------
        
        max_depth = 4 
            - Reduce complexity of our surrogate which aims to 
            provide interpretability 

        min_samples_leaf = 20 
            - Ensure the model splits based on
            a minimum sample size of 20

        random_state = SEED 
            - Ensure reproducable results when training 

        Returns: self.model

            The initialized decision tree regressor as an object
        """

        self.model = DecisionTreeRegressor(
            max_depth = max_depth,
            min_samples_leaf = min_samples_leaf,
            random_state = random_state,
        )

        return self.model

class TrainModel(SurrogateModel):
    """
    This class inherits from the parent class SurrogateModel() for training the model
    on clustered datasets. 

    The TrainModel class extends the functionality of SurrogateModel 
    by providing methods to train the surrogate decision tree on input data.
    """

    def __init__(self):
        """ Instantiate variables """
        super().__init__()
        self.performance_metrics = {}
        self.predictions = None
        self.X = None
        self.y = None

    def train(self, X, y):
        """
        Trains the underlying model using the 
        provided feature matrix X and target vector y.

        Parameters:
            X (array-like): Feature matrix used for training.
            y (array-like): Target values corresponding to X.
        Returns:
            self.model: The trained model instance.
        """
        
        self.model.fit(X, y)
        return self.model
    
    def predict(self, X):
        """
        Generate predictions for the input data X using the trained model.
        Parameters:
            X (array-like): Input features for which predictions are to be made.

        Returns:
            predictions: Predicted values corresponding to the input data.

        """
        self.predictions = self.model.predict(X)
        return self.predictions
    
class CrossValidation(SurrogateModel, TrainModel):
    """
    
    """

    def __init__(self):
        """
        
        """
        super().__init__()
        pass

    def pipeline(self):
        """
        run the cross validation here
        """

        pass

class EvaluateTraining(SurrogateModel, TrainModel):
    """
    
    """
    def __init__(self):
        """ Instantiate variables """
        super().__init__()
        self.predictions = TrainModel.predict(X)

    def evaluate(self):
        """
        Evaluates the performance of the model's predictions using common regression metrics.
        Calculates and stores the following metrics in the `self.metrics` dictionary:
            - Mean Absolute Error (mae)
            - Mean Squared Error (mse)
            - R-squared Score (r2)
            - Root Mean Squared Error (rmse)

        Returns:
            self.metrics: A dictionary containing the computed regression metrics.
        """

        self.metrics = {
            "mae": mean_absolute_error(self.y, self.predictions),
            "mse": mean_squared_error(self.y, self.predictions),
            "r2": r2_score(self.y, self.predictions),
            "rmse": np.sqrt(mean_squared_error(self.y, self.predictions))
        }

        return self.metrics
    
class ModelReport():
    """
    
    """
    def __init__(self):
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