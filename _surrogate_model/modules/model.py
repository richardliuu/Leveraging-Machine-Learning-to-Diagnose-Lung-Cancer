from sklearn.tree import DecisionTreeRegressor
from .config import MODEL_PARAMS

class SurrogateModel:
    """
    Provides an object to call the model for training
    when needed. 

    The surrogate serves as a transparent and simplified representation of 
    the random forest model used to predict lung cancer. This improves model
    explainability and interpretability through the DecisionTreeRegressor's simpler 
    rule-based structure.
    """ 
    def __init__(self):
        """ Instantiate variables """
        self.model = None
        self.predictions = None
        self.X = None
        self.y = None
        self.model_params = MODEL_PARAMS


    def initialize_model(self, model_params=None):
        """
        This function initializes the model and takes in
        the hyperparameters for the DecisionTreeRegressor.
        
        Parameters: See config.py for the model parameters
        """

        params = model_params or self.model_params
        self.model = DecisionTreeRegressor(**params)

        return self.model
    
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
        
        if self.model is None:
            self.initialize_model()

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
