"""
Author: Richard Liu
Date: October, 2025 
Description: 

High-level module docstring
---------------------------
This module provides a small framework for training and analyzing simple surrogate models
that approximate the behavior of a pre-trained Random Forest classifier. The surrogate
models are intended to improve interpretability by modeling the Random Forest's predicted
probabilities (for the positive class) on selected subsets of the data that are grouped
by the model's confidence.
Primary responsibilities:
- Load predictions and features produced by an existing Random Forest pipeline.
- Partition samples into confidence-based clusters (e.g. confident / mixed / unconfident).
- Train lightweight surrogate regressors (e.g. decision tree, linear models) to fit the
    Random Forest's predicted probabilities on those clusters.
- Evaluate surrogate performance using standard regression metrics and cross-validation.
- Produce simple visualizations including UMAP embeddings and model reports.
- Export artifacts such as trained surrogates, evaluation results, and plot images.
Intended workflow
-----------------
1. Load a CSV containing features, the Random Forest's predicted probabilities
     (prob_class_0, prob_class_1), the hard predicted label, and any metadata (patient_id,
     fold, chunk, test_idx, ...).
2. Exclude columns that would leak target information (true_label, predicted_label,
     prob_class_0/1) from the feature set and extract:
         - X: features used by surrogates
         - rf_probs: Random Forest probability for class 1 (regression target)
         - rf_hard_preds: Random Forest's hard label predictions (for validation/analysis)
         - y_original: True labels (for additional validation / analysis)
3. Partition the dataset into clusters based on rf_probs (e.g., >0.7 confident,
     0.4-0.7 mixed, <0.4 unconfident).
4. For each cluster (or for the full dataset), initialize and train a surrogate model.
     Typical surrogates include DecisionTreeRegressor (for simple rule-based explanations),
     Lasso/ElasticNet (for sparse linear explanations), or polynomial regressors.
5. Evaluate surrogates using MAE, MSE, RMSE and R²; optionally perform cross-validation
     and export fold-level metrics.
6. Produce UMAP projections of the original feature matrix and overlay cluster assignments
     or surrogate residuals for visualization.
7. Provide utilities for exporting a decision tree (DOT) visualization and saving results.
Core classes and responsibilities
---------------------------------
LoadData
        - Loads and preprocesses the CSV file containing Random Forest outputs.
        - Excludes leakage columns and returns feature matrix X, rf_probs (target), hard
            predictions, true labels, and feature column names.
        - Intended to support row/ index-based train/test slicing passed in from an outer
            cross-validation controller.
ClusterData
        - Splits the dataset into confidence-based clusters (confident, mixed, unconfident)
            according to configurable probability thresholds.
        - Stores cluster-specific DataFrames or views that include both features and RF
            predictions to enable per-cluster surrogate training.
SurrogateModel
        - Lightweight wrapper around sklearn regressors to unify initialization, training,
            and prediction APIs used by downstream utilities.
        - Example usage: initialize_model; train(X, y); predict(X).
EvaluateTraining (inherits SurrogateModel)
        - Computes regression evaluation metrics (MAE, MSE, RMSE, R²) comparing surrogate
            predictions to RF probabilities or true labels as required.
        - Provides a small API to export metrics (e.g., to CSV) for cross-validation reporting.
ModelReport (inherits SurrogateModel)
        - Utilities for plotting and summarizing surrogate performance across folds or clusters.
        - Minimal plotting scaffold is included to surface foldwise R² (can be extended).
UMAPProjection
        - Produces low-dimensional UMAP embeddings of feature data for visualization.
        - Stores fitted UMAP model and 2D coordinates to support plotting and exporting figures.
        - Intended to accept X (feature matrix) and optional labels (cluster ids, residuals)
            to color points in visualizations.
InterpretPredictions (inherits SurrogateModel)
        - Helper for exporting interpretable representations of surrogates (e.g., decision tree
            DOT export). Designed to wrap sklearn.tree export utilities and to orchestrate saving
            or printing model structure.
CrossValidation (inherits SurrogateModel)
        - High-level driver that should orchestrate group-aware cross-validation (e.g.,
            GroupKFold) over patients or other grouping unit.
        - For each fold: performs train/test splitting (via LoadData), trains a surrogate,
            evaluates it, and collects fold metrics. Intended extension point for logging and
            artifact export.
RunPipeline
        - Top-level orchestration that wires together LoadData, ClusterData, CrossValidation,
            ModelReport, InterpretPredictions, and UMAPProjection to run the complete pipeline
            from raw CSV to final outputs (models, metrics, and plots).
        - Meant as a convenience wrapper for reproducible experiments; can be adapted to
            run per-cluster surrogate experiments.
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
from sklearn.tree import DecisionTreeRegressor, export_graphviz
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

    def load_data(self, train_idx, test_idx):
        """
        This function loads data for the program and 
        """
        self.df = pd.read_csv(self.data)

        # Exclude columns to remove data leakage and crucial/not important information
        self.exclude_cols = ['true_label', 'predicted_label', 'prob_class_0', 'prob_class_1', 
                    'patient_id', 'chunk', 'fold', 'test_idx']
        
        # Include all columns that were excluded 
        self.feature_cols = [col for col in self.df.columns if col not in self.exclude_cols]

        # Extract specific information for training 
        self.X_original = self.df[self.feature_cols].values  # Original features 
        self.rf_probabilities = self.df['prob_class_1'].values  # TARGET: Class 1 probabilities
        self.rf_hard_predictions = self.df['predicted_label'].values  # For validation 
        self.y_original = self.df['true_label'].values  # For validation 

        self.X_train, self.X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
        self.y_train, self.y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
        
        # Print out information on the dataset 
        print(f"Data shape: {self.X_original.shape}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Samples: {len(self.X_original)}")
        print(f"Target range: [{self.rf_probs.min():.3f}, {self.rf_probs.max():.3f}]")
        print(f"Target mean: {self.rf_probs.mean():.3f} +- {self.rf_probs.std():.3f}")
        
        return self.X_original, self.rf_probs, self.rf_hard_preds, self.y_original, self.feature_cols

class ClusterData:
    """
    This class module clusters data based on the model's 
    confidence 
    """
    def __init__(self, confident_threshold=0.7, unconfident_threshold=0.4):
        """ Instantiate variables """
        super().__init__(LoadData)
        self.confident_threshold = confident_threshold
        self.unconfident_threshold = unconfident_threshold
        self.clusters = {}

    def cluster(self, X, rf_probs, y_original = None):
        """
        This function clusters the data and creates seperate datasets. 
        """
         
        """
        NOTE ALSO NEED TO APPEND CORRESPONDING X FEATURES 
        """

        """
        Create the clusters as dictionaries and seperate them 
        in the dictionary cluster
        """
        self.clusters = {
            "confident": {
                "X": ,
                "y": , 
                "y_original": ,
                "indices": 
            },
            "mixed": {
                "X": ,
                "y": , 
                "y_original": ,
                "indices": 

            },
            "unconfident": {
                "X": ,
                "y": , 
                "y_original": ,
                "indices": 

            },
        }

        print(f"\nCluster sizes:")
        for name, data in self.clusters.items():
            print(f" {name}: {len(data["X"])} samples")

        return self.clusters
  
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
        self.predictions = None
        self.X = None
        self.y = None

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

class EvaluateTraining(SurrogateModel):
    """
    A class for evaluating the performance of a surrogate model on training data.
    This class inherits from SurrogateModel and provides methods to:
        - Generate predictions using the surrogate model.
        - Evaluate regression metrics such as MAE, MSE, R2, and RMSE.
        - Export evaluation metrics to a CSV file.
    Attributes:
        predictions (array-like): Model predictions for the input features.
        metrics (dict): Dictionary containing computed regression metrics.
    Methods:
        __init__():
            Initializes the EvaluateTraining instance and generates predictions using the surrogate model.
        evaluate():
            Computes and returns regression metrics (MAE, MSE, R2, RMSE) comparing predictions to true values.
        cross_validation_report():
            Exports the computed metrics to a CSV file for cross-validation reporting.
    """

    def __init__(self):
        """ Instantiate variables """
        super().__init__()
        self.predictions = SurrogateModel.predict(self.X)

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
    
    def save_metrics(self, metrics_list, save_path = r"results/surrogate_training_results.csv")
        """
        
        NOTE DOCUMENTATION

        """

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        df_metrics = pd.DataFrame(metrics_list)
        df_metrics.to_csv(save_path, index=False)

        print(f"Metrics saved to {save_path}")

        return df_metrics
    
class ModelReport(SurrogateModel):
    """
    A subclass of SurrogateModel that provides reporting and visualization 
    functionalities for model performance.
    Methods
    -------
    __init__():
        Initializes the ModelReport instance and calls the parent SurrogateModel initializer.
    graph():
        Generates and displays a figure with three subplots to visualize the model's 
        performance metrics across different folds. The first subplot shows the R2 Score 
        across folds, while the other two subplots are placeholders for additional metrics.
    """
        
    def __init__(self):
        super().__init__()
        self.results_df = None

    def plot_performance(self, metrics_list, save_path = r"results/"):
        """ 
        Graph the performance of the model 
        """

        df = pd.DataFrame(metrics_list)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # R2 Score
        axes[0].plot(df["fold"], df["r2"], marker="o", linewidth=2, color="orange")
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("R2 Score")
        axes[0].set_title("R2 Score Across Folds")
        axes[0].grid(True, alpha=0.3)

        # Root Meant Squared Error
        axes[1].plot(df["fold"], df["rmse"], marker="s", linewidth=2, color="green")
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("RMSE Across Folds")
        axes[1].grid(True, alpha=0.3)

        # Mean Absolute Error
        axes[2].plot(df["fold"], df["mae"], marker="^", linewidth=2, color="blue")
        axes[2].set_xlabel("Fold")
        axes[2].set_ylabel("MAE")
        axes[2].set_title("MAE Across Folds")
        axes[2].grid(True, alpha=0.3)

        # Save the results 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches = 'tight')

        # Print path
        print(f"Performance plot saved to {save_path}")

        plt.show()

class UMAPProjection:
    """
    A class for creating and managing UMAP projections to provide
    visuals of the dataset through dimensionality reduction.

    Attributes:
        umap_model: An instance of the UMAP model.
    Methods:
        generate_umap():
            Initializes and returns a UMAP model with predefined parameters.
        visualize_umap():
            Placeholder for a method to visualize the UMAP projection.
    """

    def __init__(self):
        self.umap_model = None
        self.umap_coords = None
        self.X_original = None
        self.y_original = None

    def generate_umap(self):
        """
        Generates a UMAP embedding of the original feature matrix.
        This method initializes a UMAP model with predefined parameters and 
        fits it to the original data (`self.X_original`).
        It returns both the fitted UMAP model and the resulting 2D coordinates.
        Returns:
            tuple: 
                - umap.UMAP: The fitted UMAP model.
                - numpy.ndarray: The 2D UMAP coordinates of the input data.
        """

        self.umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42,
            metric='euclidean'
        )

        self.umap_coords = self.umap_model.fit_transform(self.X_original)

        return self.umap_model, self.umap_coords
    
    def visualize_umap(self, save_path=r"results/surrogate_umap_projection.png"):
        """
        Visualizes the data using UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction.
        This method projects high-dimensional data into a lower-dimensional space (typically 2D or 3D)
        using UMAP and displays the resulting visualization using matplotlib.
        """

        # Create multiple plots 
        axes = plt.plot(figsize=(10, 8))
        
        # Plot 1: Clusters
        scatter1 = axes[0].scatter(self.umap_coords[:, 0], 
                                   self.umap_coords[:, 1], 
                                   cmap='tab10', 
                                   alpha=0.6, 
                                   s=10)
        
        # Project Features
        axes[0].set_title('Cluster Assignments')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        axes[0].set_title("UMAP Projection of Feature Space")
        axes[0].grid(True, alpha=0.3)

        # Save the result
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        # Print the saved path
        print(f"Performance plot saved to {save_path}")
        
        plt.show()

class InterpretPredictions(SurrogateModel):
    """
    InterpretPredictions is a subclass of SurrogateModel designed to provide interpretability for model predictions.
    It includes methods to visualize the underlying surrogate model, typically a decision tree, by exporting its structure.
    Attributes:
        tree (str or None): Stores the exported visualization of the model, typically in DOT format.
    Methods:
        __init__(): Initializes the InterpretPredictions instance and its attributes.
        visualize_model(): Exports and prints the visualization of the surrogate model.
    """

    def __init__(self):
        """ Instantiate variables """
        super().__init__()
        self.tree = None
    
    def visualize_model(self, feature_names, save_path = r"results/decision_tree.dot"):
        """
        Export the fitted decision-tree surrogate model to a Graphviz DOT file and store a
        reference to the exported tree on the instance.
        This method will create any missing parent directories for save_path, call the
        model's export_graphviz routine to produce a DOT representation of the tree, and
        assign the exported graph/text to self.tree. 

        Parameters
        ----------
        feature_names : Sequence[str]
            Names of the features to use as labels in the rendered decision tree.
        save_path : str, optional
            Filesystem path for the output DOT file. If the parent directory does not
            exist it will be created. Default: r"results/decision_tree.dot".
        Returns
        -------
        None

        Raises
        ------
        OSError
            If the output directory cannot be created or the file cannot be written.
        AttributeError
            If the model does not provide an export_graphviz method.
        ValueError, TypeError
            If feature_names is not a suitable sequence of strings accepted by the
            underlying export function.

        Notes
        -----
        The exported DOT file uses filled nodes, rounded corners, and special characters
        for improved readability. To produce an image from the DOT file, use the Graphviz
        `dot` tool (installed separately).
        """
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.tree = self.model.export_graphviz(
            out_file = save_path,
            feature_names = feature_names,
            filled = True,
            rounded = True,
            special_characters = True
        )

        print(f"Visual exported to {save_path}")
        print("Use 'dot -Tpng decision_tree.dot -o decision_tree.png' to render")

class CrossValidation(SurrogateModel):
    """
    CrossValidation class for performing cross-validation on surrogate models.
    Attributes:
        train_idx (array-like): Indices for training data in the current fold.
        test_idx (array-like): Indices for test data in the current fold.
        performance_metrics (dict or None): Stores performance metrics for each fold.
        groupkfold (int): Number of folds for group-based cross-validation.
        X (array-like): Feature matrix.
        y (array-like): Target vector.
    Methods:
        __init__():
            Initializes the CrossValidation instance and its attributes.
        pipeline():
            Executes the main cross-validation pipeline. Iterates over folds,
            splits data, trains the model, and evaluates performance.
    """

    def __init__(self):
        """ Instantiate variables"""
        super().__init__()
        self.train_idx = None
        self.text_idx = None
        self.performance_metrics = None
        self.groupkfold = 4
        self.X = None
        self.y = None

    def pipeline(self):
        """
        REWRITE DOC AFTER COMPLETION

        Executes the main processing pipeline for the class.
        Iterates overdata, performing necessary operations
        at each stage. The specific actions and data processed within the pipeline should be
        implemented in the method body.
        """
        
        for fold, (self.train_idx, self.test_idx) in enumerate(self.groupkfold.split(X, y, groups), ):
            print(f"===== Fold {fold} =====")
            self.model.train()
            
            # Split data for current fold
            handler.split(df, train_idx, test_idx)
            
            print(f"Train: {len(handler.train_patients)} patients, {len(handler.X_train)} samples")
            print(f"Test:  {len(handler.test_patients)} patients, {len(handler.X_test)} samples")
            
            model.train(handler.X_train, handler.y_train, handler.feature_cols)
            
            # Evaluate model performance with overfitting check
            report, c_matrix, auc, train_accuracy, overfitting_gap = model.evaluate(
                handler.X_test, handler.y_test, handler.X_train, handler.y_train
            )


            """
            maybe static method to append results?
            """
            CrossValidation.append_result()

class RunPipeline:
    """
    RunPipeline orchestrates the end-to-end machine learning workflow, including data loading, clustering, model training, performance reporting, and visualization.
    Attributes:
        model_instance (SurrogateModel): Instance of the surrogate model used for training and prediction.
        model_report (ModelReport): Instance for generating model performance reports.
        model_interpret (InterpretPredictions): Instance for interpreting model predictions.
    Methods:
        pipeline():
            Executes the complete pipeline:
                - Loads data using LoadData.
                - Clusters data using ClusterData.
                - Trains the model with cross-validation via CrossValidation.
                - Generates performance reports using ModelReport.
                - Produces and visualizes UMAP projections with UMAPProjection.
    """
    
    def __init__(self):
        self.model_instance = SurrogateModel()
        self.model_report = ModelReport()
        self.model_interpret = InterpretPredictions()

    def pipeline():
        """
        Executes the main data processing and modeling pipeline.
        Steps performed:
            1. Loads the dataset using LoadData.
            2. Clusters the data using ClusterData.
            3. Trains the model and performs cross-validation via CrossValidation.
            4. Reports model performance using ModelReport.
            5. Generates and visualizes UMAP projections with UMAPProjection.
        Returns:
            None
        """
        
        # Load Data
        LoadData.load_data()

        # Cluster Data
        ClusterData.cluster()

        # Train Model 
        CrossValidation.pipeline()

        # Report Model Performance 
        ModelReport.graph()
        InterpretPredictions.visualize_model()

        # UMAP Projections/Visualizations
        UMAPProjection.generate_umap()
        UMAPProjection.visualize_umap()

# Call classes and run pipeline
if __name__ == "__main__":
    RunPipeline.pipeline()