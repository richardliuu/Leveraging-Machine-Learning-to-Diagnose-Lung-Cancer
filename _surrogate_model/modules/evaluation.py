import os 
import pandas as pd 
import matplotlib.pyplot as plt 
from .config import METRICS_DIR, PLOTS_DIR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class EvaluateTraining:
    """
    A class for evaluating regression model performance using common metrics and saving results.
    Attributes:
        metrics (dict): Stores the computed evaluation metrics.
    Methods:
        evaluate(y_true, y_pred):
            Computes mean absolute error (MAE), mean squared error (MSE), and R^2 score between true and predicted values.
        save_metrics(metrics_list, filename="cv_results.csv"):
            Saves a list of metrics dictionaries to a CSV file in the specified metrics directory.
    """
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate(self, y_true, y_pred):
        """
        Evaluates regression model predictions using common metrics.
        Parameters:
            y_true (array-like): Ground truth (correct) target values.
            y_pred (array-like): Estimated target values predicted by the model.
        Returns:
            dict: A dictionary containing the following regression metrics:
                - 'mae': Mean Absolute Error
                - 'mse': Mean Squared Error
                - 'r2': R^2 (coefficient of determination) score
        """
        
        self.metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }
        return self.metrics
    
    @staticmethod
    def save_metrics(metrics_list, filename="cv_results.csv"):
        """
        Saves a list of metric dictionaries to a CSV file.
        Args:
            metrics_list (list of dict): A list where each element is a dictionary containing metric names and values.
            filename (str, optional): The name of the CSV file to save the metrics. Defaults to "cv_results.csv".
        Returns:
            pandas.DataFrame: The DataFrame containing the saved metrics.
        Side Effects:
            - Creates the directory specified by METRICS_DIR if it does not exist.
            - Writes the metrics to a CSV file in METRICS_DIR.
            - Prints the path where the metrics are saved.
        """

        save_path = os.path.join(METRICS_DIR, filename)
        os.makedirs(METRICS_DIR, exist_ok=True)
        
        df_metrics = pd.DataFrame(metrics_list)
        df_metrics.to_csv(save_path, index=False)
        print(f"Metrics saved to {save_path}")

        return df_metrics
    
class ModelReport:
    """
    Generates and stores model performance reports and visualizations.
    This class provides functionality to analyze and visualize model performance metrics
    across cross-validation folds. It can generate plots showing how metrics like R² Score
    and MAE vary across different folds of cross-validation.
    Attributes:
        results_df (pandas.DataFrame, optional): DataFrame to store evaluation results.
            Defaults to None.
    Methods:
        plot_performance(metrics_list, filename): 
            Creates and saves plots showing model performance metrics across CV folds.
    """
    
    def __init__(self):
        self.results_df = None
        
    def plot_performance(self, metrics_list, filename="model_performance.png"):
        """
        Plots model performance metrics across cross-validation folds and saves the plot as an image file.
        Parameters:
            metrics_list (list of dict): A list where each element is a dictionary containing performance metrics
                for a fold. Each dictionary should have at least the keys 'fold', 'r2', and 'mae'.
            filename (str, optional): The name of the file to save the plot to. Defaults to "model_performance.png".
        Saves:
            A PNG image file of the performance plots in the directory specified by PLOTS_DIR.
        Notes:
            - The plot includes R² Score and MAE across folds.
            - The function creates the output directory if it does not exist.
            - The function prints the path where the plot is saved.
        """
        
        save_path = os.path.join(PLOTS_DIR, filename)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        df = pd.DataFrame(metrics_list)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(df['fold'], df['r2'], marker='o', linewidth=2)
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("R² Score")
        axes[0].set_title("R² Score Across Folds")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df['fold'], df['mae'], marker='^', 
                     linewidth=2, color='green')
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("MAE")
        axes[1].set_title("MAE Across Folds")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Performance plot saved to {save_path}")
        plt.close()
