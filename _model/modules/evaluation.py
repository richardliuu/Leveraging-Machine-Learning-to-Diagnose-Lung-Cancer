import matplotlib.pyplot as plt 
import pandas as pd 
import os
from sklearn.metrics import classification_report
from .config import METRICS_DIR, PLOTS_DIR, RESULTS_DIR

class Evaluation:
    """
    A class for evaluating machine learning models, generating performance plots, and saving evaluation metrics.
    Methods
    -------
    __init__():
        Initializes the Evaluation object.
    result(save_path):
        Static method to generate and save a three-panel plot of model evaluation metrics (F1 score, precision, recall) across folds.
        Expects a pandas DataFrame `df` with columns: 'fold', 'f1_score', 'precision', 'recall'.
    save_metrics(save_path, metrics_list, filename):
        Saves a list of metrics to a CSV file in the specified directory and returns the DataFrame.
    evaluate(X, y):
        Evaluates the model using the provided data and returns a classification report as a dictionary.
    """

    def __init__(self):
        self.report = None

    def result(self, metrics_list, save_path):
        """
        Generate and save a three-panel plot of model evaluation metrics across folds.
        The function creates a single-row figure with three subplots showing:
        - F1 score per fold
        - Precision per fold
        - Recall per fold

        Each subplot uses the values in the DataFrame columns named 'fold' (x-axis)
        and one of 'f1_score', 'precision', or 'recall' (y-axis). 

        Parameters
        ----------
        save_path : str or os.PathLike
            Destination file path where the generated plot image will be saved.
        Notes
        -----
        This function expects a pandas.DataFrame named `df` to be available in the
        function's scope (or as an attribute accessible to the method). The DataFrame
        must contain the following columns:
        - 'fold' : fold identifier or index (x-axis)
        - 'f1_score' : F1 score for each fold
        - 'precision' : precision for each fold
        - 'recall' : recall for each fold

        Raises
        ------
        KeyError
            If one or more of the required columns ('fold', 'f1_score', 'precision',
            'recall') are missing from `df`.
        OSError
            If the output file cannot be written to `save_path`.
        TypeError
            If `save_path` is not a string or path-like object.
        """

        df = pd.DataFrame(metrics_list)
        
        fig, axes = plt.subplots(1, 3, (10, 5))

        # F1 Score per Fold
        axes[0].plot(df['fold'], df['f1_score'], marker = '', linewidth=2)
        axes[0].xlabel("Fold")
        axes[0].ylabel("F1 Score")
        axes[0].title("Model F1 Score Across Folds")
        axes[0].grid(True, alpha=0.3)
    
        # Precision
        axes[1].plot(df['fold'], df['precision'], marker = '', linewidth = )
        axes[1].xlabel("Fold")
        axes[1].ylabel("Precision")
        axes[1].title("Model Precision Across Folds")
        axes[1].grid(True, alpha=0.3)

        # Recall
        axes[2].plot(df['fold'], df['recall'], marker = '', linewidth = )
        axes[2].xlabel("Fold")
        axes[2].ylabel("Recall")
        axes[2].title("Model Recall Across Folds")
        axes[2].grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Performance plot saved to {save_path}")
        plt.close()
    
    @staticmethod
    def save_metrics(self, save_path, metrics_list, filename):
        """
        Saves a list of metrics to a CSV file.
        Args:
            save_path (str): The directory path where the metrics file will be saved.
            metrics_list (list): A list of metric dictionaries or objects to be saved.
            filename (str): The name of the CSV file to save the metrics.
        Returns:
            pandas.DataFrame: The DataFrame containing the saved metrics.
        Side Effects:
            - Creates the metrics directory if it does not exist.
            - Writes the metrics to a CSV file at the specified location.
            - Prints a message indicating where the metrics were saved.
        """

        save_path = os.path.join(METRICS_DIR, filename)
        os.makedirs(METRICS_DIR, exist_ok=True)
        
        df_metrics = pd.DataFrame(metrics_list)
        df_metrics.to_csv(save_path, index=False)

        print(f"Metrics saved to {save_path}")
        return df_metrics

    def evaluate(self, X, y):
        """
        
        """ 

        self.report = classification_report(self.y_test, self.y_pred, output_dict=True)

        return self.report 

