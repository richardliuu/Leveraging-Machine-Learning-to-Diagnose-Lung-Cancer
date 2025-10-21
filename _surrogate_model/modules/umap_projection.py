import os
import matplotlib.pyplot as plt
import umap
from sklearn.tree import export_graphviz
from .config import PLOTS_DIR, MODELS_DIR, UMAP_PARAMS

class UMAPProjection:
    """
    UMAPProjection provides methods to generate and visualize UMAP embeddings for dimensionality reduction.
    Attributes:
        umap_model: The fitted UMAP model instance.
        umap_coords: The coordinates of the data in the UMAP embedding space.
        umap_params: Dictionary of parameters for configuring the UMAP model.
    Methods:
        generate_umap(X, **kwargs):
            Generates a UMAP embedding for the input data X.
            Args:
                X (array-like): Input data to embed.
                **kwargs: Additional UMAP parameters to override defaults.
            Returns:
                np.ndarray: UMAP embedding coordinates.
        visualize_umap(labels, filename="umap_projection.png"):
            Visualizes the UMAP embedding with cluster labels and saves the plot.
            Args:
                labels (array-like): Cluster labels for coloring the points.
                filename (str): Name of the file to save the plot.
    """
    def __init__(self, umap_params=None):
        self.umap_model = None
        self.umap_coords = None
        self.umap_params = umap_params or UMAP_PARAMS
        
    def generate_umap(self, X, **kwargs):
        """
        Generates a UMAP embedding for the given data.
        Parameters:
            X (array-like or sparse matrix of shape (n_samples, n_features)):
                The input data to be embedded.
            **kwargs:
                Additional keyword arguments to override or supplement the default UMAP parameters.
        Returns:
            numpy.ndarray:
                The coordinates of the data in the reduced UMAP space.
        Notes:
            - The method updates the instance's `umap_model` and `umap_coords` attributes.
            - UMAP parameters are taken from `self.umap_params` and can be overridden by `kwargs`.
        """

        params = {**self.umap_params, **kwargs}
        self.umap_model = umap.UMAP(**params)
        self.umap_coords = self.umap_model.fit_transform(X)
        
        return self.umap_coords
    
    def visualize_umap(self, labels, filename="umap_projection.png"):
        """
        Visualizes the UMAP projection of features and saves the resulting plot as an image file.
        Parameters:
            labels (array-like): Cluster or class labels for each data point, used for coloring the scatter plot.
            filename (str, optional): Name of the file to save the plot. Defaults to "umap_projection.png".
        Saves:
            The UMAP projection plot as an image file in the directory specified by PLOTS_DIR.
        Notes:
            - Assumes that `self.umap_coords` contains the 2D UMAP coordinates for the data points.
            - The plot uses a categorical colormap ('tab10') for coloring points by their labels.
        """

        save_path = os.path.join(PLOTS_DIR, filename)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            self.umap_coords[:, 0], 
            self.umap_coords[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('UMAP Projection of Features')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"UMAP plot saved to {save_path}")
        plt.close()

class InterpretPredictions:
    """
    A utility class for interpreting and visualizing machine learning model predictions.
    Methods
    -------
    visualize_tree(model, feature_names, filename="decision_tree.dot"):
        Exports a visualization of a decision tree model to a DOT file for further rendering.
    """
    
    def __init__(self):
        self.tree_viz = None
        
    @staticmethod
    def visualize_tree(model, feature_names, filename="decision_tree.dot"):
        """
        Visualizes and exports a decision tree model to a DOT file for further rendering.
        Args:
            model: The trained decision tree model (e.g., from scikit-learn) to visualize.
            feature_names (list of str): List of feature names corresponding to the model's input features.
            filename (str, optional): Name of the output DOT file. Defaults to "decision_tree.dot".
        Side Effects:
            - Saves the DOT file representing the decision tree to the specified path.
            - Prints the path to the exported DOT file and instructions for rendering it as a PNG.
        Raises:
            OSError: If the directory for saving the DOT file cannot be created.
        """
        
        save_path = os.path.join(MODELS_DIR, filename)
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        export_graphviz(
            model,
            out_file=save_path,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            special_characters=True
        )

        print(f"Decision tree exported to {save_path}")
        print("Render with: dot -Tpng decision_tree.dot -o decision_tree.png")
