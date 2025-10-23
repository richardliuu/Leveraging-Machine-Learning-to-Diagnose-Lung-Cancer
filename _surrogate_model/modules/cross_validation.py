import pandas as pd
from sklearn.model_selection import GroupKFold
from .data_methods import LoadData, ClusterData
from .model import SurrogateModel
from .evaluation import EvaluateTraining

class CrossValidation:
    """
    CrossValidation class for executing cross-validation pipelines on tabular data.
    Args:
        n_splits (int): Number of cross-validation splits (folds).
        data_path (str): Path to the CSV file containing the dataset.
    Attributes:
        n_splits (int): Number of cross-validation splits.
        data_path (str): Path to the dataset.
        results (list): List to store evaluation metrics for each fold and cluster.
        final_model (SurrogateModel or None): Trained model from the final fold.
        final_loader (LoadData or None): Data loader from the final fold.
    Methods:
        run_cv(cluster_mode=False):
            Executes the cross-validation pipeline. If `cluster_mode` is True, performs cluster-based cross-validation; otherwise, runs standard cross-validation.
            Returns a list of evaluation results for each fold (and cluster, if applicable).
        _run_full_cv(loader, fold):
            Runs cross-validation on the full dataset for a given fold. Trains a surrogate model and evaluates its performance.
        _run_cluster_cv(loader, fold):
            Runs cross-validation using clusters for a given fold. Trains and evaluates a surrogate model for each cluster separately.
    """
    
    def __init__(self, n_splits, data_path):
        self.n_splits = n_splits
        self.data_path = data_path
        self.results = []
        self.final_model = None
        self.final_loader = None
        
    def run_cv(self, cluster_mode=False):
        """
        Runs cross-validation using GroupKFold on the dataset specified by self.data_path.
        Parameters:
            cluster_mode (bool, optional): If True, runs cross-validation in cluster mode using self._run_cluster_cv.
                                           If False, runs standard cross-validation using self._run_full_cv.
                                           Default is False.
        Process:
            - Loads the dataset from self.data_path.
            - Uses 'patient_id' column for grouping if available.
            - Splits the data into folds using GroupKFold.
            - For each fold:
                - Loads training and testing data indices.
                - Executes the appropriate cross-validation method based on cluster_mode.
            - Prints progress for each fold.
        Returns:
            self.results: The results collected during cross-validation.
        """

        df = pd.read_csv(self.data_path)
        groups = df.get('patient_id', None)
        n_samples = len(df)
        
        gkf = GroupKFold(n_splits=self.n_splits)
        splits = gkf.split(range(n_samples), groups=groups) if groups is not None else \
                 gkf.split(range(n_samples))
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{self.n_splits}")
            print(f"{'='*50}")
            
            loader = LoadData(self.data_path)
            loader.load_data(train_idx, test_idx)
            
            if cluster_mode:
                self._run_cluster_cv(loader, fold)
            else:
                self._run_full_cv(loader, fold)
        
        return self.results
    
    def _run_full_cv(self, loader, fold):
        """
        Executes a single fold of full cross-validation using the provided data loader.
        Trains a SurrogateModel on the training data, evaluates predictions on the test set,
        and stores the evaluation metrics for the current fold. If this is the last fold,
        saves the trained model and loader for later use.
        Args:
            loader: An object containing training and test data, including features and target probabilities.
            fold (int): The current fold index in the cross-validation process.
        Side Effects:
            Appends a dictionary of evaluation metrics and fold information to self.results.
            Sets self.final_model and self.final_loader if this is the last fold.
        """

        model = SurrogateModel()
        model.train(loader.X_train, loader.rf_probs_train)
        
        preds = model.predict(loader.X_test)
        evaluator = EvaluateTraining()
        metrics = evaluator.evaluate(loader.rf_probs_test, preds)
        
        self.results.append({
            'fold': fold + 1,
            'cluster': 'full',
            **metrics
        })
        
        if fold == self.n_splits - 1:
            self.final_model = model
            self.final_loader = loader
    
    def _run_cluster_cv(self, loader, fold):
        """
        Performs cluster-based cross-validation for a given fold.
        This method clusters the training data, trains a surrogate model on each cluster,
        and evaluates the model on the corresponding test cluster. The evaluation metrics
        for each cluster and fold are stored in the `self.results` list.
        Args:
            loader: An object containing training and test datasets, including features,
                random forest probabilities, and original target values. Expected attributes:
                - X_train, rf_probs_train, y_original_train
                - X_test, rf_probs_test, y_original_test
            fold (int): The current fold number (zero-based index).
        Side Effects:
            Appends a dictionary of evaluation metrics for each cluster and fold to `self.results`.
            Prints training and evaluation progress for each cluster.
        """
        
        clusterer = ClusterData()
        train_clusters = clusterer.create_clusters(
            loader.X_train, 
            loader.rf_probs_train,
            loader.y_original_train
        )
        
        for cluster_name, cluster_data in train_clusters.items():
            if len(cluster_data['X']) == 0:
                continue
                
            print(f"\nTraining on {cluster_name} cluster...")
            model = SurrogateModel()
            model.train(cluster_data['X'], cluster_data['y'])
            
            test_clusters = clusterer.create_clusters(
                loader.X_test,
                loader.rf_probs_test,
                loader.y_original_test
            )
            
            test_cluster = test_clusters[cluster_name]
            if len(test_cluster['X']) > 0:
                preds = model.predict(test_cluster['X'])
                evaluator = EvaluateTraining()
                metrics = evaluator.evaluate(test_cluster['y'], preds)
                
                self.results.append({
                    'fold': fold + 1,
                    'cluster': cluster_name,
                    **metrics
                })
                
                print(f"{cluster_name} - R²: {metrics['r2']:.4f}")
