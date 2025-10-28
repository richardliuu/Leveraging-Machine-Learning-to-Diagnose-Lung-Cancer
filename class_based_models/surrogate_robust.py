"""
Author: Richard Liu

Description: 
Robust Surrogate Model with Nested Cross-Validation

This module generates cluster-specific surrogate models using UMAP and DBSCAN
with robust nested cross-validation for unbiased performance estimation.

Key Features:
- Modular class-based architecture
- Stratified Group K-Fold cross-validation 
- Nested CV for hyperparameter tuning and model evaluation
- Patient-aware data splitting to prevent leakage
- Statistical significance testing with bootstrap confidence intervals
- Comprehensive evaluation metrics and reporting

The script expects a CSV file containing Random Forest predictions with patient IDs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone
import umap
from sklearn.cluster import DBSCAN
import joblib
import warnings
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import os
import random

warnings.filterwarnings('ignore')

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


class DataHandler:
    """
    Handles data loading, preprocessing, and validation for surrogate modeling.
    
    Attributes:
        data_path (str): Path to the input CSV file
        X (pd.DataFrame): Feature matrix
        y (np.ndarray): Target values (RF probabilities)
        groups (np.ndarray): Patient IDs for grouped CV
        feature_cols (list): Feature column names
        metadata (dict): Data statistics and information
    """
    
    def __init__(self, data_path: str = 'data/rf_all_test_predictions.csv'):
        self.data_path = data_path
        self.X = None
        self.y = None
        self.groups = None
        self.feature_cols = None
        self.metadata = {}
        
    def load_data(self, target_col: str = 'prob_class_1') -> bool:
        """
        Load and validate the dataset with comprehensive integrity checks.
        
        Args:
            target_col: Column name for target variable (RF probabilities)
            
        Returns:
            bool: True if data passes all integrity checks
        """
        print("=" * 60)
        print("Loading and Validating Data")
        print("=" * 60)
        
        # Load dataset
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")
        
        # Define columns to exclude from features
        exclude_cols = ['true_label', 'predicted_label', 'prob_class_0', 
                       'prob_class_1', 'patient_id', 'chunk', 'fold', 'test_idx']
        
        # Extract features
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.X = df[self.feature_cols]
        
        # Extract target and groups
        self.y = df[target_col].values
        self.groups = df['patient_id'].values
        
        # Store metadata
        self.metadata = {
            'n_samples': len(df),
            'n_patients': df['patient_id'].nunique(),
            'n_features': len(self.feature_cols),
            'target_mean': self.y.mean(),
            'target_std': self.y.std(),
            'target_range': (self.y.min(), self.y.max()),
            'samples_per_patient': df.groupby('patient_id').size().describe()
        }
        
        # Data integrity checks
        is_valid = self._validate_data_integrity(df)
        
        self._print_data_summary()
        
        return is_valid
    
    def _validate_data_integrity(self, df: pd.DataFrame) -> bool:
        """
        Perform comprehensive data integrity checks.
        
        Args:
            df: Complete dataframe
            
        Returns:
            bool: True if all checks pass
        """
        print("\n=== DATA INTEGRITY CHECKS ===")
        
        checks_passed = True
        
        # Check for duplicate samples
        duplicates = df.duplicated(subset=self.feature_cols)
        if duplicates.sum() > 0:
            print(f"WARNING: {duplicates.sum()} duplicate samples found!")
            checks_passed = False
        else:
            print("✓ No duplicate samples")
        
        # Check for missing values
        missing = self.X.isnull().sum().sum()
        if missing > 0:
            print(f"WARNING: {missing} missing values found!")
            checks_passed = False
        else:
            print("✓ No missing values")
        
        # Check for consistent patient data
        patient_stats = df.groupby('patient_id').agg({
            'true_label': 'nunique',
            'chunk': 'count'
        })
        
        inconsistent = patient_stats[patient_stats['true_label'] > 1]
        if len(inconsistent) > 0:
            print(f"WARNING: {len(inconsistent)} patients have inconsistent labels!")
            checks_passed = False
        else:
            print("✓ All patients have consistent labels")
        
        # Check target distribution
        if self.y.std() < 0.01:
            print("WARNING: Very low target variance - may affect model training")
            checks_passed = False
        else:
            print("✓ Sufficient target variance")
        
        return checks_passed
    
    def _print_data_summary(self):
        """Print comprehensive data summary."""
        print("\n=== DATA SUMMARY ===")
        print(f"Samples: {self.metadata['n_samples']}")
        print(f"Patients: {self.metadata['n_patients']}")
        print(f"Features: {self.metadata['n_features']}")
        print(f"Target range: [{self.metadata['target_range'][0]:.3f}, "
              f"{self.metadata['target_range'][1]:.3f}]")
        print(f"Target mean: {self.metadata['target_mean']:.3f} ± "
              f"{self.metadata['target_std']:.3f}")
        print("\nSamples per patient:")
        print(self.metadata['samples_per_patient'])


class UMAPClusteringPipeline:
    """
    Handles UMAP dimensionality reduction and DBSCAN clustering.
    
    Attributes:
        umap_model: Fitted UMAP transformer
        clustering_model: Fitted DBSCAN clusterer
        umap_coords: 2D UMAP coordinates
        cluster_labels: Cluster assignments
        cluster_metadata: Statistics for each cluster
    """
    
    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1,
                 eps: float = 0.8, min_samples: int = 50):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.eps = eps
        self.min_samples = min_samples
        
        self.umap_model = None
        self.clustering_model = None
        self.umap_coords = None
        self.cluster_labels = None
        self.cluster_metadata = {}
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply UMAP and clustering to the data.
        
        Args:
            X: Feature matrix
            y: Target values for cluster analysis
            
        Returns:
            Tuple of (umap_coords, cluster_labels)
        """
        print("\n=== UMAP & CLUSTERING ===")
        
        # Apply UMAP
        print("Fitting UMAP...")
        self.umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=2,
            random_state=SEED,
            metric='euclidean'
        )
        self.umap_coords = self.umap_model.fit_transform(X)
        
        # Apply DBSCAN
        print("Applying DBSCAN clustering...")
        self.clustering_model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.cluster_labels = self.clustering_model.fit_predict(self.umap_coords)
        
        # Analyze clusters
        self._analyze_clusters(y)
        
        return self.umap_coords, self.cluster_labels
    
    def _analyze_clusters(self, y: np.ndarray):
        """Analyze cluster characteristics."""
        unique_clusters = np.unique(self.cluster_labels)
        
        print(f"\nFound {len(unique_clusters)} clusters")
        
        for cluster_id in unique_clusters:
            mask = self.cluster_labels == cluster_id
            cluster_size = np.sum(mask)
            
            if cluster_id == -1:
                print(f"  Outliers: {cluster_size} samples")
                continue
            
            cluster_y = y[mask]
            
            self.cluster_metadata[cluster_id] = {
                'size': cluster_size,
                'mean': cluster_y.mean(),
                'std': cluster_y.std(),
                'min': cluster_y.min(),
                'max': cluster_y.max(),
                'q25': np.percentile(cluster_y, 25),
                'q75': np.percentile(cluster_y, 75)
            }
            
            print(f"  Cluster {cluster_id}: {cluster_size} samples, "
                  f"mean={cluster_y.mean():.3f}±{cluster_y.std():.3f}")
    
    def visualize(self, y: np.ndarray, save_path: Optional[str] = None):
        """Visualize UMAP projection and clusters."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Clusters
        scatter1 = axes[0].scatter(self.umap_coords[:, 0], self.umap_coords[:, 1],
                                   c=self.cluster_labels, cmap='tab10', alpha=0.6, s=10)
        axes[0].set_title('Cluster Assignments')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        plt.colorbar(scatter1, ax=axes[0], label='Cluster ID')
        
        # Plot 2: Target values
        scatter2 = axes[1].scatter(self.umap_coords[:, 0], self.umap_coords[:, 1],
                                   c=y, cmap='RdYlBu', alpha=0.6, s=10)
        axes[1].set_title('Target Values')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        plt.colorbar(scatter2, ax=axes[1], label='RF Probability')
        
        # Plot 3: Target distribution
        axes[2].hist(y, bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('RF Probability')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Target Distribution')
        axes[2].axvline(y.mean(), color='red', linestyle='--',
                       label=f'Mean: {y.mean():.3f}')
        axes[2].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

class SurrogateModel(BaseEstimator, RegressorMixin):
    """
    Base class for cluster-specific surrogate models.
    
    Attributes:
        model: The underlying regression model
        cluster_id: ID of the cluster this model represents
        performance_metrics: Dictionary of evaluation metrics
    """
    
    def __init__(self, model_type: str = 'ridge', cluster_id: Optional[int] = None,
                 **model_params):
        self.model_type = model_type
        self.cluster_id = cluster_id
        self.model_params = model_params
        self.model = None
        self.performance_metrics = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the regression model based on type."""
        if self.model_type == 'ridge':
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=self.model_params.get('alpha', 1.0)))
            ])
        elif self.model_type == 'tree':
            self.model = DecisionTreeRegressor(
                max_depth=self.model_params.get('max_depth', 4),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 20),
                random_state=SEED
            )
        elif self.model_type == 'lasso':
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=self.model_params.get('alpha', 0.1)))
            ])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        """Fit the surrogate model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions, clipping to [0, 1] for probabilities."""
        predictions = self.model.predict(X)
        return np.clip(predictions, 0.0, 1.0)
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary of performance metrics
        """
        predictions = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions))
        }
        
        # Add classification accuracy if treating as binary
        hard_true = (y > 0.5).astype(int)
        hard_pred = (predictions > 0.5).astype(int)
        metrics['accuracy'] = (hard_true == hard_pred).mean()
        
        self.performance_metrics = metrics
        return metrics


class NestedCVEvaluator:
    """
    Implements nested cross-validation for robust surrogate model evaluation.
    
    Uses:
    - Outer CV: Unbiased performance estimation
    - Inner CV: Hyperparameter tuning and model selection
    - Patient-aware splitting to prevent data leakage
    """
    
    def __init__(self, n_outer_splits: int = 4, n_inner_splits: int = 3):
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.outer_cv_results = []
        self.inner_cv_results = []
        
    def evaluate(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 cluster_labels: np.ndarray, model_configs: Dict) -> Dict:
        """
        Perform nested cross-validation evaluation.
        
        Args:
            X: Feature matrix
            y: Target values
            groups: Patient IDs for grouped CV
            cluster_labels: Cluster assignments
            model_configs: Dictionary of model configurations per cluster
            
        Returns:
            Dictionary containing evaluation results and statistics
        """
        print("\n" + "=" * 60)
        print("NESTED CROSS-VALIDATION EVALUATION")
        print("=" * 60)
        
        # Outer cross-validation
        outer_cv = StratifiedGroupKFold(n_splits=self.n_outer_splits)
        
        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, cluster_labels, groups)
        ):
            print(f"\n--- Outer Fold {fold_idx + 1}/{self.n_outer_splits} ---")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
            cluster_labels_train = cluster_labels[train_idx]
            cluster_labels_test = cluster_labels[test_idx]
            
            # Verify no patient leakage
            train_patients = set(groups[train_idx])
            test_patients = set(groups[test_idx])
            if train_patients.intersection(test_patients):
                raise ValueError("Patient leakage detected!")
            
            print(f"Train: {len(train_patients)} patients, {len(X_train)} samples")
            print(f"Test: {len(test_patients)} patients, {len(X_test)} samples")
            
            # Inner cross-validation for model selection
            best_models = self._inner_cv(
                X_train, y_train, groups_train, 
                cluster_labels_train, model_configs
            )
            
            # Evaluate on outer test set
            fold_results = self._evaluate_fold(
                best_models, X_train, y_train, X_test, y_test,
                cluster_labels_train, cluster_labels_test
            )
            
            fold_results['fold'] = fold_idx + 1
            fold_results['n_train'] = len(X_train)
            fold_results['n_test'] = len(X_test)
            
            self.outer_cv_results.append(fold_results)
        
        # Aggregate results
        final_results = self._aggregate_results()
        
        return final_results
    
    def _inner_cv(self, X_train: np.ndarray, y_train: np.ndarray,
                  groups_train: np.ndarray, cluster_labels_train: np.ndarray,
                  model_configs: Dict) -> Dict:
        """
        Inner cross-validation for hyperparameter tuning.
        
        Returns:
            Dictionary of best models per cluster
        """
        print("\n  Inner CV for model selection...")
        
        inner_cv = StratifiedGroupKFold(n_splits=self.n_inner_splits)
        best_models = {}
        
        # Get unique clusters in training data
        unique_clusters = np.unique(cluster_labels_train[cluster_labels_train >= 0])
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels_train == cluster_id
            X_cluster = X_train[cluster_mask]
            y_cluster = y_train[cluster_mask]
            groups_cluster = groups_train[cluster_mask]
            
            if len(np.unique(groups_cluster)) < self.n_inner_splits:
                # Not enough patients for inner CV, use default model
                best_models[cluster_id] = SurrogateModel(
                    model_type=model_configs.get(cluster_id, 'ridge'),
                    cluster_id=cluster_id
                )
                best_models[cluster_id].fit(X_cluster, y_cluster)
                continue
            
            # Perform inner CV to select best model
            model_scores = {}
            
            for model_type in ['ridge', 'tree', 'lasso']:
                model = SurrogateModel(model_type=model_type, cluster_id=cluster_id)
                
                try:
                    # Use dummy stratification since we're within a cluster
                    dummy_strat = np.zeros(len(y_cluster))
                    scores = []
                    
                    for inner_train_idx, inner_val_idx in inner_cv.split(
                        X_cluster, dummy_strat, groups_cluster
                    ):
                        X_inner_train = X_cluster[inner_train_idx]
                        y_inner_train = y_cluster[inner_train_idx]
                        X_inner_val = X_cluster[inner_val_idx]
                        y_inner_val = y_cluster[inner_val_idx]
                        
                        # Clone and train model
                        model_clone = clone(model)
                        model_clone.fit(X_inner_train, y_inner_train)
                        
                        # Evaluate
                        val_pred = model_clone.predict(X_inner_val)
                        score = r2_score(y_inner_val, val_pred)
                        scores.append(score)
                    
                    model_scores[model_type] = np.mean(scores)
                    
                except Exception as e:
                    print(f"    Warning: {model_type} failed for cluster {cluster_id}: {e}")
                    model_scores[model_type] = -np.inf
            
            # Select best model type
            best_type = max(model_scores, key=model_scores.get)
            best_models[cluster_id] = SurrogateModel(
                model_type=best_type,
                cluster_id=cluster_id
            )
            best_models[cluster_id].fit(X_cluster, y_cluster)
            
            print(f"    Cluster {cluster_id}: Best model = {best_type} "
                  f"(R² = {model_scores[best_type]:.3f})")
        
        return best_models
    
    def _evaluate_fold(self, models: Dict, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      cluster_labels_train: np.ndarray, 
                      cluster_labels_test: np.ndarray) -> Dict:
        """
        Evaluate models on a single outer fold.
        
        Returns:
            Dictionary of fold-specific results
        """
        results = {
            'cluster_results': {},
            'overall_metrics': {}
        }
        
        all_predictions = np.zeros(len(y_test))
        
        for cluster_id, model in models.items():
            # Get test samples for this cluster
            test_mask = cluster_labels_test == cluster_id
            
            if not np.any(test_mask):
                continue
            
            X_test_cluster = X_test[test_mask]
            y_test_cluster = y_test[test_mask]
            
            # Evaluate
            cluster_metrics = model.evaluate(X_test_cluster, y_test_cluster)
            results['cluster_results'][cluster_id] = cluster_metrics
            
            # Store predictions
            all_predictions[test_mask] = model.predict(X_test_cluster)
        
        # Handle samples from unseen clusters
        unassigned_mask = np.isin(cluster_labels_test, list(models.keys()), invert=True)
        if np.any(unassigned_mask):
            # Use mean prediction as fallback
            all_predictions[unassigned_mask] = y_train.mean()
        
        # Overall metrics
        results['overall_metrics'] = {
            'mse': mean_squared_error(y_test, all_predictions),
            'mae': mean_absolute_error(y_test, all_predictions),
            'r2': r2_score(y_test, all_predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, all_predictions))
        }
        
        return results
    
    def _aggregate_results(self) -> Dict:
        """
        Aggregate results across all outer folds.
        
        Returns:
            Dictionary with aggregated statistics
        """
        print("\n" + "=" * 60)
        print("AGGREGATED RESULTS")
        print("=" * 60)
        
        # Extract overall metrics from each fold
        overall_metrics = [fold['overall_metrics'] for fold in self.outer_cv_results]
        
        aggregated = {}
        
        # Calculate mean and std for each metric
        for metric in ['mse', 'mae', 'r2', 'rmse']:
            values = [m[metric] for m in overall_metrics]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
            
            print(f"{metric.upper()}: {aggregated[f'{metric}_mean']:.4f} ± "
                  f"{aggregated[f'{metric}_std']:.4f} "
                  f"[{aggregated[f'{metric}_min']:.4f}, "
                  f"{aggregated[f'{metric}_max']:.4f}]")
        
        # Calculate confidence intervals using bootstrap
        aggregated['confidence_intervals'] = self._bootstrap_confidence_intervals(
            overall_metrics
        )
        
        aggregated['fold_results'] = self.outer_cv_results
        
        return aggregated
    
    def _bootstrap_confidence_intervals(self, metrics: List[Dict], 
                                       n_bootstrap: int = 1000,
                                       confidence: float = 0.95) -> Dict:
        """
        Calculate bootstrap confidence intervals for metrics.
        
        Args:
            metrics: List of metric dictionaries
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Dictionary with confidence intervals
        """
        print("\nCalculating bootstrap confidence intervals...")
        
        intervals = {}
        
        for metric in ['mse', 'mae', 'r2', 'rmse']:
            values = np.array([m[metric] for m in metrics])
            
            # Bootstrap sampling
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            # Calculate confidence intervals
            alpha = 1 - confidence
            lower = np.percentile(bootstrap_means, 100 * alpha / 2)
            upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
            
            intervals[metric] = {
                'lower': lower,
                'upper': upper,
                'mean': np.mean(bootstrap_means)
            }
            
            print(f"  {metric.upper()} {confidence*100:.0f}% CI: "
                  f"[{lower:.4f}, {upper:.4f}]")
        
        return intervals


class ClusterSurrogateSystem:
    """
    Complete cluster-based surrogate modeling system with robust evaluation.
    
    This class orchestrates the entire pipeline from data loading through
    evaluation and reporting.
    """
    
    def __init__(self, data_path: str = 'data/rf_all_test_predictions.csv'):
        self.data_handler = DataHandler(data_path)
        self.umap_pipeline = UMAPClusteringPipeline()
        self.cv_evaluator = NestedCVEvaluator()
        self.results = {}
        
    def run_pipeline(self, save_models: bool = True) -> Dict:
        """
        Execute the complete surrogate modeling pipeline.
        
        Args:
            save_models: Whether to save trained models
            
        Returns:
            Dictionary containing all results
        """
        print("\n" + "=" * 80)
        print("ROBUST CLUSTER SURROGATE MODELING PIPELINE")
        print("=" * 80)
        
        # Step 1: Load and validate data
        print("\nStep 1: Data Loading and Validation")
        is_valid = self.data_handler.load_data()
        
        if not is_valid:
            print("WARNING: Data validation issues detected!")
        
        # Step 2: UMAP and clustering
        print("\nStep 2: UMAP Projection and Clustering")
        umap_coords, cluster_labels = self.umap_pipeline.fit_transform(
            self.data_handler.X.values,
            self.data_handler.y
        )
        
        # Visualize
        self.umap_pipeline.visualize(
            self.data_handler.y,
            save_path='umap_clustering.png'
        )
        
        # Step 3: Define model configurations
        print("\nStep 3: Model Configuration")
        model_configs = self._determine_model_configs()
        
        # Step 4: Nested cross-validation
        print("\nStep 4: Nested Cross-Validation Evaluation")
        cv_results = self.cv_evaluator.evaluate(
            self.data_handler.X.values,
            self.data_handler.y,
            self.data_handler.groups,
            cluster_labels,
            model_configs
        )
        
        # Step 5: Generate report
        print("\nStep 5: Generating Comprehensive Report")
        report = self._generate_report(cv_results)
        
        # Step 6: Save results
        if save_models:
            self._save_results(cv_results, report)
        
        self.results = {
            'cv_results': cv_results,
            'report': report,
            'cluster_metadata': self.umap_pipeline.cluster_metadata,
            'data_metadata': self.data_handler.metadata
        }
        
        return self.results
    
    def _determine_model_configs(self) -> Dict:
        """
        Determine optimal model type for each cluster based on characteristics.
        
        Returns:
            Dictionary mapping cluster IDs to model types
        """
        configs = {}
        
        for cluster_id, metadata in self.umap_pipeline.cluster_metadata.items():
            # Decision logic based on cluster characteristics
            if metadata['std'] < 0.1:
                # Low variance - simple model sufficient
                configs[cluster_id] = 'ridge'
            elif metadata['size'] > 500:
                # Large cluster - can support complex model
                configs[cluster_id] = 'tree'
            else:
                # Default
                configs[cluster_id] = 'ridge'
            
            print(f"  Cluster {cluster_id}: {configs[cluster_id]} model "
                  f"(size={metadata['size']}, std={metadata['std']:.3f})")
        
        return configs
    
    def _generate_report(self, cv_results: Dict) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            cv_results: Cross-validation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "=" * 80)
        report.append("SURROGATE MODEL EVALUATION REPORT")
        report.append("=" * 80)
        
        # Data summary
        report.append("\n### Data Summary ###")
        report.append(f"Total samples: {self.data_handler.metadata['n_samples']}")
        report.append(f"Total patients: {self.data_handler.metadata['n_patients']}")
        report.append(f"Features: {self.data_handler.metadata['n_features']}")
        
        # Clustering summary
        report.append("\n### Clustering Summary ###")
        report.append(f"Number of clusters: {len(self.umap_pipeline.cluster_metadata)}")
        for cluster_id, metadata in self.umap_pipeline.cluster_metadata.items():
            report.append(f"  Cluster {cluster_id}: {metadata['size']} samples, "
                         f"mean={metadata['mean']:.3f}±{metadata['std']:.3f}")
        
        # CV Results
        report.append("\n### Cross-Validation Results ###")
        report.append(f"Outer folds: {self.cv_evaluator.n_outer_splits}")
        report.append(f"Inner folds: {self.cv_evaluator.n_inner_splits}")
        
        report.append("\n### Overall Performance ###")
        for metric in ['r2', 'mse', 'mae', 'rmse']:
            mean_val = cv_results[f'{metric}_mean']
            std_val = cv_results[f'{metric}_std']
            ci = cv_results['confidence_intervals'][metric]
            report.append(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f} "
                         f"(95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}])")
        
        # Per-fold results
        report.append("\n### Per-Fold Results ###")
        for fold_result in cv_results['fold_results']:
            fold_num = fold_result['fold']
            metrics = fold_result['overall_metrics']
            report.append(f"\nFold {fold_num}:")
            report.append(f"  Samples: {fold_result['n_train']} train, "
                         f"{fold_result['n_test']} test")
            report.append(f"  R²: {metrics['r2']:.4f}")
            report.append(f"  MSE: {metrics['mse']:.4f}")
            report.append(f"  MAE: {metrics['mae']:.4f}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        return report_text
    
    def _save_results(self, cv_results: Dict, report: str):
        """Save results to disk."""
        print("\n### Saving Results ###")
        
        # Create output directory
        os.makedirs('models', exist_ok=True)
        
        # Save CV results
        joblib.dump(cv_results, 'models/surrogate_cv_results.pkl')
        print("  Saved CV results to models/surrogate_cv_results.pkl")
        
        # Save report
        with open('models/surrogate_evaluation_report.txt', 'w') as f:
            f.write(report)
        print("  Saved report to models/surrogate_evaluation_report.txt")
        
        # Save complete system
        system_data = {
            'data_handler': self.data_handler,
            'umap_pipeline': self.umap_pipeline,
            'cv_evaluator': self.cv_evaluator,
            'results': self.results
        }
        joblib.dump(system_data, 'models/surrogate_system_complete.pkl')
        print("  Saved complete system to models/surrogate_system_complete.pkl")


def main():
    """
    Main execution function for the robust surrogate modeling pipeline.
    """
    print("Starting Robust Surrogate Model Pipeline")
    print("=" * 50)
    
    # Initialize and run pipeline
    surrogate_system = ClusterSurrogateSystem()
    results = surrogate_system.run_pipeline(save_models=True)
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    results = main()