"""
Author: Richard Liu

Description: 

This module generates UMAP coordinates and clusters data points using 
DBSCAN. We use these data points to create specific datasets for our surrogate models
to provide interpretability and insight into our random forest. 

Key Features:
- UMAP coordinate generation to map data points before DBSCAN
- Data analysis and graphs for visualization
- Surrogate model training for selected clusters based on size
- Prediction analysis 

The script expects a CSV file containing the predictions of the random forest.
"""

# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import umap
from sklearn.cluster import DBSCAN, KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

"""
NOTE 

May not need the deploy functions as we are not deploying the model 
"""

"""
Polynomial Regressor
"""

poly_ridge_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=100, solver='lsqr'))
])

"""
This class loads data 
"""
class LoadData:
    def __init__():
        """
        Initialize variables
        """

    def load_data(file_path='data/rf_all_test_predictions.csv'):
        """
        This function loads and prepares the data for UMAP clustering.

        The function has the parameter file_path to determine where the 
        data is coming from for UMAP handling.
        """

        print("="*60)
        print("Loading and Preparing Data")
        print("="*60)
        
        # Load your completed predictions
        df_surrogate = pd.read_csv(file_path)
        
        # Extract components for probability-based surrogate training
        exclude_cols = ['true_label', 'predicted_label', 'prob_class_0', 'prob_class_1', 
                    'patient_id', 'chunk', 'fold', 'test_idx']
        feature_cols = [col for col in df_surrogate.columns if col not in exclude_cols]
        
        X_original = df_surrogate[feature_cols].values  # Original features (no probabilities)
        rf_probabilities = df_surrogate['prob_class_1'].values  # TARGET: Class 1 probabilities
        rf_hard_predictions = df_surrogate['predicted_label'].values  # For validation only
        y_original = df_surrogate['true_label'].values  # For validation only
        
        print(f"Data shape: {X_original.shape}")
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X_original)}")
        print(f"Target range: [{rf_probabilities.min():.3f}, {rf_probabilities.max():.3f}]")
        print(f"Target mean: {rf_probabilities.mean():.3f} ± {rf_probabilities.std():.3f}")
        
        return X_original, rf_probabilities, rf_hard_predictions, y_original, feature_cols

"""
This class is used for the generation of the UMAP coordinates for DBSCAN clustering. 
"""
class UMAPGeneration:
    def __init__(self):
        """
        Initialize variables
        """

        self.umap_model = None
        self.X_original = None
        self.rf_probabilities = None
        self.y_original = None

    def generate_umap(self, X_original, rf_probabilities, y_original):
        """
        This function generates the UMAP projection and takes 
        in the parameters self.X_original, self.rf_probabilites and 
        self.y_original
        """
        print("\n" + "="*60)
        print("PHASE 2.2: Generating UMAP Coordinates")
        print("="*60)
        
        # Apply UMAP to create 2D representation
        self.umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42,
            metric='euclidean'
        )
        
        print("Fitting UMAP...")
        self.umap_coords = self.umap_model.fit_transform(self.X_original)
        print("UMAP fitting completed")

        return self.umap_coords

    def visualize_umap(self):
        """
        This function visualizes the UMAP projection using 3 plots generated
        through matplotlib. 

        Plot 1 displays the mapping of the original data set (cancer stages)
        Plot 2 displays the predictions of the random forest by probabilities
        Plot 3 displays a probability histogram to graphically view the distribution
        """
        
        # Visualize probability distribution in UMAP space
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Original cancer stages
        plt.subplot(1, 3, 1)
        scatter1 = plt.scatter(self.umap_coords[:, 0], self.umap_coords[:, 1], 
                            c=self.y_original, cmap='RdYlBu', alpha=0.6, s=10)
        plt.colorbar(scatter1, label='Original Cancer Stage')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('Original Cancer Stages')
        
        # Plot 2: RF probability predictions (our target)
        plt.subplot(1, 3, 2)
        scatter2 = plt.scatter(self.umap_coords[:, 0], self.umap_coords[:, 1], 
                            c=self.rf_probabilities, cmap='RdYlBu', alpha=0.6, s=10)
        plt.colorbar(scatter2, label='RF Class 1 Probability')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('RF Probability Predictions (Target)')
        
        # Plot 3: Probability histogram
        plt.subplot(1, 3, 3)
        plt.hist(self.rf_probabilities, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('RF Class 1 Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Target Probabilities')
        plt.axvline(self.rf_probabilities.mean(), color='red', linestyle='--', 
                    label=f'Mean: {self.rf_probabilities.mean():.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return self.umap_model, self.umap_coords

def apply_clustering(umap_coords, rf_probabilities):
    """
    This function applies DBSCAN clustering to the UMAP projection.

    We find unique clusters through the data points and categorize outliers in
    another cluster. 
    """

    print("\n" + "="*60)
    print("PHASE 2.3: Applying DBSCAN Clustering")
    print("="*60)
    
    # Apply clustering to UMAP coordinates
    clustering = DBSCAN(eps=0.8, min_samples=50)
    cluster_labels = clustering.fit_predict(umap_coords)
    
    print("DBSCAN Results:")
    unique_clusters = np.unique(cluster_labels)
    print(f"Clusters found: {unique_clusters}")
    
    valid_clusters = []
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            outliers = np.sum(cluster_labels == cluster_id)
            print(f"  Outliers (cluster -1): {outliers} samples")
            continue
        
        cluster_size = np.sum(cluster_labels == cluster_id)
        avg_prob = rf_probabilities[cluster_labels == cluster_id].mean()
        prob_std = rf_probabilities[cluster_labels == cluster_id].std()
        
        if cluster_size >= 50:  # Only keep valid clusters
            valid_clusters.append(cluster_id)
            print(f"  Cluster {cluster_id}: {cluster_size} samples, "
                  f"avg_prob={avg_prob:.3f}±{prob_std:.3f}")

        return 

    def clustering_results(self):
        # Visualize clusters with probability information
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Cluster assignments
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Cluster ID')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('Cluster Assignments')
        
        # Plot 2: Probability by cluster
        plt.subplot(1, 3, 2)
        for cluster_id in valid_clusters:
            mask = cluster_labels == cluster_id
            plt.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                    c=rf_probabilities[mask], cmap='RdYlBu', 
                    alpha=0.6, s=10, label=f'Cluster {cluster_id}')
        plt.colorbar(label='RF Probability')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('Probabilities within Clusters')
        
        # Plot 3: Probability distributions by cluster
        plt.subplot(1, 3, 3)
        for cluster_id in valid_clusters[:6]:  # Limit to first 6 for readability
            mask = cluster_labels == cluster_id
            plt.hist(rf_probabilities[mask], bins=20, alpha=0.5, 
                    label=f'Cluster {cluster_id}', density=True)
        plt.xlabel('RF Class 1 Probability')
        plt.ylabel('Density')
        plt.title('Probability Distributions by Cluster')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return clustering, cluster_labels, valid_clusters

class ClusterAnalysis:
    def __init__(self):
        self.cluster_analysis = {}
        self.prob_stats = {}

    def analyze_probability_clusters(self, cluster_labels, rf_probabilities, y_original, valid_clusters):
        """Step 3.1: Analyze Cluster Probability Characteristics"""
        print("\n" + "="*60)
        print("PHASE 3.1: Analyzing Cluster Probability Characteristics")
        print("="*60)
        
        for cluster_id in valid_clusters:
            mask = cluster_labels == cluster_id
            cluster_size = np.sum(mask)
            
            # Analyze probability distribution
            cluster_probs = rf_probabilities[mask]
            cluster_original = y_original[mask]
            
            self.prob_stats = {
                'mean': cluster_probs.mean(),
                'std': cluster_probs.std(),
                'min': cluster_probs.min(),
                'max': cluster_probs.max(),
                'q25': np.percentile(cluster_probs, 25),
                'q75': np.percentile(cluster_probs, 75)
            }
            
            # Classify cluster type based on probability distribution
            if self.prob_stats['mean'] > 0.8:
                cluster_type = 'high_confidence_positive'
            elif self.prob_stats['mean'] < 0.2:
                cluster_type = 'high_confidence_negative'  
            elif self.prob_stats['std'] < 0.1:
                cluster_type = 'homogeneous_uncertain'
            else:
                cluster_type = 'mixed_heterogeneous'
            
            self.cluster_analysis[cluster_id] = {
                'size': cluster_size,
                'prob_stats': self.prob_stats,
                'cluster_type': cluster_type,
                'original_stages': np.bincount(cluster_original.astype(int))
            }
            
            print(f"\nCluster {cluster_id} ({cluster_type}):")
            print(f"  Size: {cluster_size}")
            print(f"  RF Probability: {self.prob_stats['mean']:.3f} ± {self.prob_stats['std']:.3f}")
            print(f"  Range: [{self.prob_stats['min']:.3f}, {self.prob_stats['max']:.3f}]")
            print(f"  IQR: [{self.prob_stats['q25']:.3f}, {self.prob_stats['q75']:.3f}]")
            print(f"  Original stages distribution: {np.bincount(cluster_original.astype(int))}")
        
        return self.cluster_analysis

def select_probability_surrogate_models(cluster_analysis, valid_clusters):
    """Step 3.2: Select Regression Models for Probability Prediction with Merged Clusters"""
    print("\n" + "="*60)
    print("PHASE 3.2: Selecting Surrogate Models (Modified)")
    print("="*60)
    
    surrogate_models = {}
    training_clusters = []  # Clusters that will have surrogate models trained
    qualitative_clusters = []  # Clusters excluded from training
    merged_clusters = []  # Clusters that will be merged
    
    for cluster_id in valid_clusters:
        cluster_info = cluster_analysis[cluster_id]
        
        # Exclude cluster 3 from training (save for qualitative analysis)
        if cluster_id == 3:
            qualitative_clusters.append(cluster_id)
            print(f"Cluster {cluster_id} excluded from surrogate training - saved for qualitative analysis")
            continue
        
        # Merge clusters 1 and 2 for combined Ridge regression
        if cluster_id in [1, 2]:
            merged_clusters.append(cluster_id)
            continue
        
        # Regular processing for other clusters
        size = cluster_info['size']
        cluster_type = cluster_info['cluster_type']
        
        if cluster_type == 'mixed_heterogeneous' and size > 200:
            surrogate_models[cluster_id] = DecisionTreeRegressor(
                max_depth=4, 
                min_samples_leaf=20, 
                random_state=42
            )
            training_clusters.append(cluster_id)
        else:
            # Default model for other clusters
            surrogate_models[cluster_id] = poly_ridge_model
            training_clusters.append(cluster_id)
        
        model_name = type(surrogate_models[cluster_id]).__name__
        cluster_type = cluster_analysis[cluster_id]['cluster_type']
        avg_prob = cluster_analysis[cluster_id]['prob_stats']['mean']
        size = cluster_analysis[cluster_id]['size']
        
        print(f"Cluster {cluster_id} ({cluster_type}, n={size}, avg_prob={avg_prob:.3f}): {model_name}")
    
    # Handle merged clusters (1 and 2)
    if len(merged_clusters) > 0:
        merged_cluster_key = 'merged_1_2'  # Use string key for merged clusters
        surrogate_models[merged_cluster_key] = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=100, solver='lsqr'))
])
        
        training_clusters.append(merged_cluster_key)
        
        # Calculate combined stats for merged clusters
        combined_size = sum(cluster_analysis[cid]['size'] for cid in merged_clusters)
        combined_avg_prob = np.average(
            [cluster_analysis[cid]['prob_stats']['mean'] for cid in merged_clusters],
            weights=[cluster_analysis[cid]['size'] for cid in merged_clusters]
        )
        
        print(f"Merged clusters {merged_clusters} -> {merged_cluster_key} (high_confidence_negative, "
              f"n={combined_size}, avg_prob={combined_avg_prob:.3f}): Ridge")
    
    return surrogate_models, training_clusters, qualitative_clusters, merged_clusters

# =============================================================================
# PHASE 4: TRAIN PROBABILITY-BASED SURROGATES
# =============================================================================

def train_probability_surrogates(X_original, rf_probabilities, cluster_labels, 
                               training_clusters, surrogate_models, cluster_analysis, 
                               merged_clusters):
    """Step 4.1: Train Individual Probability Surrogate Models with Merged Clusters"""
    print("\n" + "="*60)
    print("PHASE 4: Training Probability-Based Surrogates (Modified)")
    print("="*60)
    
    trained_surrogates = {}
    surrogate_performance = {}
    
    for cluster_key in training_clusters:
        print(f"\n{'='*40}")
        print(f"Training Surrogate for {cluster_key}")
        print(f"{'='*40}")
        
        # Handle merged clusters
        if cluster_key == 'merged_1_2':
            # Combine data from clusters 1 and 2
            cluster_mask = np.zeros(len(cluster_labels), dtype=bool)
            for cid in merged_clusters:
                cluster_mask |= (cluster_labels == cid)
            
            # Calculate combined cluster statistics
            combined_size = np.sum(cluster_mask)
            cluster_probs = rf_probabilities[cluster_mask]
            
            cluster_stats = {
                'size': combined_size,
                'prob_stats': {
                    'mean': cluster_probs.mean(),
                    'std': cluster_probs.std(),
                    'min': cluster_probs.min(),
                    'max': cluster_probs.max()
                },
                'cluster_type': 'high_confidence_negative_merged'
            }
        else:
            # Regular single cluster
            cluster_mask = cluster_labels == cluster_key
            cluster_stats = cluster_analysis[cluster_key]
        
        X_cluster = X_original[cluster_mask]
        y_cluster_prob = rf_probabilities[cluster_mask]
        
        # Train regression model
        model = surrogate_models[cluster_key]
        model.fit(X_cluster, y_cluster_prob)
        trained_surrogates[cluster_key] = model
        
        # Evaluate probability prediction quality
        prob_predictions = model.predict(X_cluster)
        prob_predictions = np.clip(prob_predictions, 0.0, 1.0)
        
        # Regression metrics
        mse = mean_squared_error(y_cluster_prob, prob_predictions)
        mae = mean_absolute_error(y_cluster_prob, prob_predictions)
        r2 = r2_score(y_cluster_prob, prob_predictions)
        
        # Convert to hard predictions for classification metrics
        hard_targets = (y_cluster_prob > 0.5).astype(int)
        hard_predictions = (prob_predictions > 0.5).astype(int)
        classification_accuracy = (hard_targets == hard_predictions).mean()
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_cluster, y_cluster_prob, 
                                       cv=min(5, len(X_cluster)//10), 
                                       scoring='neg_mean_squared_error')
            cv_mse_mean = -cv_scores.mean()
            cv_mse_std = cv_scores.std()
        except:
            cv_mse_mean = mse
            cv_mse_std = 0.0
        
        surrogate_performance[cluster_key] = {
            'cluster_size': len(X_cluster),
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'classification_accuracy': classification_accuracy,
            'cv_mse': cv_mse_mean,
            'cv_mse_std': cv_mse_std,
            'model_type': type(model).__name__,
            'cluster_type': cluster_stats['cluster_type'],
            'target_prob_range': [y_cluster_prob.min(), y_cluster_prob.max()]
        }
        
        print(f"  Cluster size: {len(X_cluster)}")
        print(f"  Target prob range: [{y_cluster_prob.min():.3f}, {y_cluster_prob.max():.3f}]")
        print(f"  Regression MSE: {mse:.4f}")
        print(f"  Regression MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Classification Accuracy: {classification_accuracy:.4f}")
        print(f"  CV MSE: {cv_mse_mean:.4f} ± {cv_mse_std:.4f}")
        print(f"  Model: {type(model).__name__}")
    
    return trained_surrogates, surrogate_performance

# =============================================================================
# PHASE 5: INTEGRATION & ROUTING SYSTEM
# =============================================================================

class ClusterRouter:
    """Step 5.1: Modified Cluster Router"""
    def __init__(self, umap_model, clustering_model, training_clusters, qualitative_clusters, 
                 merged_clusters, umap_coords, cluster_labels):
        self.umap_model = umap_model
        self.clustering_model = clustering_model
        self.training_clusters = training_clusters
        self.qualitative_clusters = qualitative_clusters
        self.merged_clusters = merged_clusters
        self.default_cluster = training_clusters[0] if training_clusters else 0
        
        # Pre-compute cluster centers for DBSCAN
        self.cluster_centers_ = {}
        
        # Centers for individual training clusters (excluding merged)
        for cluster_key in training_clusters:
            if cluster_key != 'merged_1_2':
                mask = cluster_labels == cluster_key
                if np.any(mask):
                    self.cluster_centers_[cluster_key] = umap_coords[mask].mean(axis=0)
        
        # Center for merged clusters
        if 'merged_1_2' in training_clusters:
            merged_mask = np.zeros(len(cluster_labels), dtype=bool)
            for cid in merged_clusters:
                merged_mask |= (cluster_labels == cid)
            if np.any(merged_mask):
                self.cluster_centers_['merged_1_2'] = umap_coords[merged_mask].mean(axis=0)
        
        # Centers for qualitative clusters (for routing, but no surrogate)
        for cluster_id in qualitative_clusters:
            mask = cluster_labels == cluster_id
            if np.any(mask):
                self.cluster_centers_[cluster_id] = umap_coords[mask].mean(axis=0)
    
    def assign_cluster(self, X):
        """Assign new samples to clusters with merged cluster handling"""
        umap_coords_new = self.umap_model.transform(X)
        
        predicted_clusters = []
        for coord in umap_coords_new:
            distances = {}
            
            # Calculate distances to all cluster centers
            for cluster_key, center in self.cluster_centers_.items():
                distances[cluster_key] = np.linalg.norm(coord - center)
            
            # Find nearest cluster
            nearest_cluster = min(distances, key=distances.get)
            
            # Map original clusters 1,2 to merged cluster key
            if nearest_cluster in self.merged_clusters:
                nearest_cluster = 'merged_1_2'
            
            predicted_clusters.append(nearest_cluster)
        
        return np.array(predicted_clusters)

class ClusterAwareProbabilitySurrogate:
    """Step 5.2: Modified Probability-Based Surrogate System"""
    def __init__(self, cluster_router, trained_surrogates, feature_cols, qualitative_clusters):
        self.router = cluster_router
        self.surrogates = trained_surrogates
        self.feature_cols = feature_cols
        self.qualitative_clusters = qualitative_clusters
        
    def predict_proba(self, X):
        """Predict Class 1 probabilities using cluster-specific surrogates"""
        cluster_assignments = self.router.assign_cluster(X)
        probabilities = np.zeros(len(X))
        
        for i, cluster_assignment in enumerate(cluster_assignments):
            if cluster_assignment in self.surrogates:
                # Use trained surrogate
                cluster_prob = self.surrogates[cluster_assignment].predict(X[i:i+1])
                probabilities[i] = np.clip(cluster_prob[0], 0.0, 1.0)
            elif cluster_assignment in self.qualitative_clusters:
                # Qualitative cluster - return neutral probability or flag
                probabilities[i] = 0.5  # Neutral prediction for qualitative analysis samples
            else:
                # Fallback to default model
                if self.router.default_cluster in self.surrogates:
                    cluster_prob = self.surrogates[self.router.default_cluster].predict(X[i:i+1])
                    probabilities[i] = np.clip(cluster_prob[0], 0.0, 1.0)
                else:
                    probabilities[i] = 0.5
        
        # Return in sklearn format: [prob_class_0, prob_class_1]
        prob_class_0 = 1 - probabilities
        return np.column_stack([prob_class_0, probabilities])
    
    def predict(self, X):
        """Get hard predictions from probabilities"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def explain_probability_prediction(self, X, sample_idx=0):
        """Provide cluster-specific probability explanation"""
        cluster_assignment = self.router.assign_cluster(X[sample_idx:sample_idx+1])[0]
        
        explanation = {
            'sample_index': sample_idx,
            'cluster_assignment': str(cluster_assignment),
            'is_qualitative_cluster': cluster_assignment in self.qualitative_clusters,
        }
        
        if cluster_assignment in self.surrogates:
            model = self.surrogates[cluster_assignment]
            prob_prediction = self.predict_proba(X[sample_idx:sample_idx+1])[0, 1]
            
            explanation.update({
                'model_type': type(model).__name__,
                'probability_class_1': prob_prediction,
                'hard_prediction': int(prob_prediction > 0.5),
            })
            
            # Add model-specific interpretations
            if hasattr(model, 'coef_'):
                coeffs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                sample_features = X[sample_idx]
                contributions = coeffs * sample_features
                feature_contributions = list(zip(self.feature_cols, contributions))
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                explanation['probability_contributions'] = feature_contributions[:10]
                
        else:
            explanation.update({
                'model_type': 'None (Qualitative Analysis)',
                'probability_class_1': 0.5,
                'hard_prediction': 'N/A',
            })
        
        return explanation

# =============================================================================
# PHASE 6: VALIDATION & EVALUATION
# =============================================================================

def validate_probability_system(integrated_surrogate, X_original, rf_probabilities, 
                               cluster_labels, training_clusters, qualitative_clusters):
    """Step 6.1: Test Modified Probability Prediction System"""
    print("\n" + "="*60)
    print("PHASE 6: Validation & Evaluation (Modified)")
    print("="*60)
    
    # Create mask for training samples only (exclude qualitative clusters)
    training_mask = np.zeros(len(cluster_labels), dtype=bool)
    for cluster_key in training_clusters:
        if cluster_key == 'merged_1_2':
            # Include samples from merged clusters 1 and 2
            training_mask |= (cluster_labels == 1) | (cluster_labels == 2)
        else:
            training_mask |= (cluster_labels == cluster_key)
    
    # Test on training samples only
    X_training = X_original[training_mask]
    rf_probs_training = rf_probabilities[training_mask]
    
    # Test probability predictions
    surrogate_probs = integrated_surrogate.predict_proba(X_training)
    surrogate_class1_probs = surrogate_probs[:, 1]
    
    # Probability prediction quality
    prob_mse = mean_squared_error(rf_probs_training, surrogate_class1_probs)
    prob_mae = mean_absolute_error(rf_probs_training, surrogate_class1_probs)
    prob_r2 = r2_score(rf_probs_training, surrogate_class1_probs)
    
    print(f"=== MODIFIED SURROGATE SYSTEM EVALUATION ===")
    print(f"Evaluated on {len(X_training)} training samples (excluding qualitative clusters)")
    print(f"Overall Probability Prediction Quality:")
    print(f"  MSE: {prob_mse:.4f}")
    print(f"  MAE: {prob_mae:.4f}")
    print(f"  R² Score: {prob_r2:.4f}")
    
    # Hard prediction fidelity 
    surrogate_hard = integrated_surrogate.predict(X_training)
    rf_hard = (rf_probs_training > 0.5).astype(int)
    classification_fidelity = (surrogate_hard == rf_hard).mean()
    
    print(f"\nHard Prediction Fidelity: {classification_fidelity:.4f}")
    
    # Report on qualitative clusters
    qualitative_mask = np.zeros(len(cluster_labels), dtype=bool)
    for cluster_id in qualitative_clusters:
        qualitative_mask |= (cluster_labels == cluster_id)
    
    if np.any(qualitative_mask):
        n_qualitative = np.sum(qualitative_mask)
        print(f"\nQualitative Analysis Samples: {n_qualitative}")
        print(f"  These samples are excluded from surrogate training")
        print(f"  Available for qualitative analysis in cluster(s): {qualitative_clusters}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Actual vs Predicted Probabilities (training samples only)
    plt.subplot(1, 3, 1)
    plt.scatter(rf_probs_training, surrogate_class1_probs, alpha=0.5, s=10)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    plt.xlabel('RF Probability (Actual)')
    plt.ylabel('Surrogate Probability (Predicted)')
    plt.title(f'Probability Prediction Quality\nR² = {prob_r2:.3f}')
    plt.legend()
    
    # Plot 2: Residuals
    plt.subplot(1, 3, 2)
    residuals = surrogate_class1_probs - rf_probs_training
    plt.scatter(rf_probs_training, residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('RF Probability (Actual)')
    plt.ylabel('Residuals (Predicted - Actual)')
    plt.title(f'Residual Analysis\nMAE = {prob_mae:.3f}')
    
    # Plot 3: Distribution comparison
    plt.subplot(1, 3, 3)
    plt.hist(rf_probs_training, bins=30, alpha=0.5, label='RF Probabilities', density=True)
    plt.hist(surrogate_class1_probs, bins=30, alpha=0.5, label='Surrogate Probabilities', density=True)
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title('Probability Distributions (Training Samples)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'overall_mse': prob_mse,
        'overall_mae': prob_mae,
        'overall_r2': prob_r2,
        'classification_fidelity': classification_fidelity,
        'n_training_samples': len(X_training),
        'n_qualitative_samples': np.sum(qualitative_mask)
    }

def generate_probability_interpretation_report(integrated_surrogate, cluster_analysis, 
                                            surrogate_performance, training_clusters, 
                                            qualitative_clusters, merged_clusters, feature_cols):
    """Step 6.2: Modified Probability-Based Interpretation Report"""
    print("\n" + "="*60)
    print("PHASE 6.2: Generating Interpretation Report (Modified)")
    print("="*60)
    
    interpretation_report = {}
    
    for cluster_key in training_clusters:
        model = integrated_surrogate.surrogates[cluster_key]
        performance = surrogate_performance[cluster_key]
        
        # Handle merged clusters
        if cluster_key == 'merged_1_2':
            # Combined stats for merged clusters
            combined_size = sum(cluster_analysis[cid]['size'] for cid in merged_clusters)
            combined_prob_stats = {
                'mean': np.average([cluster_analysis[cid]['prob_stats']['mean'] for cid in merged_clusters],
                                 weights=[cluster_analysis[cid]['size'] for cid in merged_clusters]),
                'min': min(cluster_analysis[cid]['prob_stats']['min'] for cid in merged_clusters),
                'max': max(cluster_analysis[cid]['prob_stats']['max'] for cid in merged_clusters),
                'std': np.sqrt(np.average([cluster_analysis[cid]['prob_stats']['std']**2 for cid in merged_clusters],
                                        weights=[cluster_analysis[cid]['size'] for cid in merged_clusters]))
            }
            
            report = {
                'cluster_key': cluster_key,
                'original_clusters': merged_clusters,
                'cluster_size': combined_size,
                'cluster_type': 'high_confidence_negative_merged',
                'rf_probability_stats': combined_prob_stats,
                'model_type': type(model).__name__,
                'probability_performance': {
                    'mse': performance['mse'],
                    'r2': performance['r2_score'],
                    'classification_accuracy': performance['classification_accuracy']
                }
            }
        else:
            # Regular single cluster
            cluster_info = cluster_analysis[cluster_key]
            report = {
                'cluster_key': cluster_key,
                'cluster_size': cluster_info['size'],
                'cluster_type': cluster_info['cluster_type'],
                'rf_probability_stats': cluster_info['prob_stats'],
                'model_type': type(model).__name__,
                'probability_performance': {
                    'mse': performance['mse'],
                    'r2': performance['r2_score'],
                    'classification_accuracy': performance['classification_accuracy']
                }
            }
        
        # Extract probability-focused interpretations
        if hasattr(model, 'coef_'):
            coeffs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            top_features = sorted(zip(feature_cols, coeffs), 
                                key=lambda x: abs(x[1]), reverse=True)[:10]
            
            report['decision_strategy'] = "Linear probability model"
            report['probability_drivers'] = [
                {
                    'feature': feat, 
                    'coefficient': coef,
                    'interpretation': f"Each unit ↑ changes P(Class=1) by {coef:+.3f}"
                }
                for feat, coef in top_features[:5]
            ]
            
        elif hasattr(model, 'feature_importances_'):
            top_features = sorted(zip(feature_cols, model.feature_importances_), 
                                key=lambda x: x[1], reverse=True)[:10]
            report['decision_strategy'] = "Tree-based probability model"
            report['probability_drivers'] = [
                {
                    'feature': feat,
                    'importance': imp,
                    'interpretation': "Key factor in probability prediction"
                }
                for feat, imp in top_features[:5]
            ]
        
        interpretation_report[cluster_key] = report
    
    # Add qualitative clusters info
    for cluster_id in qualitative_clusters:
        cluster_info = cluster_analysis[cluster_id]
        interpretation_report[f"qualitative_{cluster_id}"] = {
            'cluster_key': cluster_id,
            'cluster_size': cluster_info['size'],
            'cluster_type': cluster_info['cluster_type'],
            'rf_probability_stats': cluster_info['prob_stats'],
            'model_type': 'None (Qualitative Analysis)',
            'status': 'Excluded from surrogate training'
        }
    
    # Print summary
    print("\n=== MODIFIED PROBABILITY INTERPRETATION SUMMARY ===")
    
    # Training clusters
    print("\nTRAINED SURROGATE MODELS:")
    for cluster_key, report in interpretation_report.items():
        if not cluster_key.startswith('qualitative_'):
            prob_stats = report['rf_probability_stats']
            print(f"\n{cluster_key}:")
            if cluster_key == 'merged_1_2':
                print(f"  Original clusters: {report['original_clusters']}")
            print(f"  Size: {report['cluster_size']} samples")
            print(f"  RF Prob Range: [{prob_stats['min']:.3f}, {prob_stats['max']:.3f}]")
            print(f"  RF Prob Mean: {prob_stats['mean']:.3f} ± {prob_stats['std']:.3f}")
            print(f"  Model: {report['model_type']}")
            print(f"  Surrogate R²: {report['probability_performance']['r2']:.3f}")
            print(f"  Strategy: {report.get('decision_strategy', 'Custom model')}")
            
            if 'probability_drivers' in report:
                print(f"  Top probability drivers:")
                for driver in report['probability_drivers'][:3]:
                    print(f"    • {driver['feature']}: {driver['interpretation']}")
    
    # Qualitative clusters
    if qualitative_clusters:
        print(f"\nQUALITATIVE ANALYSIS CLUSTERS:")
        for cluster_key, report in interpretation_report.items():
            if cluster_key.startswith('qualitative_'):
                prob_stats = report['rf_probability_stats']
                actual_cluster_id = report['cluster_key']
                print(f"\nCluster {actual_cluster_id} (qualitative):")
                print(f"  Size: {report['cluster_size']} samples")
                print(f"  RF Prob Range: [{prob_stats['min']:.3f}, {prob_stats['max']:.3f}]")
                print(f"  RF Prob Mean: {prob_stats['mean']:.3f} ± {prob_stats['std']:.3f}")
                print(f"  Status: {report['status']}")
                print(f"  Note: Saved for manual qualitative analysis")
    
    return interpretation_report

# =============================================================================
# PHASE 7: SAVE & DEPLOY
# =============================================================================

def save_probability_system(integrated_surrogate, interpretation_report, cluster_analysis,
                          surrogate_performance, feature_cols, training_clusters, 
                          qualitative_clusters, merged_clusters, validation_results):
    """Step 7.1: Save Complete Modified Probability-Based System"""
    print("\n" + "="*60)
    print("PHASE 7.1: Saving Complete System (Modified)")
    print("="*60)
    
    # Save all components including modification info
    probability_system_components = {
        'integrated_surrogate': integrated_surrogate,
        'interpretation_report': interpretation_report,
        'cluster_analysis': cluster_analysis,
        'surrogate_performance': surrogate_performance,
        'feature_columns': feature_cols,
        'training_clusters': training_clusters,
        'qualitative_clusters': qualitative_clusters,
        'merged_clusters': merged_clusters,
        'validation_results': validation_results,
        'target_type': 'probability',
        'modifications': {
            'merged_clusters_1_2': True,
            'excluded_cluster_3': True,
            'qualitative_analysis_available': len(qualitative_clusters) > 0
        }
    }
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(probability_system_components, 'models/modified_probability_cluster_surrogate_system.pkl')
    print("Complete modified surrogate system saved to 'models/modified_probability_cluster_surrogate_system.pkl'!")
    
    return probability_system_components

def create_probability_interface(system_components):
    """Step 7.2: Create Modified Probability-Focused Usage Interface"""
    print("\n" + "="*60)
    print("PHASE 7.2: Creating Usage Interface (Modified)")
    print("="*60)
    
    def explain_rf_probability_decision(sample, system_components, original_rf_model=None):
        """Easy interface for explaining RF probability decisions"""
        
        surrogate = system_components['integrated_surrogate']
        interpretations = system_components['interpretation_report']
        qualitative_clusters = system_components['qualitative_clusters']
        
        # Get predictions
        if original_rf_model is not None:
            rf_prob = original_rf_model.predict_proba(sample.reshape(1, -1))[0, 1]
            rf_hard = int(rf_prob > 0.5)
        else:
            rf_prob = None
            rf_hard = None
        
        # Get cluster assignment and prediction
        explanation = surrogate.explain_probability_prediction(sample.reshape(1, -1))
        cluster_assignment = explanation['cluster_assignment']
        
        print(f"=== MODIFIED RF PROBABILITY DECISION EXPLANATION ===")
        
        if explanation['is_qualitative_cluster']:
            print(f"  Sample assigned to QUALITATIVE ANALYSIS cluster: {cluster_assignment}")
            print(f"   This sample is excluded from surrogate predictions")
            print(f"   Available for manual qualitative analysis")
            if rf_prob is not None:
                print(f"   Original RF Probability: {rf_prob:.3f}")
            return explanation
        
        # Regular surrogate prediction
        surrogate_prob = explanation['probability_class_1']
        surrogate_hard = explanation['hard_prediction']
        
        if rf_prob is not None:
            print(f"RF Class 1 Probability: {rf_prob:.3f}")
            print(f"RF Hard Prediction: {rf_hard}")
            print(f"Surrogate Probability: {surrogate_prob:.3f}")
            print(f"Surrogate Hard Prediction: {surrogate_hard}")
            print(f"Probability Error: {abs(rf_prob - surrogate_prob):.3f}")
            print(f"Agreement: {'✓' if rf_hard == surrogate_hard else '✗'}")
        else:
            print(f"Surrogate Probability: {surrogate_prob:.3f}")
            print(f"Surrogate Hard Prediction: {surrogate_hard}")
        
        print(f"\nPatient falls in: {cluster_assignment}")
        
        # Get cluster interpretation
        cluster_interpretation = None
        for key, report in interpretations.items():
            if (key == cluster_assignment or 
                (cluster_assignment == 'merged_1_2' and key == 'merged_1_2')):
                cluster_interpretation = report
                break
        
        if cluster_interpretation:
            print(f"Cluster type: {cluster_interpretation['cluster_type']}")
            print(f"Typical RF probability in this region: {cluster_interpretation['rf_probability_stats']['mean']:.3f}")
            print(f"Decision model: {cluster_interpretation['model_type']}")
            
            if 'probability_drivers' in cluster_interpretation:
                print(f"\nKey probability drivers in this cluster:")
                for driver in cluster_interpretation['probability_drivers'][:5]:
                    print(f"  • {driver['feature']}: {driver['interpretation']}")
        
        return explanation
    
    def batch_explain_probability_decisions(X, system_components, n_examples=5):
        """Explain multiple probability decisions with qualitative handling"""
        surrogate = system_components['integrated_surrogate']
        qualitative_clusters = system_components['qualitative_clusters']
        
        print(f"=== BATCH PROBABILITY EXPLANATIONS (showing {n_examples} examples) ===")
        
        # Get predictions for all samples
        cluster_assignments = surrogate.router.assign_cluster(X)
        
        # Select diverse examples including qualitative samples if available
        indices = np.linspace(0, len(X)-1, n_examples, dtype=int)
        
        # Try to include qualitative samples if they exist
        qualitative_indices = []
        for i, assignment in enumerate(cluster_assignments):
            if assignment in qualitative_clusters:
                qualitative_indices.append(i)
        
        if qualitative_indices and n_examples > 3:
            # Replace one regular example with qualitative
            indices[-1] = qualitative_indices[0]
        
        for i, idx in enumerate(indices):
            print(f"\n--- Example {i+1}: Sample {idx} ---")
            explain_rf_probability_decision(X[idx], system_components)
    
    def generate_cluster_summary_report(system_components):
        """Generate comprehensive cluster summary with modifications"""
        interpretations = system_components['interpretation_report']
        training_clusters = system_components['training_clusters']
        qualitative_clusters = system_components['qualitative_clusters']
        merged_clusters = system_components['merged_clusters']
        
        print(f"=== MODIFIED CLUSTER SUMMARY REPORT ===")
        
        # Count trained vs qualitative
        n_training_clusters = len(training_clusters)
        n_qualitative_clusters = len(qualitative_clusters)
        
        print(f"Training clusters: {n_training_clusters}")
        print(f"Qualitative clusters: {n_qualitative_clusters}")
        print(f"Merged clusters: {merged_clusters} -> 'merged_1_2'")
        
        # Summary statistics for training clusters
        training_reports = [report for key, report in interpretations.items() 
                          if not key.startswith('qualitative_')]
        
        if training_reports:
            total_training_samples = sum(report['cluster_size'] for report in training_reports)
            avg_r2 = np.mean([report['probability_performance']['r2'] 
                             for report in training_reports])
            
            print(f"Total training samples: {total_training_samples}")
            print(f"Average R² across training clusters: {avg_r2:.3f}")
        
        # Qualitative statistics
        qualitative_reports = [report for key, report in interpretations.items() 
                             if key.startswith('qualitative_')]
        
        if qualitative_reports:
            total_qualitative_samples = sum(report['cluster_size'] for report in qualitative_reports)
            print(f"Total qualitative samples: {total_qualitative_samples}")
        
        # Model type distribution for training clusters
        if training_reports:
            model_types = [report['model_type'] for report in training_reports]
            model_counts = {mt: model_types.count(mt) for mt in set(model_types)}
            
            print(f"\nModel type distribution (training):")
            for model_type, count in model_counts.items():
                print(f"  {model_type}: {count} clusters")
        
        # Performance by cluster type
        cluster_types = {}
        for report in training_reports:
            cluster_type = report['cluster_type']
            if cluster_type not in cluster_types:
                cluster_types[cluster_type] = []
            cluster_types[cluster_type].append(report['probability_performance']['r2'])
        
        if cluster_types:
            print(f"\nPerformance by cluster type:")
            for cluster_type, r2_scores in cluster_types.items():
                avg_r2 = np.mean(r2_scores)
                std_r2 = np.std(r2_scores) if len(r2_scores) > 1 else 0
                print(f"  {cluster_type}: R² = {avg_r2:.3f} (±{std_r2:.3f})")
    
    def get_qualitative_samples(X, system_components):
        """Extract samples assigned to qualitative clusters"""
        surrogate = system_components['integrated_surrogate']
        qualitative_clusters = system_components['qualitative_clusters']
        
        cluster_assignments = surrogate.router.assign_cluster(X)
        qualitative_mask = np.array([assignment in qualitative_clusters 
                                   for assignment in cluster_assignments])
        
        if np.any(qualitative_mask):
            qualitative_samples = X[qualitative_mask]
            qualitative_indices = np.where(qualitative_mask)[0]
            
            print(f"Found {len(qualitative_samples)} samples for qualitative analysis")
            print(f"Sample indices: {qualitative_indices[:10]}..." if len(qualitative_indices) > 10 
                  else f"Sample indices: {qualitative_indices}")
            
            return qualitative_samples, qualitative_indices
        else:
            print("No samples assigned to qualitative clusters")
            return None, None
    
    # Store interface functions in system components
    system_components['explain_function'] = explain_rf_probability_decision
    system_components['batch_explain_function'] = batch_explain_probability_decisions
    system_components['summary_report_function'] = generate_cluster_summary_report
    system_components['get_qualitative_samples_function'] = get_qualitative_samples
    
    print("Modified probability-focused interface functions created!")
    print("Available functions:")
    print("  - explain_rf_probability_decision(sample, system_components)")
    print("  - batch_explain_probability_decisions(X, system_components)")
    print("  - generate_cluster_summary_report(system_components)")
    print("  - get_qualitative_samples(X, system_components)")
    
    return system_components

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main_pipeline(data_file_path='data/rf_all_test_predictions.csv'):
    """Complete modified pipeline execution"""
    print("="*80)
    print("MODIFIED CLUSTER-SPECIFIC SURROGATE MODELS FOR PROBABILITY PREDICTION")
    print("="*80)
    print("Modifications:")
    print("   • Clusters 1 & 2 merged for single Ridge regression")
    print("   • Cluster 3 excluded from training (saved for qualitative analysis)")
    print("="*80)
    
    # Phase 2: UMAP & Clustering Setup (unchanged)
    print("\nStarting Phase 2: UMAP & Clustering Setup")
    X_original, rf_probabilities, rf_hard_predictions, y_original, feature_cols = load_and_prepare_data(data_file_path)
    umap_model, umap_coords = generate_umap_coordinates(X_original, rf_probabilities, y_original)
    clustering, cluster_labels, valid_clusters = apply_clustering(umap_coords, rf_probabilities)
    
    # Phase 3: Modified Cluster Analysis & Model Selection
    print("\nStarting Phase 3: Cluster Analysis & Model Selection (Modified)")
    cluster_analysis = analyze_probability_clusters(cluster_labels, rf_probabilities, y_original, valid_clusters)
    surrogate_models, training_clusters, qualitative_clusters, merged_clusters = select_probability_surrogate_models(cluster_analysis, valid_clusters)
    
    # Phase 4: Modified Training 
    print("\nStarting Phase 4: Training Probability-Based Surrogates (Modified)")
    trained_surrogates, surrogate_performance = train_probability_surrogates(
        X_original, rf_probabilities, cluster_labels, training_clusters, 
        surrogate_models, cluster_analysis, merged_clusters
    )
    
    # Phase 5: Modified Integration & Routing System
    print("\nStarting Phase 5: Integration & Routing System (Modified)")
    router = ClusterRouter(umap_model, clustering, training_clusters, qualitative_clusters, 
                          merged_clusters, umap_coords, cluster_labels)
    integrated_surrogate = ClusterAwareProbabilitySurrogate(router, trained_surrogates, feature_cols, qualitative_clusters)
    
    # Phase 6: Modified Validation & Evaluation
    print("\nStarting Phase 6: Validation & Evaluation (Modified)")
    validation_results = validate_probability_system(
        integrated_surrogate, X_original, rf_probabilities, cluster_labels, 
        training_clusters, qualitative_clusters
    )
    interpretation_report = generate_probability_interpretation_report(
        integrated_surrogate, cluster_analysis, surrogate_performance, 
        training_clusters, qualitative_clusters, merged_clusters, feature_cols
    )
    
    # Phase 7: Modified Save & Deploy
    print("\nStarting Phase 7: Save & Deploy (Modified)")
    system_components = save_probability_system(
        integrated_surrogate, interpretation_report, cluster_analysis, 
        surrogate_performance, feature_cols, training_clusters, qualitative_clusters,
        merged_clusters, validation_results
    )
    system_components = create_probability_interface(system_components)
    
    # Final Summary
    print("\n" + "="*80)
    print("MODIFIED PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Trained {len(training_clusters)} cluster-specific surrogate models")
    print(f"   • Merged clusters 1&2 into single Ridge model")
    print(f"   • Excluded cluster 3 for qualitative analysis")
    print(f"Overall probability prediction R²: {validation_results['overall_r2']:.3f}")
    print(f"Classification fidelity: {validation_results['classification_fidelity']:.3f}")
    print(f"Training samples: {validation_results['n_training_samples']}")
    print(f"Qualitative samples: {validation_results['n_qualitative_samples']}")
    print(f"System saved to: 'models/modified_probability_cluster_surrogate_system.pkl'")
    
    # Generate final summary report
    system_components['summary_report_function'](system_components)
    
    return system_components

# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATION (MODIFIED)
# =============================================================================

def demonstrate_system(system_components, X_original):
    """Demonstrate the modified system capabilities"""
    print("\n" + "="*80)
    print("MODIFIED SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Show some example explanations including qualitative samples
    system_components['batch_explain_function'](X_original, system_components, n_examples=4)
    
    # Demonstrate qualitative sample extraction
    print(f"\n--- Qualitative Sample Extraction ---")
    qual_samples, qual_indices = system_components['get_qualitative_samples_function'](X_original, system_components)
    
    if qual_samples is not None:
        print(f"Extracted {len(qual_samples)} samples for qualitative analysis")
        print(f"These samples can be analyzed manually for insights")
    
    # Test single sample explanation
    print(f"\n--- Detailed Single Sample Explanation ---")
    sample_idx = len(X_original) // 2  # Middle sample
    explanation = system_components['explain_function'](X_original[sample_idx], system_components)
    
    print(f"\nDetailed explanation object:")
    for key, value in explanation.items():
        if key == 'probability_contributions':
            print(f"  {key}: {value[:3]}..." if value else f"  {key}: None")
        else:
            print(f"  {key}: {value}")

def load_saved_system(filepath='models/modified_probability_cluster_surrogate_system.pkl'):
    """Load previously saved modified system"""
    print(f"Loading saved modified system from {filepath}...")
    system_components = joblib.load(filepath)
    print("Modified system loaded successfully!")
    
    # Print modification info
    if 'modifications' in system_components:
        print("System modifications:")
        for mod, status in system_components['modifications'].items():
            print(f"  • {mod}: {status}")
    
    return system_components

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Starting Modified Cluster-Specific Surrogate Models Implementation...")
    print("Make sure your data file 'rf_all_test_predictions.csv' is in the 'data/' directory")
    
    try:
        # Run the complete modified pipeline
        system_components = main_pipeline()
        
        # Print out system progress
        print(f"\nAll phases completed successfully!")
        print(f"\nModifications applied:")
        print(f"   Merged clusters 1 & 2 for single Ridge regression")
        print(f"   Excluded cluster 3 for qualitative analysis")
        print(f"   System saved with modification tracking")
        
        print("\nTo use the saved system later:")
        print("  system = load_saved_system('models/modified_probability_cluster_surrogate_system.pkl')")
        print("  explanation = system['explain_function'](sample, system)")
        print("  qual_samples, qual_indices = system['get_qualitative_samples_function'](X, system)")
        
    except FileNotFoundError:
        print("Error: Could not find 'data/rf_all_test_predictions.csv'")
        print("Please ensure your data file is in the correct location.")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        print("Please check your data format and dependencies.")