import shap
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) analyzer for Random Forest model interpretability.
    
    This class encapsulates all SHAP analysis functionality, providing a unified interface
    for generating and visualizing feature contributions to model predictions. SHAP values
    offer a game-theoretic approach to understanding how each feature impacts the model's
    output for individual predictions.
    
    Attributes:
        model: The trained Random Forest model to analyze
        feature_cols (list): Names of feature columns for labeling
        fold_shap_values (list): Stores SHAP values from each fold
        fold_explainers (list): Stores TreeExplainer objects from each fold
    """
    
    def __init__(self):
        """
        Initialize the SHAP analyzer.
        """
        self.model = None
        self.feature_cols = None
        self.fold_shap_values = []
        self.fold_explainers = []
    
    def analyze_fold(self, model, X_test, feature_cols, fold_num, save_plot=False):
        """
        Generate SHAP analysis for a specific fold.
        
        Args:
            model: Trained RandomForestClassifier model
            X_test (pd.DataFrame): Test data for generating SHAP values
            feature_cols (list): Feature column names
            fold_num (int): Current fold number (1-indexed)
            save_plot (bool): Whether to save the SHAP plot to file
            
        Returns:
            tuple: (shap_values, explainer) where:
                - shap_values: SHAP values for the positive class
                - explainer: TreeExplainer object for further analysis
        
        SHAP Analysis Process:
            1. Creates background dataset for baseline comparisons
            2. Initializes TreeExplainer optimized for tree-based models
            3. Computes SHAP values showing feature contributions
            4. Generates summary plot showing feature importance and impact
        """
        self.model = model
        self.feature_cols = feature_cols
        
        if self.model is None or self.feature_cols is None:
            print("Model not provided. Cannot generate SHAP analysis.")
            return None, None
        
        print(f"\nGenerating SHAP analysis for fold {fold_num}...")
        
        # Convert to DataFrame if needed and ensure column names
        X_test_df = pd.DataFrame(X_test, columns=self.feature_cols)
        
        # Sample background data for SHAP baseline
        # Using small background set for computational efficiency
        background_size = min(20, len(X_test_df))
        background = X_test_df.sample(n=background_size, random_state=42).to_numpy()
        
        # Initialize SHAP TreeExplainer with background data
        explainer = shap.TreeExplainer(self.model, background)
        
        # Calculate SHAP values for test set
        X_explain_np = X_test_df.to_numpy()
        shap_values = explainer.shap_values(X_explain_np)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # 3D array: (samples, features, classes)
            print(f"SHAP values shape (3D): {shap_values.shape}")
            class_index = 1  # Focus on positive class
            shap_vals_to_plot = shap_values[:, :, class_index]
        elif isinstance(shap_values, list):
            # List of arrays for each class
            shap_vals_to_plot = shap_values[1]  # Positive class
        else:
            # 2D array for binary classification
            shap_vals_to_plot = shap_values
        
        # Verify shape consistency
        assert shap_vals_to_plot.shape == X_explain_np.shape, \
            f"SHAP values shape {shap_vals_to_plot.shape} != input shape {X_explain_np.shape}"
        
        # Store fold results
        self.fold_shap_values.append(shap_vals_to_plot)
        self.fold_explainers.append(explainer)
        
        # Generate SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals_to_plot, X_test_df, 
                         feature_names=self.feature_cols, 
                         show=False)
        
        # Add title with fold information
        plt.title(f'SHAP Feature Importance - Fold {fold_num}', fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            filename = f'shap_analysis_fold_{fold_num}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"SHAP plot saved as {filename}")
        
        plt.show()
        
        # Calculate and display mean absolute SHAP values for this fold
        mean_abs_shap = np.abs(shap_vals_to_plot).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        print(f"\nFold {fold_num} - Top 10 Features by Mean Absolute SHAP Value:")
        print(shap_importance_df.head(10))
        
        return shap_vals_to_plot, explainer
    
    def generate_cross_fold_summary(self, save_plot=False):
        """
        Generate a summary of SHAP values across all folds.
        
        This method aggregates SHAP values from all folds to provide an overall
        view of feature importance consistency and variation across different
        data splits.
        
        Args:
            save_plot (bool): Whether to save the summary plot to file
        
        Returns:
            pd.DataFrame: Summary statistics of SHAP values across folds
        """
        if not self.fold_shap_values:
            print("No SHAP values available. Run analyze_fold first.")
            return None
        
        print("\n" + "="*60)
        print("CROSS-FOLD SHAP SUMMARY")
        print("="*60)
        
        # Concatenate all SHAP values across folds
        all_shap_values = np.concatenate(self.fold_shap_values, axis=0)
        
        # Calculate overall statistics
        mean_abs_shap_overall = np.abs(all_shap_values).mean(axis=0)
        std_abs_shap_overall = np.abs(all_shap_values).std(axis=0)
        
        # Create summary dataframe
        summary_df = pd.DataFrame({
            'feature': self.feature_cols,
            'mean_importance': mean_abs_shap_overall,
            'std_importance': std_abs_shap_overall,
            'cv_coefficient': std_abs_shap_overall / (mean_abs_shap_overall + 1e-10)
        }).sort_values('mean_importance', ascending=False)
        
        print("\nOverall Feature Importance (across all folds):")
        print(summary_df.head(15))
        
        # Calculate per-fold importance for consistency check
        fold_importances = []
        for fold_shap in self.fold_shap_values:
            fold_importance = np.abs(fold_shap).mean(axis=0)
            fold_importances.append(fold_importance)
        
        # Create consistency matrix
        fold_importance_matrix = np.array(fold_importances)
        
        # Plot feature importance consistency across folds
        top_n = 15
        top_features_idx = np.argsort(mean_abs_shap_overall)[-top_n:][::-1]
        top_features = [self.feature_cols[i] for i in top_features_idx]
        
        plt.figure(figsize=(14, 8))
        for fold_idx, fold_imp in enumerate(fold_importances):
            plt.plot(range(top_n), fold_imp[top_features_idx], 
                    'o-', alpha=0.7, label=f'Fold {fold_idx+1}')
        
        plt.plot(range(top_n), mean_abs_shap_overall[top_features_idx], 
                'k-', linewidth=3, label='Mean', marker='s', markersize=8)
        
        plt.xticks(range(top_n), top_features, rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Mean Absolute SHAP Value')
        plt.title('Feature Importance Consistency Across Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('shap_cross_fold_consistency.png', dpi=150, bbox_inches='tight')
            print("Cross-fold consistency plot saved as shap_cross_fold_consistency.png")
        
        plt.show()
        
        return summary_df

