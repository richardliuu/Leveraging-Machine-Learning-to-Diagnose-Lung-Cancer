"""
Author: Richard Liu

Description:
This module implements a Random Forest approach for lung cancer stage classification
using scikit-learn's RandomForestClassifier. 

Key Features:
- Patient-grouped cross-validation to prevent data leakage
- Data integrity checks for duplicate samples and label consistency
- Balanced class weighting to handle class imbalance
- Comprehensive performance evaluation with ROC-AUC metrics
- Feature importance analysis for model interpretability
- Reproducible results with fixed random seeds
- Model persistence with joblib

The script expects a CSV file containing lung cancer data with 
features, patient IDs, and cancer stage labels.
"""

import pandas as pd
import numpy as np
import random
import os
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Seed setting for reproducibility
os.environ['PYTHONHASHSEED'] = '42'
SEED = 141
np.random.seed(SEED)
random.seed(SEED)

class DataHandling:
    """
    
    """

    def __init__(self):
        self.data = "data/train_data.csv"
        
        self.reports = []
        self.conf_matrices = []
        self.roc_aucs = []
        self.fold_details = []
        self.predictions = []
        
        # Data attributes
        self.X = None
        self.y = None
        self.groups = None
        self.feature_cols = None
        
        # Current fold data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_patients = None
        self.test_patients = None
        
    def load_data(self):
        print("Loading dataset...")
        df = pd.read_csv(self.data)
        print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")
        
        # Prepare features and labels and drop metadata
        metadata_cols = ['chunk', 'cancer_stage', 'patient_id', 'filename', 'bandwidth', 'skew', 'rolloff', 'zcr', 'rms']
        cols_to_drop = [col for col in metadata_cols if col in df.columns]
        
        self.X = df.drop(columns=cols_to_drop)
        self.y = df['cancer_stage']
        self.groups = df['patient_id']
        self.feature_cols = self.X.columns.tolist()
        
        print(f"Total samples: {len(df)}")
        print(f"Total patients: {df['patient_id'].nunique()}")
        print(f"Features: {self.X.shape[1]}")
        
    def split(self, df, train_idx, test_idx):
        """

        """

        self.X_train, self.X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
        self.y_train, self.y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
        
        # Track patients in each split to verify no leakage
        self.train_patients = set(df.iloc[train_idx]['patient_id'])
        self.test_patients = set(df.iloc[test_idx]['patient_id'])

class RandomForestModel:
    """
    
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            criterion="log_loss",
            n_estimators=200,
            max_depth=5,
            max_leaf_nodes=None,
            max_features=0.3,
            min_samples_split=25,
            min_samples_leaf=10,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True,
            random_state=SEED,
            n_jobs=-1,
        )
        self.feature_cols = None
    
    def train(self, X_train, y_train, feature_cols):
        """
    
        """
        self.feature_cols = feature_cols
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self):
        """

        """

        if self.feature_cols is None:
            raise ValueError("Model must be trained first")
            
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, 
                                top_n=15, 
                                save_path='data/results/rf_feature_importance.png'
                                ):
        """

        """
        importance_df = self.get_feature_importance()
        
        plt.figure(figsize=(10, 6))
        
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances from Random Forest Model')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
        plt.show()
    
    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        y_pred = self.predict(X_test)
        y_pred_prob = self.predict_proba(X_test)[:, 1] 
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        c_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC-AUC score with error handling
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
        except:
            auc = np.nan
        
        # Calculate training performance for overfitting check if training data provided
        train_accuracy = None
        overfitting_gap = None
        if X_train is not None and y_train is not None:
            y_train_pred = self.predict(X_train)
            train_accuracy = (y_train_pred == y_train).mean()
            overfitting_gap = train_accuracy - report['accuracy']
        
        return report, c_matrix, auc, train_accuracy, overfitting_gap

class SHAPAnalyzer:
    """
    
    """

    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.fold_shap_values = []
        self.fold_explainers = []
    
    def analyze_fold(self, 
                     model, 
                     X_test, 
                     feature_cols, 
                     fold_num, 
                     save_plot=False
                     ):
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

        """
        Small background set 
        """

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
            shap_vals_to_plot = shap_values[1]  
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
        shap.summary_plot(shap_vals_to_plot, 
                          X_test_df, 
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
    
class Summary(RandomForestModel):  
    """
    Class will be used to summarize results of the pipeline
    """
    def __init__(self):
        super().__init__()

    def save_results(self, handler, fold, y_pred):
        """Save fold results to handler"""
        report, c_matrix, auc, train_accuracy, overfitting_gap = self.evaluate(
            handler.X_test, 
            handler.y_test, 
            handler.X_train, 
            handler.y_train
        )

        handler.reports.append(report)
        handler.conf_matrices.append(c_matrix)
        handler.roc_aucs.append(auc)
        handler.predictions.append(y_pred)
        handler.fold_details.append({
            'fold': fold + 1,
            'train_patients': len(handler.train_patients),
            'test_patients': len(handler.test_patients),
            'train_samples': len(handler.X_train),
            'test_samples': len(handler.X_test),
            'accuracy': report['accuracy'],
            'train_accuracy': train_accuracy,
            'overfitting_gap': overfitting_gap
        })

        return report, c_matrix, auc, train_accuracy, overfitting_gap
    
    def display_fold_results(self, 
                             fold,
                             handler, 
                             report, 
                             c_matrix, 
                             auc, 
                             train_accuracy, 
                             overfitting_gap
                             ):
        
        """Display results for a single fold"""
        y_pred = handler.predictions[-1]
        print(f"\nFold {fold+1} Results:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {report['accuracy']:.4f}")
        print(f"Overfitting Gap: {overfitting_gap:.4f}")
        print("\nTest Set Classification Report:")
        print(classification_report(handler.y_test, y_pred))
        print("Confusion Matrix:")
        print(c_matrix)
        print(f"ROC AUC Score: {auc:.4f}")
    
    def print_summary(self, handler):
        """Print cross-validation summary statistics"""
        print(f"\n{'='*60}\nCROSS-VALIDATION SUMMARY\n{'='*60}")
        
        # Calculate accuracy statistics
        accuracies = [r['accuracy'] for r in handler.reports]
        train_accuracies = [d['train_accuracy'] for d in handler.fold_details]
        overfitting_gaps = [d['overfitting_gap'] for d in handler.fold_details]
        
        print("Per-fold results:")
        for i, details in enumerate(handler.fold_details):
            print(f"Fold {i+1}: Train={details['train_accuracy']:.4f}, Test={details['accuracy']:.4f}, "
                  f"Gap={details['overfitting_gap']:.4f} "
                  f"({details['test_patients']} patients, {details['test_samples']} samples)")
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        avg_train_accuracy = np.mean(train_accuracies)
        avg_overfitting_gap = np.mean(overfitting_gaps)
        
        print(f"\nOverall Performance:")
        print(f"Mean Training Accuracy: {avg_train_accuracy:.4f}")
        print(f"Mean Test Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Mean Overfitting Gap: {avg_overfitting_gap:.4f}")
        print(f"Min Test Accuracy: {min(accuracies):.4f}")
        print(f"Max Test Accuracy: {max(accuracies):.4f}")
        
        # Calculate class-wise F1 scores
        class_0_f1 = [r['0']['f1-score'] for r in handler.reports]
        class_1_f1 = [r['1']['f1-score'] for r in handler.reports]
        print(f"\nClass-wise F1-scores:")
        print(f"Class 0: {np.mean(class_0_f1):.4f} ± {np.std(class_0_f1):.4f}")
        print(f"Class 1: {np.mean(class_1_f1):.4f} ± {np.std(class_1_f1):.4f}")
        
        # Display average confusion matrix
        avg_conf_matrix = np.mean(handler.conf_matrices, axis=0)
        print(f"\nAverage Confusion Matrix:")
        print(np.round(avg_conf_matrix).astype(int))
        
        return class_0_f1, class_1_f1
    
    def plot_f1_scores(self, class_0_f1, class_1_f1):
        """Plot F1 scores per fold"""
        plt.figure(figsize=(10, 6))
        folds = list(range(1, len(class_0_f1) + 1))
        
        plt.plot(folds, class_0_f1, 'o-', label='Class 0 F1-score', linewidth=2, markersize=8)
        plt.plot(folds, class_1_f1, 's-', label='Class 1 F1-score', linewidth=2, markersize=8)
        
        plt.xlabel('Fold')
        plt.ylabel('F1-score')
        plt.title('F1-score per Fold for Random Forest Classifier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(folds)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
        
class CrossValidation(Summary):
    """
    
    """
    
    def __init__(self, handler):
        super().__init__()
        self.df = pd.read_csv(handler.data)
        self.group_kfold = StratifiedGroupKFold(n_splits=4)
        self.handler = handler
        self.shap_analyzer = None
        self.final_model = None
    
    def save_surrogate_data(self, handler, test_idx, y_pred):
        """Save predictions for surrogate model training"""
        results_df = handler.X_test.copy()
        results_df['true_label'] = handler.y_test.values
        results_df['predicted_label'] = y_pred
        y_pred_prob = self.model.predict_proba(handler.X_test)[:, 1]
        results_df['c1_prob'] = y_pred_prob

        training_data = pd.read_csv("data/train_data.csv")
        results_df['patient_id'] = training_data.iloc[test_idx]['patient_id'].values
        results_df['chunk'] = training_data.iloc[test_idx]['chunk'].values
        results_df.to_csv('data/rf2_surrogate_data.csv', index=False)
        
    def pipeline(self):
        """Execute cross-validation pipeline"""
        handler = self.handler
        
        # Execute cross-validation
        for fold, (train_idx, test_idx) in enumerate(self.group_kfold.split(handler.X, handler.y, handler.groups)):
            print(f"\n{'='*50}\nFOLD {fold+1}/4\n{'='*50}")
            
            # Split data for current fold
            handler.split(self.df, train_idx, test_idx)
            
            print(f"Train: {len(handler.train_patients)} patients, {len(handler.X_train)} samples")
            print(f"Test:  {len(handler.test_patients)} patients, {len(handler.X_test)} samples")
            
            # Train model 
            self.train(handler.X_train, handler.y_train, handler.feature_cols)
            
            # Evaluate and save results
            y_pred = self.predict(handler.X_test)
            report, c_matrix, auc, train_accuracy, overfitting_gap = self.save_results(handler, fold, y_pred)
            
            # Save data for surrogate model
            self.save_surrogate_data(handler, 
                                     test_idx, 
                                     y_pred
                                     )
            
            # Display fold results
            self.display_fold_results(fold, 
                                      handler, 
                                      report, 
                                      c_matrix, 
                                      auc, 
                                      train_accuracy, 
                                      overfitting_gap
                                      )
            
            # SHAP analysis
            if fold == 0:
                self.shap_analyzer = SHAPAnalyzer()
            
            shap_values, explainer = self.shap_analyzer.analyze_fold(
                self.model,
                handler.X_test,
                handler.feature_cols,
                fold_num=fold+1,
                save_plot=False
            )
            
            # Store final model for feature importance
            if fold == 3:
                self.final_model = self
        
        # Generate feature importance plot from final model
        print("\nGenerating feature importance plot...")
        self.final_model.plot_feature_importance(top_n=15)
        
        # Print summary statistics
        class_0_f1, class_1_f1 = self.print_summary(handler)
        
        # Plot F1 scores
        self.plot_f1_scores(class_0_f1, class_1_f1)
        
        # Generate cross-fold SHAP summary
        print("\nGenerating cross-fold SHAP summary...")
        shap_summary = self.shap_analyzer.generate_cross_fold_summary(save_plot=False)
        
        return handler, self.shap_analyzer

def main():
    print("Random Forest Classification for Lung Cancer Stage Prediction")
    print("=" * 65)
    
    # Initialize Class Objects 
    handler = DataHandling()
    
    # Step 1: Load and validate data
    print("\nStep 1: Data Loading and Integrity Checks")
    is_clean = handler.load_data()
    
    if not is_clean:
        print("\nDATA QUALITY ISSUES DETECTED! Results may be unreliable")
    
    # Step 2: Execute cross-validation pipeline
    print("\nStep 2: Cross-Validation Pipeline with SHAP Analysis")
    cv = CrossValidation(handler)
    results = cv.pipeline()
    
    if results is not None:
        handler, shap_analyzer = results
        print("\nPipeline completed successfully with SHAP analysis for all folds!")

if __name__ == "__main__":
    main()