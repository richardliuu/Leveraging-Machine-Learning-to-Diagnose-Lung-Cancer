"""
Author: Richard Liu

Description: 

This module creates a global decision tree surrogate model to explain 
the Random Forest classifier predictions. The surrogate provides 
interpretability by approximating the complex Random Forest with 
a simpler, more interpretable decision tree.

Key Features:
- Global decision tree surrogate training
- Fidelity analysis (how well surrogate mimics Random Forest)
- Feature importance extraction
- Decision path visualization
- Model persistence (save/load)

The script expects a CSV file containing the Random Forest predictions.
"""

# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import StratifiedGroupKFold
import joblib
import warnings
import os
warnings.filterwarnings('ignore')


class LoadData:
    """Load and prepare data for surrogate modeling"""
    
    @staticmethod
    def load_data(file_path='data/rf_test_predictions.csv'):
        """
        Load Random Forest predictions for surrogate training.
        
        Args:
            file_path: Path to CSV file with RF predictions
            
        Returns:
            X: Feature matrix
            y_true: True labels
            y_pred_rf: Random Forest predictions
            y_prob_rf: Random Forest probabilities
            groups: Patient IDs for group-based splitting
            feature_cols: List of feature column names
        """
        print("="*60)
        print("Loading Data for Surrogate Model")
        print("="*60)
        
        # Load predictions from Random Forest
        df = pd.read_csv(file_path)
        
        # Extract components
        exclude_cols = ['true_label', 'predicted_label', 'probability', 
                       'patient_id', 'chunk', 'fold']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y_true = df['true_label'].values
        y_pred_rf = df['predicted_label'].values
        y_prob_rf = df['probability'].values
        groups = df['patient_id'].values
        
        print(f"Data shape: {X.shape}")
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Unique patients: {len(np.unique(groups))}")
        print(f"RF Accuracy: {accuracy_score(y_true, y_pred_rf):.3f}")
        print(f"Class distribution: {np.bincount(y_pred_rf.astype(int))}")
        
        return X, y_true, y_pred_rf, y_prob_rf, groups, feature_cols


class GlobalSurrogate:
    """
    Global decision tree surrogate model that explains Random Forest predictions.
    Uses regression to predict probabilities directly.
    """
    
    def __init__(self, max_depth=5, min_samples_split=20, min_samples_leaf=10):
        """
        Initialize the surrogate model.
        
        Args:
            max_depth: Maximum depth of decision tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at leaf node
        """
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        self.feature_cols = None
        self.is_trained = False
        self.threshold = 0.5  # Threshold for binary classification
        
    def train(self, X, y_prob_rf, feature_cols):
        """
        Train the surrogate to mimic Random Forest probabilities.
        
        Args:
            X: Feature matrix
            y_prob_rf: Random Forest probabilities (targets for surrogate)
            feature_cols: List of feature names
        """
        print("\n" + "="*60)
        print("Training Global Decision Tree Surrogate (Regressor)")
        print("="*60)
        
        self.feature_cols = feature_cols
        self.model.fit(X, y_prob_rf)
        self.is_trained = True
        
        print(f"Surrogate trained successfully")
        print(f"Tree depth: {self.model.get_depth()}")
        print(f"Number of leaves: {self.model.get_n_leaves()}")
        
    def predict(self, X):
        """Predict binary class using surrogate model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        probs = self.model.predict(X)
        return (probs >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities using surrogate model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        prob_class1 = self.model.predict(X)
        # Clip to [0, 1] range
        prob_class1 = np.clip(prob_class1, 0, 1)
        prob_class0 = 1 - prob_class1
        return np.column_stack([prob_class0, prob_class1])
    
    def get_feature_importance(self):
        """
        Get feature importance from the decision tree.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_decision_rules(self, max_rules=10):
        """
        Extract human-readable decision rules from the tree.
        
        Args:
            max_rules: Maximum depth to display
            
        Returns:
            String representation of decision rules
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        tree_rules = export_text(
            self.model, 
            feature_names=self.feature_cols,
            max_depth=max_rules
        )
        return tree_rules
    
    def visualize_tree(self, save_path='data/results/surrogate_tree.png', max_depth=3):
        """
        Visualize the decision tree.
        
        Args:
            save_path: Path to save the visualization
            max_depth: Maximum depth to display
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model, 
            feature_names=self.feature_cols,
            filled=True,
            rounded=True,
            max_depth=max_depth,
            fontsize=10
        )
        plt.title('Global Decision Tree Surrogate (Regressor)', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tree visualization saved to {save_path}")
        plt.show()


class SurrogateEvaluator:
    """Evaluate surrogate model fidelity and performance"""
    
    @staticmethod
    def evaluate_fidelity(y_true, y_pred_rf, y_pred_surrogate, y_prob_rf, y_prob_surrogate):
        """
        Evaluate how well the surrogate mimics the Random Forest.
        
        Args:
            y_true: True labels
            y_pred_rf: Random Forest predictions
            y_pred_surrogate: Surrogate predictions
            y_prob_rf: Random Forest probabilities
            y_prob_surrogate: Surrogate probabilities
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("Evaluating Surrogate Fidelity")
        print("="*60)
        
        # Fidelity: Agreement between RF and Surrogate
        fidelity = accuracy_score(y_pred_rf, y_pred_surrogate)
        
        # Accuracy: Surrogate vs ground truth
        surrogate_accuracy = accuracy_score(y_true, y_pred_surrogate)
        rf_accuracy = accuracy_score(y_true, y_pred_rf)
        
        # Probability agreement
        prob_mse = mean_squared_error(y_prob_rf, y_prob_surrogate[:, 1])
        prob_r2 = r2_score(y_prob_rf, y_prob_surrogate[:, 1])
        
        results = {
            'fidelity': fidelity,
            'surrogate_accuracy': surrogate_accuracy,
            'rf_accuracy': rf_accuracy,
            'accuracy_gap': rf_accuracy - surrogate_accuracy,
            'probability_mse': prob_mse,
            'probability_r2': prob_r2
        }
        
        print(f"\nFidelity Metrics:")
        print(f"  Fidelity (RF-Surrogate agreement): {fidelity:.3f}")
        print(f"  Random Forest Accuracy: {rf_accuracy:.3f}")
        print(f"  Surrogate Accuracy: {surrogate_accuracy:.3f}")
        print(f"  Accuracy Gap: {results['accuracy_gap']:.3f}")
        print(f"\nProbability Metrics:")
        print(f"  MSE: {prob_mse:.4f}")
        print(f"  R²: {prob_r2:.3f}")
        
        return results
    
    @staticmethod
    def plot_confusion_matrices(y_true, y_pred_rf, y_pred_surrogate):
        """
        Plot confusion matrices comparing RF and Surrogate.
        
        Args:
            y_true: True labels
            y_pred_rf: Random Forest predictions
            y_pred_surrogate: Surrogate predictions
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # RF confusion matrix
        cm_rf = confusion_matrix(y_true, y_pred_rf)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Random Forest\nConfusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Surrogate confusion matrix
        cm_surrogate = confusion_matrix(y_true, y_pred_surrogate)
        sns.heatmap(cm_surrogate, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('Decision Tree Surrogate\nConfusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        plt.tight_layout()
        save_path = 'data/results/confusion_matrices.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
        plt.show()
    
    @staticmethod
    def plot_probability_comparison(y_prob_rf, y_prob_surrogate):
        """
        Compare probability distributions of RF and Surrogate.
        
        Args:
            y_prob_rf: Random Forest probabilities
            y_prob_surrogate: Surrogate probabilities (class 1)
        """
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_prob_rf, y_prob_surrogate[:, 1], alpha=0.3, s=10)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
        plt.xlabel('Random Forest Probability (Class 1)')
        plt.ylabel('Surrogate Probability (Class 1)')
        plt.title('Probability Agreement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Distribution comparison
        plt.subplot(1, 2, 2)
        plt.hist(y_prob_rf, bins=30, alpha=0.5, label='Random Forest', density=True)
        plt.hist(y_prob_surrogate[:, 1], bins=30, alpha=0.5, label='Surrogate', density=True)
        plt.xlabel('Probability (Class 1)')
        plt.ylabel('Density')
        plt.title('Probability Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = 'data/results/probability_comparison.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability comparison saved to {save_path}")
        plt.show()
    
    @staticmethod
    def plot_feature_importance(importance_df, top_n=15):
        """
        Plot feature importance from surrogate model.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to display
        """
        plt.figure(figsize=(10, 6))
        
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances from Surrogate Model')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        save_path = 'data/results/surrogate_feature_importance.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
        plt.show()


def main_pipeline(data_file_path='data/rf_test_predictions.csv', 
                 max_depth=4, 
                 min_samples_split=20,
                 min_samples_leaf=10):
    """
    Complete pipeline with 4-fold stratified group k-fold cross-validation.
    
    Args:
        data_file_path: Path to RF predictions CSV
        max_depth: Maximum depth of surrogate tree
        min_samples_split: Minimum samples to split node
        min_samples_leaf: Minimum samples at leaf
        
    Returns:
        Dictionary with surrogate model and evaluation results
    """
    print("="*60)
    print("GLOBAL DECISION TREE SURROGATE MODEL")
    print("WITH 4-FOLD STRATIFIED GROUP K-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Step 1: Load data
    X, y_true, y_pred_rf, y_prob_rf, groups, feature_cols = LoadData.load_data(data_file_path)
    
    # Step 2: Set up cross-validation
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    
    # Storage for cross-validation results
    cv_results = {
        'fidelity_scores': [],
        'surrogate_accuracy_scores': [],
        'rf_accuracy_scores': [],
        'prob_mse_scores': [],
        'prob_r2_scores': [],
        'fold_details': []
    }
    
    all_y_pred_surrogate = np.zeros_like(y_pred_rf)
    all_y_prob_surrogate = np.zeros((len(y_pred_rf), 2))
    
    # Step 3: Cross-validation loop
    print("\n" + "="*60)
    print("CROSS-VALIDATION")
    print("="*60)
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_pred_rf, groups)):
        print(f"\n{'='*50}\nFOLD {fold+1}/4\n{'='*50}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_rf, y_test_rf = y_pred_rf[train_idx], y_pred_rf[test_idx]
        y_train_true, y_test_true = y_true[train_idx], y_true[test_idx]
        y_train_prob, y_test_prob = y_prob_rf[train_idx], y_prob_rf[test_idx]
        
        # Verify no patient leakage
        train_patients = set(groups[train_idx])
        test_patients = set(groups[test_idx])
        overlap = train_patients.intersection(test_patients)
        if overlap:
            print(f"WARNING: Patient leakage detected! {overlap}")
        else:
            print(f"✓ No patient leakage")
        
        print(f"Train: {len(train_patients)} patients, {len(X_train)} samples")
        print(f"Test:  {len(test_patients)} patients, {len(X_test)} samples")
        
        # Train surrogate on this fold (using probabilities as target)
        surrogate = GlobalSurrogate(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        surrogate.train(X_train, y_train_prob, feature_cols)
        
        # Make predictions on test set
        y_pred_fold = surrogate.predict(X_test)
        y_prob_fold = surrogate.predict_proba(X_test)
        
        # Store predictions for later full evaluation
        all_y_pred_surrogate[test_idx] = y_pred_fold
        all_y_prob_surrogate[test_idx] = y_prob_fold
        
        # Calculate fold metrics
        fidelity = accuracy_score(y_test_rf, y_pred_fold)
        surrogate_acc = accuracy_score(y_test_true, y_pred_fold)
        rf_acc = accuracy_score(y_test_true, y_test_rf)
        prob_mse = mean_squared_error(y_test_prob, y_prob_fold[:, 1])
        prob_r2 = r2_score(y_test_prob, y_prob_fold[:, 1])
        
        # Store metrics
        cv_results['fidelity_scores'].append(fidelity)
        cv_results['surrogate_accuracy_scores'].append(surrogate_acc)
        cv_results['rf_accuracy_scores'].append(rf_acc)
        cv_results['prob_mse_scores'].append(prob_mse)
        cv_results['prob_r2_scores'].append(prob_r2)
        cv_results['fold_details'].append({
            'fold': fold + 1,
            'train_patients': len(train_patients),
            'test_patients': len(test_patients),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'fidelity': fidelity,
            'surrogate_accuracy': surrogate_acc,
            'rf_accuracy': rf_acc,
            'prob_mse': prob_mse,
            'prob_r2': prob_r2
        })
        
        # Print fold results
        print(f"\nFold {fold+1} Results:")
        print(f"  Fidelity (RF-Surrogate agreement): {fidelity:.3f}")
        print(f"  Surrogate Accuracy: {surrogate_acc:.3f}")
        print(f"  RF Accuracy: {rf_acc:.3f}")
        print(f"  Probability MSE: {prob_mse:.4f}")
        print(f"  Probability R²: {prob_r2:.3f}")
    
    # Step 4: Print cross-validation summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    print("\nPer-fold results:")
    for details in cv_results['fold_details']:
        print(f"Fold {details['fold']}: Fidelity={details['fidelity']:.3f}, "
              f"Acc={details['surrogate_accuracy']:.3f}, "
              f"R²={details['prob_r2']:.3f} "
              f"({details['test_patients']} patients, {details['test_samples']} samples)")
    
    print(f"\nMean Metrics (across folds):")
    print(f"  Fidelity: {np.mean(cv_results['fidelity_scores']):.3f} ± {np.std(cv_results['fidelity_scores']):.3f}")
    print(f"  Surrogate Accuracy: {np.mean(cv_results['surrogate_accuracy_scores']):.3f} ± {np.std(cv_results['surrogate_accuracy_scores']):.3f}")
    print(f"  RF Accuracy: {np.mean(cv_results['rf_accuracy_scores']):.3f} ± {np.std(cv_results['rf_accuracy_scores']):.3f}")
    print(f"  Probability MSE: {np.mean(cv_results['prob_mse_scores']):.4f} ± {np.std(cv_results['prob_mse_scores']):.4f}")
    print(f"  Probability R²: {np.mean(cv_results['prob_r2_scores']):.3f} ± {np.std(cv_results['prob_r2_scores']):.3f}")
    
    # Step 5: Train final model on all data for interpretation
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("="*60)
    
    final_surrogate = GlobalSurrogate(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    final_surrogate.train(X, y_prob_rf, feature_cols)
    
    # Step 6: Get feature importance
    importance_df = final_surrogate.get_feature_importance()
    print(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Step 7: Get decision rules
    print(f"\nDecision Rules (simplified):")
    rules = final_surrogate.get_decision_rules(max_rules=3)
    print(rules)
    
    # Step 8: Evaluate on full dataset (using cross-validated predictions)
    evaluator = SurrogateEvaluator()
    full_results = evaluator.evaluate_fidelity(
        y_true, y_pred_rf, all_y_pred_surrogate, y_prob_rf, all_y_prob_surrogate
    )
    
    # Step 9: Visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_confusion_matrices(y_true, y_pred_rf, all_y_pred_surrogate)
    evaluator.plot_probability_comparison(y_prob_rf, all_y_prob_surrogate)
    evaluator.plot_feature_importance(importance_df, top_n=15)
    final_surrogate.visualize_tree(max_depth=3)
    
    # Step 10: Save model
    print("\nSaving surrogate model...")
    model_data = {
        'surrogate': final_surrogate,
        'feature_cols': feature_cols,
        'importance_df': importance_df,
        'cv_results': cv_results,
        'full_evaluation': full_results,
        'hyperparameters': {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }
    }
    model_save_path = 'data/results/global_surrogate_model.pkl'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model_data, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Cross-Validation Mean Fidelity: {np.mean(cv_results['fidelity_scores']):.3f}")
    print(f"Cross-Validation Mean Accuracy: {np.mean(cv_results['surrogate_accuracy_scores']):.3f}")
    print(f"Final Model Tree Depth: {final_surrogate.model.get_depth()}")
    print(f"Final Model Number of Leaves: {final_surrogate.model.get_n_leaves()}")
    
    return model_data


def load_saved_model(filepath='data/results/global_surrogate_model.pkl'):
    """
    Load previously saved surrogate model.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Dictionary with model and metadata
    """
    print(f"Loading surrogate model from {filepath}...")
    model_data = joblib.load(filepath)
    print("Model loaded successfully!")
    
    print("\nModel Information:")
    if 'cv_results' in model_data:
        print(f"  Cross-Validation Fidelity: {np.mean(model_data['cv_results']['fidelity_scores']):.3f} ± {np.std(model_data['cv_results']['fidelity_scores']):.3f}")
        print(f"  Cross-Validation Accuracy: {np.mean(model_data['cv_results']['surrogate_accuracy_scores']):.3f} ± {np.std(model_data['cv_results']['surrogate_accuracy_scores']):.3f}")
    elif 'evaluation_results' in model_data:
        print(f"  Fidelity: {model_data['evaluation_results']['fidelity']:.3f}")
        print(f"  Accuracy: {model_data['evaluation_results']['surrogate_accuracy']:.3f}")
    print(f"  Tree Depth: {model_data['surrogate'].model.get_depth()}")
    print(f"  Features: {len(model_data['feature_cols'])}")
    
    return model_data


if __name__ == "__main__":
    print("Starting Global Decision Tree Surrogate Model Training...")
    print("Make sure 'data/rf_test_predictions.csv' exists with RF predictions\n")
    
    try:
        # Run the pipeline
        model_data = main_pipeline(
            data_file_path='data/rf_test_predictions.csv',
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10
        )
        
        print("\nTo use the saved model later:")
        print("  model_data = load_saved_model('data/results/global_surrogate_model.pkl')")
        print("  surrogate = model_data['surrogate']")
        print("  predictions = surrogate.predict(X)")
        
    except FileNotFoundError:
        print("Error: Could not find 'data/rf_test_predictions.csv'")
        print("Please run the Random Forest model first to generate predictions.")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
