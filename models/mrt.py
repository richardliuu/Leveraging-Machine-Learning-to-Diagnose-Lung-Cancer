import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import joblib


class MostRepresentativeTree:
    """
    Extract the Most Representative Tree (MRT) from a Random Forest model.
    Uses prediction similarity to ensemble as the primary criterion.
    """
    
    def __init__(self, random_forest: RandomForestClassifier):
        """
        Initialize with a trained Random Forest model.
        
        Args:
            random_forest: Trained sklearn RandomForestClassifier
        """
        self.rf = random_forest
        self.mrt_index = None
        self.mrt_model = None
        self.similarity_scores = None
        
    def calculate_tree_similarities(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """
        Calculate how similar each tree's predictions are to the ensemble predictions.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Array of similarity scores for each tree
        """
        # Get ensemble predictions
        ensemble_pred = self.rf.predict(X_test)
        ensemble_pred_proba = self.rf.predict_proba(X_test)
        
        similarities = []
        
        for i, tree in enumerate(self.rf.estimators_):
            # Get individual tree predictions
            tree_pred = tree.predict(X_test)
            tree_pred_proba = tree.predict_proba(X_test)
            
            # Calculate similarity metrics
            accuracy_similarity = accuracy_score(ensemble_pred, tree_pred)
            
            # Probability similarity (mean squared error, lower is better)
            prob_similarity = 1 / (1 + mean_squared_error(ensemble_pred_proba, tree_pred_proba))
            
            # Combined similarity score
            combined_similarity = (accuracy_similarity + prob_similarity) / 2
            similarities.append(combined_similarity)
            
        return np.array(similarities)
    
    def find_most_representative_tree(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[int, DecisionTreeClassifier]:
        """
        Find the tree that best represents the ensemble behavior.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tuple of (tree_index, tree_model)
        """
        self.similarity_scores = self.calculate_tree_similarities(X_test, y_test)
        
        # Find tree with highest similarity
        self.mrt_index = np.argmax(self.similarity_scores)
        self.mrt_model = self.rf.estimators_[self.mrt_index]
        
        return self.mrt_index, self.mrt_model
    
    def get_tree_statistics(self) -> dict:
        """
        Get statistics about the selected representative tree.
        
        Returns:
            Dictionary containing tree statistics
        """
        if self.mrt_model is None:
            raise ValueError("Must call find_most_representative_tree first")
            
        return {
            'tree_index': self.mrt_index,
            'max_similarity_score': self.similarity_scores[self.mrt_index],
            'mean_similarity_score': np.mean(self.similarity_scores),
            'std_similarity_score': np.std(self.similarity_scores),
            'tree_depth': self.mrt_model.get_depth(),
            'n_leaves': self.mrt_model.get_n_leaves(),
            'n_nodes': self.mrt_model.tree_.node_count
        }
    
    def plot_similarity_distribution(self, save_path: Optional[str] = None):
        """
        Plot distribution of tree similarity scores.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.similarity_scores is None:
            raise ValueError("Must call find_most_representative_tree first")
            
        plt.figure(figsize=(10, 6))
        plt.hist(self.similarity_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(self.similarity_scores[self.mrt_index], color='red', linestyle='--', 
                   label=f'Most Representative Tree (Index {self.mrt_index})')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Tree Similarity Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_tree_structure(self, feature_names: list, save_path: str):
        """
        Export the representative tree structure to DOT format.
        
        Args:
            feature_names: List of feature names
            save_path: Path to save the DOT file
        """
        if self.mrt_model is None:
            raise ValueError("Must call find_most_representative_tree first")
            
        dot_data = export_graphviz(
            self.mrt_model,
            feature_names=feature_names,
            class_names=['0', '1'],
            filled=True,
            rounded=True,
            special_characters=True
        )
        
        with open(save_path, 'w') as f:
            f.write(dot_data)
            
        print(f"Tree structure exported to {save_path}")
    
    def compare_predictions(self, X_test: np.ndarray) -> pd.DataFrame:
        """
        Compare predictions between the representative tree and ensemble.
        
        Args:
            X_test: Test features
            
        Returns:
            DataFrame with ensemble and tree predictions
        """
        if self.mrt_model is None:
            raise ValueError("Must call find_most_representative_tree first")
            
        ensemble_pred = self.rf.predict(X_test)
        ensemble_proba = self.rf.predict_proba(X_test)[:, 1]
        
        tree_pred = self.mrt_model.predict(X_test)
        tree_proba = self.mrt_model.predict_proba(X_test)[:, 1]
        
        comparison_df = pd.DataFrame({
            'ensemble_prediction': ensemble_pred,
            'ensemble_probability': ensemble_proba,
            'tree_prediction': tree_pred,
            'tree_probability': tree_proba,
            'prediction_match': ensemble_pred == tree_pred
        })
        
        return comparison_df


def extract_mrt_from_saved_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray, 
                                feature_names: list) -> MostRepresentativeTree:
    rf_model = joblib.load(model_path)
    
    mrt_extractor = MostRepresentativeTree(rf_model)
    mrt_extractor.find_most_representative_tree(X_test, y_test)
    
    print("=== MOST REPRESENTATIVE TREE ANALYSIS ===")
    stats = mrt_extractor.get_tree_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return mrt_extractor


if __name__ == "__main__":
    # Example usage with the saved Random Forest model
    try:
        # Load test data (you may need to adjust this path)
        df = pd.read_csv("data/jitter_shimmerlog.csv")
        
        # Prepare features (same as in randfor.py)
        X = df.drop(columns=['chunk', 'cancer_stage', 'patient_id', 'filename', 'rolloff', 'bandwidth', "skew", "zcr", 'rms'])
        y = df['cancer_stage']
        feature_names = X.columns.tolist()
        
        # Use a subset for testing
        X_test = X.iloc[:100].values
        y_test = y.iloc[:100].values
        
        # Extract MRT from saved model
        mrt = extract_mrt_from_saved_model("models/rf2_model.pkl", X_test, y_test, feature_names)
        
        # Generate visualizations and exports
        mrt.plot_similarity_distribution("tree_similarity_distribution.png")
        mrt.export_tree_structure(feature_names, "representative_tree.dot")
        
        # Compare predictions
        comparison = mrt.compare_predictions(X_test)
        print(f"\nPrediction agreement: {comparison['prediction_match'].mean():.2%}")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure the Random Forest model is saved as 'rf_model.pkl'")
    except Exception as e:
        print(f"Error: {e}")