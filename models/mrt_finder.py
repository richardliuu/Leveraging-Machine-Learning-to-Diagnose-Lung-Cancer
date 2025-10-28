import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import joblib
import copy 


class MostRepresentativeTree:
    """
    Extract the Most Representative Tree (MRT) from a Random Forest model.
    Uses prediction similarity to ensemble as the primary criterion.
    """
    
    def __init__(self, random_forest: RandomForestClassifier):
        self.rf = random_forest
        self.mrt_index = None
        self.mrt_model = None
        self.similarity_scores = None
        
    def calculate_tree_similarities(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        ensemble_pred = self.rf.predict(X_test)
        ensemble_pred_proba = self.rf.predict_proba(X_test)
        
        similarities = []
        prob_sims = []
        
        for i, tree in enumerate(self.rf.estimators_):
            tree_pred = tree.predict(X_test)
            tree_pred_proba = tree.predict_proba(X_test)

            accuracy_similarity = accuracy_score(ensemble_pred, tree_pred)
            prob_similarity = 1 / (1 + mean_squared_error(ensemble_pred_proba, tree_pred_proba))
            
            combined_similarity = (accuracy_similarity + prob_similarity) / 2
            similarities.append(combined_similarity)
            prob_sims.append(float(prob_similarity))
            
        return np.array(similarities), prob_sims
    
    def find_most_representative_tree(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[int, DecisionTreeClassifier]:
        self.similarity_scores, self.prob_sims = self.calculate_tree_similarities(X_test, y_test)
        self.mrt_index = np.argmax(self.similarity_scores)
        self.mrt_model = self.rf.estimators_[self.mrt_index]
        return self.mrt_index, self.mrt_model
    
    def get_tree_statistics(self) -> dict:
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
        if self.mrt_model is None:
            raise ValueError("Must call find_most_representative_tree first")
            
        ensemble_pred = self.rf.predict(X_test)
        ensemble_proba = self.rf.predict_proba(X_test)[:, 1]
        
        tree_pred = self.mrt_model.predict(X_test)
        tree_proba = self.mrt_model.predict_proba(X_test)[:, 1]

        # Row-wise probability similarity (1 / (1 + abs diff))
        prob_similarity_row = 1 / (1 + np.abs(ensemble_proba - tree_proba))
        
        comparison_df = pd.DataFrame({
            'ensemble_prediction': ensemble_pred,
            'ensemble_probability': ensemble_proba,
            'tree_prediction': tree_pred,
            'tree_probability': tree_proba,
            'prediction_match': ensemble_pred == tree_pred,
            'prob_similarity': prob_similarity_row
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
    try:
        df = pd.read_csv("data/train_data.csv")
        
        X = df.drop(columns=['chunk', 'cancer_stage', 'patient_id', 'filename', 
                             'rolloff', 'bandwidth', "skew", "zcr", 'rms'])
        y = df['cancer_stage']
        feature_names = X.columns.tolist()
        
        X_test = X.iloc[:100].values
        y_test = y.iloc[:100].values
        
        mrt = extract_mrt_from_saved_model("models/rf_model.pkl", X_test, y_test, feature_names)
        
        mrt.plot_similarity_distribution("tree_similarity_distribution.png")
        mrt.export_tree_structure(feature_names, "representative_tree.dot")

        comparison = mrt.compare_predictions(X_test)
        print(f"\nPrediction agreement: {comparison['prediction_match'].mean():.2%}")
        print(f"Probability agreement (avg): {comparison['prob_similarity'].mean():.4f}")

        mrt_extracted = copy.deepcopy(mrt)
        mrt_extracted.rf = None
        joblib.dump(mrt_extracted, "models/mrt_model.pkl")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")
