import pandas as pd
import numpy as np
import joblib
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Optional, Tuple

# =================== MRT CLASS ===================
class MostRepresentativeTree:
    def __init__(self, random_forest: RandomForestClassifier):
        self.rf = random_forest
        self.mrt_index = None
        self.mrt_model = None
        self.similarity_scores = None

    def calculate_tree_similarities(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        ensemble_pred = self.rf.predict(X_test)
        ensemble_pred_proba = self.rf.predict_proba(X_test)
        similarities, prob_sims = [], []

        for tree in self.rf.estimators_:
            tree_pred = tree.predict(X_test)
            tree_pred_proba = tree.predict_proba(X_test)
            accuracy_similarity = accuracy_score(ensemble_pred, tree_pred)
            prob_similarity = 1 / (1 + mean_squared_error(ensemble_pred_proba, tree_pred_proba))
            similarities.append((accuracy_similarity + prob_similarity)/2)
            prob_sims.append(float(prob_similarity))
        return np.array(similarities), prob_sims

    def find_most_representative_tree(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[int, RandomForestClassifier]:
        self.similarity_scores, self.prob_sims = self.calculate_tree_similarities(X_test, y_test)
        self.mrt_index = np.argmax(self.similarity_scores)
        self.mrt_model = self.rf.estimators_[self.mrt_index]
        return self.mrt_index, self.mrt_model

    def compare_predictions(self, X_test: np.ndarray) -> pd.DataFrame:
        if self.mrt_model is None:
            raise ValueError("Call find_most_representative_tree first")
        ensemble_pred = self.rf.predict(X_test)
        ensemble_proba = self.rf.predict_proba(X_test)[:, 1]
        tree_pred = self.mrt_model.predict(X_test)
        tree_proba = self.mrt_model.predict_proba(X_test)[:, 1]
        prob_similarity_row = 1 / (1 + np.abs(ensemble_proba - tree_proba))
        return pd.DataFrame({
            'ensemble_prediction': ensemble_pred,
            'ensemble_probability': ensemble_proba,
            'tree_prediction': tree_pred,
            'tree_probability': tree_proba,
            'prediction_match': ensemble_pred == tree_pred,
            'prob_similarity': prob_similarity_row
        })

# =================== SCRIPT ===================
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/train_data.csv")
    feature_cols = df.drop(columns=['chunk', 'cancer_stage', 'patient_id', 'filename', 'rolloff', 'bandwidth', "skew", "zcr", 'rms']).columns.tolist()
    X = df[feature_cols].values
    y = df['cancer_stage'].values

    # Load ensemble model
    rf_model = joblib.load("models/rf2_model.pkl")

    # Split: same as cross-validation split
    # Here we'll just take first 100 samples for demonstration
    X_test = X[:100]
    y_test = y[:100]
    test_df = df.iloc[:100].copy()
    test_df['sample_id'] = range(len(test_df))  # Add sample ID

    # Extract MRT
    mrt_extractor = MostRepresentativeTree(rf_model)
    mrt_extractor.find_most_representative_tree(X_test, y_test)

    # Compare predictions
    comparison = mrt_extractor.compare_predictions(X_test)
    combined_df = test_df[['patient_id', 'true_label']].copy() if 'true_label' in test_df else test_df[['patient_id']].copy()
    combined_df['true_label'] = y_test
    combined_df = pd.concat([combined_df, comparison], axis=1)

    # Save all MRT predictions to CSV
    combined_df.to_csv("mrt_predictions.csv", index=False)
    print("Saved MRT predictions with probabilities, true label, patient_id, and sample_id to mrt_predictions.csv")

    # Automatically select showcase examples
    showcase = {}
    # Correct & confident (high prob similarity)
    correct_confident = combined_df[(combined_df['prediction_match']==True)].sort_values('prob_similarity', ascending=False)
    showcase['correct_confident'] = correct_confident.iloc[0]
    # Borderline (ensemble_prob ~0.5)
    borderline = combined_df[(combined_df['prediction_match']==True) & 
                             (combined_df['ensemble_probability'].between(0.45, 0.55))]
    if not borderline.empty:
        showcase['borderline'] = borderline.iloc[0]
    # Wrong (prediction mismatch)
    wrong = combined_df[(combined_df['prediction_match']==False)]
    if not wrong.empty:
        showcase['wrong'] = wrong.iloc[0]
    # Feature variability (max deviation from median of test set)
    median_features = np.median(X_test, axis=0)
    feature_diff = np.abs(X_test - median_features).sum(axis=1)
    combined_df['feature_diff'] = feature_diff
    variability_candidates = combined_df[combined_df['prediction_match']==True]
    if not variability_candidates.empty:
        showcase['feature_variability'] = variability_candidates.loc[variability_candidates['feature_diff'].idxmax()]

    print("\n=== SHOWCASE EXAMPLES ===")
    for k,v in showcase.items():
        print(f"\n{k.upper()}:\n{v.to_dict()}")
