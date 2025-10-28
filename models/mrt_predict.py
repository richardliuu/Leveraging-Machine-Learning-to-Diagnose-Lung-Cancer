import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 141
np.random.seed(SEED)

# -------------------------------
# Fixed MRT class
# -------------------------------
class MostRepresentativeTree:
    """
    Wraps a single chosen tree from a pretrained RF.
    Here we fix the MRT index (e.g., 158).
    """
    def __init__(self, rf_model, tree_idx=158):
        self.rf_model = rf_model
        self.mrt_model = rf_model.estimators_[tree_idx]
        self.best_tree_idx = tree_idx
        print(f"✅ Using MRT = Tree #{tree_idx}")

    def predict(self, X):
        return self.mrt_model.predict(X)

    def predict_proba(self, X):
        return self.mrt_model.predict_proba(X)


# -------------------------------
# Cross-validation using fixed MRT
# -------------------------------
def run_mrt_cross_validation(df, rf_model_path, mrt_idx=158):
    # Load pretrained RF
    rf = joblib.load(rf_model_path)
    mrt = MostRepresentativeTree(rf, tree_idx=mrt_idx)

    # Features/labels/groups
    X = df.drop(columns=['chunk', 'cancer_stage', 'patient_id', 'filename', 
                         'rolloff', 'bandwidth', "skew", "zcr", 'rms'])
    y = df['cancer_stage']
    groups = df['patient_id']

    print(f"Total samples: {len(df)}")
    print(f"Total patients: {df['patient_id'].nunique()}")
    print(f"Features: {X.shape[1]}")

    group_kfold = StratifiedGroupKFold(n_splits=4)

    all_reports, all_conf_matrices, all_roc_aucs, fold_details = [], [], [], []
    all_fold_results = pd.DataFrame()  # collect per-fold predictions

    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
        print(f"\n{'='*50}\nMRT FOLD {fold+1}/4\n{'='*50}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # --- Check patient leakage ---
        train_patients = set(df.iloc[train_idx]['patient_id'])
        test_patients = set(df.iloc[test_idx]['patient_id'])
        overlap = train_patients.intersection(test_patients)
        if overlap:
            print(f"CRITICAL: Patient leakage detected! {overlap}")
            return None, None, None, None, None

        print(f"Train: {len(train_patients)} patients, {len(X_train)} samples")
        print(f"Test:  {len(test_patients)} patients, {len(X_test)} samples")

        # --- Predict with fixed MRT ---
        y_train_pred = mrt.predict(X_train.values)
        y_train_prob = mrt.predict_proba(X_train.values)[:, 1]
        train_accuracy = (y_train_pred == y_train).mean()

        y_pred = mrt.predict(X_test.values)
        y_pred_prob = mrt.predict_proba(X_test.values)[:, 1]

        # --- Metrics ---
        report = classification_report(y_test, y_pred, output_dict=True)
        c_matrix = confusion_matrix(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
        except:
            auc = np.nan

        print(f"\nFold {fold+1} Results:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {report['accuracy']:.4f}")
        print(f"Overfitting Gap: {(train_accuracy - report['accuracy']):.4f}")
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(c_matrix)
        print(f"ROC AUC Score: {auc:.4f}")

        # Save test results for analysis
        fold_results = X_test.copy()
        fold_results['true_label'] = y_test.values
        fold_results['mrt_predicted_label'] = y_pred
        fold_results['mrt_probability'] = y_pred_prob
        fold_results['patient_id'] = df.iloc[test_idx]['patient_id'].values
        fold_results['chunk'] = df.iloc[test_idx]['chunk'].values
        fold_results['fold'] = fold + 1   # track which fold this came from

        all_fold_results = pd.concat([all_fold_results, fold_results], ignore_index=True)

        # Collect metrics
        all_reports.append(report)
        all_conf_matrices.append(c_matrix)
        all_roc_aucs.append(auc)
        fold_details.append({
            'fold': fold+1,
            'train_patients': len(train_patients),
            'test_patients': len(test_patients),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': train_accuracy,
            'test_accuracy': report['accuracy'],
            'overfitting_gap': train_accuracy - report['accuracy']
        })

    return all_reports, all_conf_matrices, fold_details, all_roc_aucs, all_fold_results


# -------------------------------
# Summarization
# -------------------------------
def summarize_results(all_reports, all_conf_matrices, fold_details, all_roc_aucs):
    print("\n" + "="*50)
    print("MRT CROSS-VALIDATION SUMMARY (Fixed Index)")
    print("="*50)

    accs = [fd['test_accuracy'] for fd in fold_details]
    train_accs = [fd['train_accuracy'] for fd in fold_details]

    print(f"Average Training Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"Average Test Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Average Overfitting Gap: {np.mean([fd['overfitting_gap'] for fd in fold_details]):.4f}")
    print(f"Average ROC AUC: {np.nanmean(all_roc_aucs):.4f} ± {np.nanstd(all_roc_aucs):.4f}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/train_data.csv")
    print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")

    results = run_mrt_cross_validation(df, "models/rf_model.pkl", mrt_idx=158)

    if results[0] is not None:
        all_reports, all_conf_matrices, fold_details, all_roc_aucs, all_fold_results = results

        # Save all test predictions for analysis
        all_fold_results.to_csv("data/mrt_test_predictions.csv", index=False)
        print(f"✅ Saved MRT test predictions for all folds to data/mrt_test_predictions.csv")

        summarize_results(all_reports, all_conf_matrices, fold_details, all_roc_aucs)

        # Save the chosen MRT model
        rf = joblib.load("models/rf_model.pkl")
        mrt_final = rf.estimators_[143]
        joblib.dump(mrt_final, "models/mrt_model_idx143.pkl")
        print("✅ Saved MRT (Tree #143) as models/mrt_model_idx143.pkl")
