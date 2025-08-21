import pandas as pd
import numpy as np
import random
import os
import joblib
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GroupKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

os.environ['PYTHONHASHSEED'] = '42'
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def verify_data_integrity(df):
    print("=== DATA INTEGRITY CHECKS ===")
    
    feature_cols = df.drop(columns=['cancer_stage', 'patient_id', 'chunk']).columns
    duplicates = df.duplicated(subset=feature_cols)
    print(f"Duplicate feature rows: {duplicates.sum()}")
    
    if duplicates.sum() > 0:
        print("WARNING: Duplicate samples found!")
        dup_rows = df[duplicates]
        print(f"Example duplicate patients: {dup_rows['patient_id'].unique()[:5]}")
    else:
        print("No duplicate feature rows found")
    
    patient_labels = df.groupby('patient_id')['cancer_stage'].nunique()
    inconsistent_patients = patient_labels[patient_labels > 1]
    if len(inconsistent_patients) > 0:
        print(f"WARNING: {len(inconsistent_patients)} patients have inconsistent labels!")
        print("Inconsistent patients:", inconsistent_patients.index.tolist()[:10])
    else:
        print("All patients have consistent labels")
    
    print(f"\nOverall class distribution:")
    class_counts = df['cancer_stage'].value_counts()
    print(class_counts)
    print(f"Class ratio: {class_counts.iloc[0]/class_counts.iloc[1]:.2f}:1")
    
    return duplicates.sum() == 0 and len(inconsistent_patients) == 0

def run_rf_cross_validation(df):
    # Prepare features and labels
    X = df.drop(columns=['chunk', 'cancer_stage', 'patient_id', 'filename', 'rolloff', 'bandwidth', "skew", "zcr", 'rms'])
    y = df['cancer_stage']
    groups = df['patient_id']

    feature_cols = X.columns.tolist()

    print(f"Total samples: {len(df)}")
    
    # Incorrect, inflated by the u1, u2 file format
    print(f"Total patients: {df['patient_id'].nunique()}")
    print(f"Features: {X.shape[1]}")

    group_kfold = GroupKFold(n_splits=4)

    all_reports = []
    all_conf_matrices = []
    all_roc_aucs = []
    fold_details = []

    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
        print(f"\n{'='*50}\nFOLD {fold+1}/4\n{'='*50}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Verify patient leakage
        train_patients = set(df.iloc[train_idx]['patient_id'])
        test_patients = set(df.iloc[test_idx]['patient_id'])
        overlap = train_patients.intersection(test_patients)
        if overlap:
            print(f"CRITICAL: Patient leakage detected! {overlap}")
            return None, None

        print(f"Train: {len(train_patients)} patients, {len(X_train)} samples")
        print(f"Test:  {len(test_patients)} patients, {len(X_test)} samples")

        # Train Random Forest
        rf = RandomForestClassifier(
            criterion="log_loss",
            n_estimators=200,
            max_depth=5,
            max_features=None,
            min_samples_split=12,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=SEED, 
            n_jobs=-1, 
        )
        rf.fit(X_train, y_train)
    
        y_pred = rf.predict(X_test)
        y_pred_prob = rf.predict_proba(X_test)[:, 1]

        np.set_printoptions(threshold=np.inf)

        probs = rf.predict_proba(X_test)
        print(y_pred_prob)

        results_df = X_test.copy()
        results_df['true_label'] = y_test.values
        results_df['predicted_label'] = y_pred
        results_df['c1_prob'] = y_pred_prob

        training_data = pd.read_csv("data/jitter_shimmerlog.csv")
        results_df['patient_id'] = training_data.iloc[test_idx]['patient_id'].values
        results_df['chunk'] = training_data.iloc[test_idx]['chunk'].values
        results_df.to_csv('data/rf_surrogate_data.csv', index=False)

        """
        mis_idx = np.where((y_test == 0) & (y_pred == 1))[0]
        misclassified_samples = X.iloc[mis_idx]

        print(misclassified_samples)

        misclassified_samples.to_csv("data/randomforest_incorrect.csv",index=False)
        """

        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        c_matrix = confusion_matrix(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
        except:
            auc = np.nan

        print(f"\nFold {fold+1} Results:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(c_matrix)
        print(f"ROC AUC Score: {auc:.4f}")

        all_reports.append(report)
        all_conf_matrices.append(c_matrix)
        all_roc_aucs.append(auc)
        fold_details.append({
            'fold': fold+1,
            'train_patients': len(train_patients),
            'test_patients': len(test_patients),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': report['accuracy']
        })

        individual_tree = rf.estimators_[0]

        print(export_graphviz(individual_tree, feature_names=feature_cols))

        X_test_df = pd.DataFrame(X_test, columns=feature_cols)
        X_explain = X_test_df.iloc[:]
        X_explain_np = X_explain.to_numpy()

        # Background for SHAP
        background = X_test_df.sample(n=20, random_state=42).to_numpy()
        explainer = shap.TreeExplainer(rf, background)
        shap_values = explainer.shap_values(X_explain_np)

        # If output is a 3D array: (samples, features, classes)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            print("SHAP values shape (3D):", shap_values.shape)
            class_index = 1  
            shap_vals_to_plot = shap_values[:, :, class_index]
        else:
            shap_vals_to_plot = shap_values[1] 

        # Confirm shape match
        assert shap_vals_to_plot.shape == X_explain_np.shape, \
            f"SHAP values shape {shap_vals_to_plot.shape} != input shape {X_explain_np.shape}"

        shap.summary_plot(shap_vals_to_plot, X_explain, feature_names=feature_cols)

    return all_reports, all_conf_matrices, fold_details, all_roc_aucs, rf

# ====================== SUMMARY ======================
def summarize_rf_results(all_reports, all_conf_matrices, fold_details, all_roc_aucs):
    print(f"\n{'='*60}\nCROSS-VALIDATION SUMMARY\n{'='*60}")

    accuracies = [r['accuracy'] for r in all_reports]
    print("Per-fold results:")
    for i, (acc, details) in enumerate(zip(accuracies, fold_details)):
        print(f"Fold {i+1}: {acc:.4f} accuracy "
              f"({details['test_patients']} patients, {details['test_samples']} samples)")

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"\nOverall Performance:")
    print(f"Mean Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Min Accuracy: {min(accuracies):.4f}")
    print(f"Max Accuracy: {max(accuracies):.4f}")

    class_0_f1 = [r['0']['f1-score'] for r in all_reports]
    class_1_f1 = [r['1']['f1-score'] for r in all_reports]
    print(f"\nClass-wise F1-scores:")
    print(f"Class 0: {np.mean(class_0_f1):.4f} ± {np.std(class_0_f1):.4f}")
    print(f"Class 1: {np.mean(class_1_f1):.4f} ± {np.std(class_1_f1):.4f}")

    avg_conf_matrix = np.mean(all_conf_matrices, axis=0)
    print(f"\nAverage Confusion Matrix:")

    print(np.round(avg_conf_matrix).astype(int))

if __name__ == "__main__":
    print("Loading dataset")
    df = pd.read_csv("data/jitter_shimmerlog.csv")
    print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")

    print("\nStep 1: Data Integrity Check")
    is_clean = verify_data_integrity(df)
    if not is_clean:
        print("\nDATA QUALITY ISSUES DETECTED! Results may be unreliable")

    print("\nStep 2: Run Random Forest Cross-Validation")
    results = run_rf_cross_validation(df)
    if results[0] is not None:
        all_reports, all_conf_matrices, fold_details, all_roc_aucs, rf = results
        print("\nStep 3: Summarize Results")
        summarize_rf_results(all_reports, all_conf_matrices, fold_details, all_roc_aucs)
    else:
        print("\nCross-validation failed due to patient leakage!")

    #joblib.dump(rf, "models/rf_model.pkl")
    #print("Random Forest saved as rf_model.pkl")
