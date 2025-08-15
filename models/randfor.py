import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import GroupKFold, train_test_split
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
    
    feature_cols = df.drop(columns=['segment', 'cancer_stage', 'patient_id']).columns
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
    X = df.drop(columns=['segment', 'cancer_stage', 'patient_id'])
    y = df['cancer_stage']
    groups = df['patient_id']

    print(f"Total samples: {len(df)}")
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

        # Apply SMOTEENN to training data only
        print("Applying SMOTEENN to training data")
        smote = SMOTEENN(random_state=SEED)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"After SMOTEENN: {Counter(y_train_res)}")

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=SEED,
            n_jobs=-1
        )
        rf.fit(X_train_res, y_train_res)

        y_pred = rf.predict(X_test)
        y_pred_prob = rf.predict_proba(X_test)[:, 1]

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

    return all_reports, all_conf_matrices, fold_details, all_roc_aucs

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

# ====================== MAIN ======================
if __name__ == "__main__":
    print("Loading dataset")
    df = pd.read_csv("data/train_data.csv")
    print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")

    print("\nStep 1: Data Integrity Check")
    is_clean = verify_data_integrity(df)
    if not is_clean:
        print("\nDATA QUALITY ISSUES DETECTED! Results may be unreliable")

    print("\nStep 2: Run Random Forest Cross-Validation")
    results = run_rf_cross_validation(df)
    if results[0] is not None:
        all_reports, all_conf_matrices, fold_details, all_roc_aucs = results
        print("\nStep 3: Summarize Results")
        summarize_rf_results(all_reports, all_conf_matrices, fold_details, all_roc_aucs)
    else:
        print("\nCross-validation failed due to patient leakage!")
