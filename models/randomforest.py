import pandas as pd
import numpy as np
import logging
import time
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Changed from SMOTEENN
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ===== Note to clean up the prints in the terminal and go back to the final model (clean code) 

# ========= Needs GroupKFold Cross Validation =============

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

start_total = time.time()
logging.info("Script started")

try:
    # Load data
    logging.info("Loading dataset...")
    df = pd.read_csv("data/binary_features_log.csv")
    logging.info(f"Dataset shape: {df.shape}")

    # Convert labels
    logging.info("Converting cancer_stage to binary labels...")
    df['binary_label'] = df['cancer_stage'].apply(lambda x: 0 if x == 0 else 1)
    patient_labels = df.groupby('patient_id')['binary_label'].first()
    valid_patients = patient_labels[patient_labels.isin([0, 1])]

    # Patient-level stratified split
    logging.info("Splitting patient IDs...")
    train_pids, temp_pids = train_test_split(
        valid_patients.index,
        test_size=0.3,
        stratify=valid_patients,
        random_state=42
    )
    val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=42)

    # Apply masks
    logging.info("Creating data masks and extracting features...")
    train_mask = df['patient_id'].isin(train_pids)
    val_mask = df['patient_id'].isin(val_pids)
    test_mask = df['patient_id'].isin(test_pids)

    drop_cols = ['segment', 'cancer_stage', 'patient_id', 'binary_label']
    X_train_raw = df[train_mask].drop(columns=drop_cols)
    y_train = df[train_mask]['binary_label']
    X_val_raw = df[val_mask].drop(columns=drop_cols)
    y_val = df[val_mask]['binary_label']
    X_test_raw = df[test_mask].drop(columns=drop_cols)
    y_test = df[test_mask]['binary_label']

    # Normalize
    logging.info("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Resample
    logging.info("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    logging.info(f"Train class counts before: {Counter(y_train)}")
    logging.info(f"Train class counts after:  {Counter(y_train_resampled)}")

    # Class imbalance weight
    class_counts = Counter(y_train_resampled)
    scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1

    # Train model
    logging.info("Training XGBoost model...")
    xgb = XGBClassifier(
        n_estimators=100,
        eval_metric='logloss',
        use_label_encoder=False,
        early_stopping_rounds=10,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=1
    )

    train_start = time.time()
    xgb.fit(X_train_resampled, y_train_resampled, eval_set=[(X_val, y_val)], verbose=True)
    logging.info(f"Training completed in {time.time() - train_start:.2f} seconds")

    # Test prediction
    logging.info("Predicting on test set...")
    y_pred = xgb.predict(X_test)

    # Evaluation
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    logging.info("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Cancer']).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

except Exception as e:
    logging.exception("An error occurred:")

finally:
    logging.info(f"Total runtime: {time.time() - start_total:.2f} seconds")

"""
# Optional: SHAP explanation on last fold
print("\nGenerating SHAP explanations for final fold...")
explainer = shap.Explainer(xgb, X_train_resampled)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test_raw, feature_names=X_test_raw.columns)
"""