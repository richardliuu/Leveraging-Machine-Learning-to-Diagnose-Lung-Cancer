import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("voice_features_log.csv")

# --- CONVERT TO BINARY LABELS ---
# Healthy (stage 0) -> 0, Cancer (stages 1-4) -> 1
df['binary_label'] = df['cancer_stage'].apply(lambda x: 0 if x == 0 else 1)

# Get one binary label per patient
patient_labels = df.groupby('patient_id')['binary_label'].first()

# Filter only patients from both classes
valid_patients = patient_labels[patient_labels.isin([0, 1])]

# Patient-level stratified split
train_pids, temp_pids = train_test_split(
    valid_patients.index,
    test_size=0.3,
    stratify=valid_patients,
    random_state=42
)

val_pids, test_pids = train_test_split(
    temp_pids,
    test_size=0.5,
    random_state=42
)

# Masks for each split
train_mask = df['patient_id'].isin(train_pids)
val_mask = df['patient_id'].isin(val_pids)
test_mask = df['patient_id'].isin(test_pids)

# Feature and label extraction
X_train_raw = df[train_mask].drop(columns=['segment', 'cancer_stage', 'patient_id', 'binary_label'])
y_train = df[train_mask]['binary_label']
X_val_raw = df[val_mask].drop(columns=['segment', 'cancer_stage', 'patient_id', 'binary_label'])
y_val = df[val_mask]['binary_label']
X_test_raw = df[test_mask].drop(columns=['segment', 'cancer_stage', 'patient_id', 'binary_label'])
y_test = df[test_mask]['binary_label']

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

# SMOTEENN resampling
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

print("Train class counts before SMOTEENN:", Counter(y_train))
print("Train class counts after SMOTEENN:", Counter(y_train_resampled))

# Class weight for XGBoost (useful for imbalanced classes)
class_counts = Counter(y_train_resampled)
scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    eval_metric='logloss',
    use_label_encoder=False,
    early_stopping_rounds=10,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb.fit(X_train_resampled, y_train_resampled, eval_set=[(X_val, y_val)], verbose=True)

# Evaluate on test set
y_pred = xgb.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Cancer']).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# --- SHAP EXPLANATIONS ---
explainer = shap.Explainer(xgb, X_train_resampled)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test_raw, feature_names=X_test_raw.columns)
