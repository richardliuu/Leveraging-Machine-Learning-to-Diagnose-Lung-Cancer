import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.combine import SMOTEENN
from collections import Counter
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("updated_file.csv")

# Limit max samples per patient (optional)
MAX_SAMPLES_PER_PATIENT = None  

# Robustness can be heavily improved on 

if MAX_SAMPLES_PER_PATIENT is not None:
    def limit_samples(group):
        return group.sample(n=min(len(group), MAX_SAMPLES_PER_PATIENT), random_state=42)
    df = df.groupby('patient_id').apply(limit_samples).reset_index(drop=True)

# Prepare patient-level info for stratification and grouping
patient_df = df.groupby('patient_id').agg({
    'cancer_stage': lambda x: x.mode()[0],  
    'segment': 'count'
}).rename(columns={'segment': 'num_samples'}).reset_index()

# Label encode patient cancer stage for stratification
le_patient = LabelEncoder()
patient_df['label_enc'] = le_patient.fit_transform(patient_df['cancer_stage'])
num_classes = len(le_patient.classes_)

print(f"Number of patients: {len(patient_df)}")
print(f"Class distribution at patient level:\n{Counter(patient_df['label_enc'])}")

# Confirm patient groups by class
class_0_patients = patient_df[patient_df['label_enc'] == 0]
class_1_patients = patient_df[patient_df['label_enc'] == 1]

print("\nClass 0 patients samples:", class_0_patients['num_samples'].values)
print("Class 1 patients samples:", class_1_patients['num_samples'].values)

k = 4  

assert len(class_0_patients) >= k, "Expected 4 patients in class 0"
assert len(class_1_patients) >= k, "Expected 4 patients in class 1"

# Create fold patient groups explicitly
folds_patient_ids = []
for i in range(k):
    fold_patients = pd.concat([
        class_0_patients.iloc[[i]],
        class_1_patients.iloc[[i]]
    ])
    folds_patient_ids.append(fold_patients['patient_id'].values)

print("\nPatient IDs per fold:")
for i, pids in enumerate(folds_patient_ids):
    print(f"Fold {i+1}: {pids}")

# Prepare feature matrix and labels
X = df.drop(columns=['segment', 'cancer_stage', 'patient_id'])
y = df['cancer_stage']
y_encoded = le_patient.transform(df['cancer_stage']) 

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

all_reports = []
all_conf_matrices = []

for fold in range(k):
    print(f"\n--- Fold {fold+1}/{k} ---")

    # Patient IDs in test fold
    test_patient_ids = folds_patient_ids[fold]
    # Train patients = all others
    train_patient_ids = np.setdiff1d(patient_df['patient_id'], test_patient_ids)

    # Get train/test indices from df based on patient_id
    train_idx = df['patient_id'].isin(train_patient_ids)
    test_idx = df['patient_id'].isin(test_patient_ids)

    X_train_fold = X_scaled[train_idx]
    y_train_fold_int = y_encoded[train_idx]

    X_test_fold = X_scaled[test_idx]
    y_test_fold_int = y_encoded[test_idx]

    print("Train class distribution:", Counter(y_train_fold_int))
    print("Test class distribution:", Counter(y_test_fold_int))

    # Apply SMOTEENN only on training data
    smote = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled_int = smote.fit_resample(X_train_fold, y_train_fold_int)
    y_train_resampled = to_categorical(y_train_resampled_int, num_classes=num_classes)

    print("After SMOTEENN:", Counter(y_train_resampled_int))

    # One-hot encode test labels
    y_test_fold = to_categorical(y_test_fold_int, num_classes=num_classes)

    # Split resampled training data into train and validation sets
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.1, stratify=y_train_resampled_int, random_state=42)

    # Build model fresh for each fold
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train_final.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val_final, y_val_final),
        epochs=50,
        batch_size=16,
        callbacks=[early_stopping, lr_schedule],
        verbose=1
    )

    y_pred_prob = model.predict(X_test_fold)
    y_pred = np.argmax(y_pred_prob, axis=1)

    target_names = [str(cls) for cls in le_patient.classes_]
    print(classification_report(y_test_fold_int, y_pred, target_names=target_names))

    c_matrix = confusion_matrix(y_test_fold_int, y_pred)
    print("Confusion Matrix:\n", c_matrix)

    all_reports.append(classification_report(y_test_fold_int, y_pred, target_names=le_patient.classes_, output_dict=True))
    all_conf_matrices.append(c_matrix)

# Summarize results
avg_accuracy = np.mean([report['accuracy'] for report in all_reports])
print(f"\nAverage accuracy over {k} folds: {avg_accuracy:.4f}")

avg_conf_matrix = np.mean(all_conf_matrices, axis=0)
print("Average Confusion Matrix over all folds:\n", np.round(avg_conf_matrix).astype(int))

# Plot last fold history
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve (last fold)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve (last fold)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
