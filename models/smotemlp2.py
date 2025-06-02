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

# Prepare features and target
X = df.drop(columns=['segment', 'cancer_stage', 'patient_id'])
y = df['cancer_stage']  

# Label encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-fold Cross Validation parameters
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Store metrics and confusion matrices for each fold
all_reports = []
all_conf_matrices = []

for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y_encoded)):
    print(f"\n--- Fold {fold+1}/{k} ---")

    # Split data
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold_int, y_test_fold_int = y_encoded[train_index], y_encoded[test_index]

    # Apply SMOTEENN to training data only
    smote = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled_int = smote.fit_resample(X_train_fold, y_train_fold_int)
    y_train_resampled = to_categorical(y_train_resampled_int, num_classes=num_classes)

    print("Before SMOTEENN:", Counter(y_train_fold_int))
    print("After SMOTEENN:", Counter(y_train_resampled_int))

    # One-hot encode test labels
    y_test_fold = to_categorical(y_test_fold_int, num_classes=num_classes)

    # Split resampled training data into train and validation sets (e.g., 10% val)
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.1, stratify=y_train_resampled_int, random_state=42)

    # Build the model (fresh for each fold)
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

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

    # Train model
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val_final, y_val_final),
        epochs=50,
        batch_size=16,
        callbacks=[early_stopping, lr_schedule],
        verbose=1
    )

    # Evaluate model on test fold
    y_pred_prob = model.predict(X_test_fold)
    y_pred = np.argmax(y_pred_prob, axis=1)

    target_names = [str(cls) for cls in le.classes_]
    print(classification_report(y_test_fold_int, y_pred, target_names=target_names))

    c_matrix = confusion_matrix(y_test_fold_int, y_pred)
    print("Confusion Matrix:\n", c_matrix)

    all_reports.append(classification_report(y_test_fold_int, y_pred, target_names=le.classes_, output_dict=True))
    all_conf_matrices.append(c_matrix)

# Summarize results over all folds
avg_accuracy = np.mean([report['accuracy'] for report in all_reports])
print(f"\nAverage accuracy over {k} folds: {avg_accuracy:.4f}")

# Optional: average confusion matrix
avg_conf_matrix = np.mean(all_conf_matrices, axis=0)
print("Average Confusion Matrix over all folds:\n", np.round(avg_conf_matrix).astype(int))

# Plot average training and validation loss and accuracy of last fold (example)
