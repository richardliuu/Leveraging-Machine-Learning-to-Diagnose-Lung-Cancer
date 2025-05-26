import pandas as pd 
import numpy as np 
from sklearn.model_selection import GroupKFold    
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt 
from imblearn.combine import SMOTEENN
from collections import Counter

# Load data
df = pd.read_csv("voice_features_log.csv")

# Prepare features and labels
X = df.drop(columns=['segment', 'cancer_stage', 'patient_id'])
y = df['cancer_stage']
groups = df['patient_id'].values  # Patient IDs for group splitting

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))
target_names = [str(cls) for cls in le.classes_]  # convert labels to strings

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Group K-Fold cross-validation
gkf = GroupKFold(n_splits=5)

all_reports = []
fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y_encoded, groups=groups)):
    print(f"\n--- Fold {fold + 1} ---")

    # Split data
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train_int, y_test_int = y_encoded[train_idx], y_encoded[test_idx]

    # One-hot encoding labels for training and testing
    y_train = to_categorical(y_train_int, num_classes=num_classes)
    y_test = to_categorical(y_test_int, num_classes=num_classes)

    print("Original class distribution in training set:", Counter(y_train_int))

    # Apply SMOTEENN to training data only
    smoteenn = SMOTEENN(random_state=42)
    X_train_res, y_train_res_int = smoteenn.fit_resample(X_train, y_train_int)
    y_train_res = to_categorical(y_train_res_int, num_classes=num_classes)

    print("After SMOTEENN class distribution:", Counter(y_train_res_int))

    # Split training into train/val (stratify by classes)
    # Use train_test_split for validation split inside train set
    from sklearn.model_selection import train_test_split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_res, y_train_res, test_size=0.2, stratify=y_train_res_int, random_state=42)

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

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=1,
        callbacks=[early_stopping, lr_schedule]
    )

    # Predict on test data
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Metrics and report
    print("Classification report for fold", fold + 1)
    report = classification_report(y_test_int, y_pred, target_names=target_names)
    print(report)
    all_reports.append(classification_report(y_test_int, y_pred, target_names=target_names, output_dict=True))

    fold_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}")
    fold_accuracies.append(fold_acc)

    # Confusion matrix plot
    c_matrix = confusion_matrix(y_test_int, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fold {fold + 1}")
    plt.show()

# Aggregate metrics
avg_accuracy = np.mean(fold_accuracies)
print(f"\nAverage Accuracy across folds: {avg_accuracy:.4f}")

# Aggregate classification report averages (macro avg)
def average_reports(reports, metric='f1-score'):
    vals = []
    for rep in reports:
        if 'macro avg' in rep:
            vals.append(rep['macro avg'][metric])
    return np.mean(vals)

avg_f1 = average_reports(all_reports, 'f1-score')
print(f"Average Macro F1-score across folds: {avg_f1:.4f}")
