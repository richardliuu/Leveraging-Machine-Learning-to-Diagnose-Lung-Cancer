import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import Counter

# ============ Update MFCCS.npy to match the new binary classification models ============= 
# ============ Includes remaking the data ===================


# Load data from .npy
data = np.load('all_mfccs.npy', allow_pickle=True)

X = []
y = []
groups = []

for mfcc, label, patient_id in data:
    if mfcc.shape[0] >= 60:
        mfcc = mfcc[:60]
    else:
        pad_width = 60 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))

    X.append(mfcc)
    y.append(label)
    groups.append(patient_id)

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

# Normalize X
X = X / np.max(np.abs(X))

# Add channel dimension for CNN (samples, 60, 13, 1)
X = X[..., np.newaxis]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# Set up GroupKFold
k = 5
gkf = GroupKFold(n_splits=k)

all_reports = []
all_conf_matrices = []

for fold, (train_index, test_index) in enumerate(gkf.split(X, y_encoded, groups=groups)):
    print(f"\n--- Fold {fold+1}/{k} ---")

    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold_int, y_test_fold_int = y_encoded[train_index], y_encoded[test_index]
    groups_train, groups_test = groups[train_index], groups[test_index]

    y_train_fold = to_categorical(y_train_fold_int, num_classes=num_classes)
    y_test_fold = to_categorical(y_test_fold_int, num_classes=num_classes)

    # Further split train fold into train + validation
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_fold, y_train_fold,
        test_size=0.1,
        stratify=y_train_fold_int,
        random_state=42
    )

    # Build CNN model fresh each fold
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(60, 13, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

    # Train model
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val_final, y_val_final),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate on test fold
    y_pred_prob = model.predict(X_test_fold)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = y_test_fold_int

    target_names = [f"Stage {cls}" for cls in encoder.classes_]
    print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

    c_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    print("Confusion Matrix:\n", c_matrix)

    all_reports.append(classification_report(y_true_classes, y_pred_classes, output_dict=True))
    all_conf_matrices.append(c_matrix)

# Average accuracy across folds
avg_accuracy = np.mean([report['accuracy'] for report in all_reports])
print(f"\nAverage accuracy over {k} folds: {avg_accuracy:.4f}")

# Average confusion matrix across folds
avg_conf_matrix = np.mean(all_conf_matrices, axis=0)
print("Average Confusion Matrix over all folds:\n", np.round(avg_conf_matrix).astype(int))

# Plot training/validation accuracy & loss from last fold
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve (Last Fold)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve (Last Fold)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
