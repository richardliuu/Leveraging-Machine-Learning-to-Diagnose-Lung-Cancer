import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical # type: ignore 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

def verify_data_integrity_from_array(data_array):
    rows = [{'patient_id': pid, 'cancer_stage': label} for _, label, pid in data_array]
    df_meta = pd.DataFrame(rows)

    patient_labels = df_meta.groupby('patient_id')['cancer_stage'].nunique()
    inconsistent_patients = patient_labels[patient_labels > 1]

    if len(inconsistent_patients) > 0:
        print(f"WARNING: {len(inconsistent_patients)} patients have inconsistent labels!")
        print("Inconsistent patients:", inconsistent_patients.index.tolist()[:10])
    else:
        print("All patients have consistent labels")

    class_counts = df_meta['cancer_stage'].value_counts()
    print(f"\nClass distribution:\n{class_counts}")

    samples_per_patient = df_meta.groupby('patient_id').size()
    print(f"\nSamples per patient: Mean={samples_per_patient.mean():.1f}, Min={samples_per_patient.min()}, Max={samples_per_patient.max()}")

    return len(inconsistent_patients) == 0

def cross_validation(data_array):
    X, y, groups = [], [], []

    for mfcc, label, patient_id in data_array:
        mfcc = mfcc[:60] if mfcc.shape[0] >= 60 else np.pad(mfcc, ((0, 60 - mfcc.shape[0]), (0, 0)))
        X.append(mfcc)
        y.append(label)
        groups.append(patient_id)

    X = np.array(X)[..., np.newaxis]  # Shape: (N, 60, 13, 1)
    X = X / np.max(np.abs(X))
    y = np.array(y)
    groups = np.array(groups)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    k = 4
    gkf = GroupKFold(n_splits=4)

    all_reports = []
    all_conf_matrices = []
    fold_details = []
    all_histories = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups=groups)):
        print(f"\n--- Fold {fold + 1}/{k} ---")

        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold_int, y_test_fold_int = y_encoded[train_idx], y_encoded[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        y_train_fold = to_categorical(y_train_fold_int, num_classes=num_classes)
        y_test_fold = to_categorical(y_test_fold_int, num_classes=num_classes)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        val_train_idx, val_idx = next(gss.split(X_train_fold, y_train_fold_int, groups=groups_train))

        X_train_final, X_val_final = X_train_fold[val_train_idx], X_train_fold[val_idx]
        y_train_final, y_val_final = y_train_fold[val_train_idx], y_train_fold[val_idx]

        train_patients = set(groups_train[val_train_idx])
        val_patients = set(groups_train[val_idx])
        test_patients = set(groups_test)

        overlap = (train_patients & test_patients) | (val_patients & test_patients)
        if overlap:
            print("CRITICAL: Patient leakage detected!")
            print("Overlapping patients:", list(overlap)[:5])
            return None, None, None, None

        print("No patient overlap between train/val/test")
        print(f"Train: {len(train_patients)} patients, {len(X_train_final)} samples")
        print(f"Val:   {len(val_patients)} patients, {len(X_val_final)} samples")
        print(f"Test:  {len(test_patients)} patients, {len(X_test_fold)} samples")

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(60, 13, 1)),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val_final, y_val_final),
            epochs=50, batch_size=32,
            callbacks=[early_stopping, lr_schedule],
            verbose=1
        )

        y_pred_prob = model.predict(X_test_fold)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true_classes = y_test_fold_int

        report = classification_report(y_true_classes, y_pred, output_dict=True)
        print(classification_report(y_true_classes, y_pred))

        c_matrix = confusion_matrix(y_true_classes, y_pred)
        print("Confusion Matrix:\n", c_matrix)

        all_reports.append(report)
        all_conf_matrices.append(c_matrix)
        fold_details.append({
            'fold': fold + 1,
            'train_patients': len(train_patients),
            'test_patients': len(test_patients),
            'train_samples': len(X_train_final),
            'test_samples': len(X_test_fold),
            'accuracy': report['accuracy'],
            'epochs_trained': len(history.history['loss'])
        })
        all_histories.append(history.history)

    return all_reports, all_conf_matrices, fold_details, all_histories

def summarize_results(all_reports, all_conf_matrices, fold_details, all_histories):
    print(f"\n{'=' * 60}\nCROSS-VALIDATION SUMMARY\n{'=' * 60}")

    accuracies = [report['accuracy'] for report in all_reports]
    print("Per-fold results:")
    for i, (acc, details) in enumerate(zip(accuracies, fold_details)):
        print(f"Fold {i + 1}: {acc:.4f} accuracy ({details['test_patients']} patients, {details['test_samples']} samples, {details['epochs_trained']} epochs)")

    for i, history in enumerate(all_histories):
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Fold {i + 1} Performance", fontsize=14)

        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Train Acc')
        plt.plot(history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"\nOverall Performance:\nMean Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Min Accuracy:  {min(accuracies):.4f}\nMax Accuracy:  {max(accuracies):.4f}")

    class_0_f1 = [report['0']['f1-score'] for report in all_reports]
    class_1_f1 = [report['1']['f1-score'] for report in all_reports]
    print(f"\nClass-wise F1-scores:\nClass 0: {np.mean(class_0_f1):.4f} ± {np.std(class_0_f1):.4f}\nClass 1: {np.mean(class_1_f1):.4f} ± {np.std(class_1_f1):.4f}")

    avg_conf_matrix = np.mean(all_conf_matrices, axis=0)
    print(f"\nAverage Confusion Matrix:\n{np.round(avg_conf_matrix).astype(int)}")

    print("\nModel Evaluation:")
    if avg_accuracy > 0.95:
        print("VERY HIGH accuracy")
    elif avg_accuracy > 0.85:
        print("HIGH accuracy")
    elif avg_accuracy > 0.7:
        print("Good accuracy")
    else:
        print("Moderate accuracy (Model tuning)")

    if std_accuracy > 0.1:
        print("HIGH variance across folds - results may not be stable")
    else:
        print("Low variance across folds - stable results")

if __name__ == "__main__":
    df = np.load('models/binary_mfccs.npy', allow_pickle=True)
    print("\nStep 1: Data Integrity Check")
    is_clean = verify_data_integrity_from_array(df)

    if not is_clean:
        print("\nDATA QUALITY ISSUES DETECTED!\nResults may be unreliable")

    print("\nStep 2: Cross-Validation")
    results = cross_validation(df)

    if results[0] is not None:
        print("\nStep 3: Results Summary")
        summarize_results(*results)
    else:
        print("\nCross-validation failed due to data leakage!")
