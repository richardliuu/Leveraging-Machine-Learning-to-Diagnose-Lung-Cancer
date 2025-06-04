import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from collections import Counter

# ======== IMPORTANT =======
# That the methodology of the CNN model (data leakage, kfold cv) are the same as SMOTEMLP.py
# SMOTEMLP.py (most accurate in every way model)

def verify_data_integrity(df):
    feature_cols = df.drop(columns=['segment', 'cancer_stage', 'patient_id']).columns
    duplicates = df.duplicated(subset=feature_cols)
    print(f"Duplicate feature rows: {duplicates.sum()}")

    if duplicates.sum() > 0:
        print("WARNING: Duplicate samples found - could inflate performance!")
        # Show some examples
        dup_rows = df[duplicates]
        print(f"Example duplicate patients: {dup_rows['patient_id'].unique()[:5]}")
    else:
        print("No duplicate feature rows found")
    
    # Check patient-label consistency
    patient_labels = df.groupby('patient_id')['cancer_stage'].nunique()
    inconsistent_patients = patient_labels[patient_labels > 1]
    
    if len(inconsistent_patients) > 0:
        print(f"WARNING: {len(inconsistent_patients)} patients have inconsistent labels!")
        print("Inconsistent patients:", inconsistent_patients.index.tolist()[:10])
        # Show details for first inconsistent patient
        if len(inconsistent_patients) > 0:
            first_patient = inconsistent_patients.index[0]
            patient_data = df[df['patient_id'] == first_patient][['patient_id', 'cancer_stage']]
            print(f"Example - Patient {first_patient}:")
            print(patient_data['cancer_stage'].value_counts())
    else:
        print("All patients have consistent labels")
    
    # Check class balance
    print(f"\nOverall class distribution:")
    class_counts = df['cancer_stage'].value_counts()
    print(class_counts)
    print(f"Class ratio: {class_counts.iloc[0]/class_counts.iloc[1]:.2f}:1")
    
    # Check samples per patient
    samples_per_patient = df.groupby('patient_id').size()
    print(f"\nSamples per patient statistics:")
    print(f"Mean: {samples_per_patient.mean():.1f}, Std: {samples_per_patient.std():.1f}")
    print(f"Min: {samples_per_patient.min()}, Max: {samples_per_patient.max()}")
    print(f"Median: {samples_per_patient.median():.1f}")
    
    # Patient-level class distribution
    patient_df = df.groupby('patient_id').agg({
        'cancer_stage': lambda x: x.mode()[0]
    }).reset_index()
    print(f"\nPatient-level class distribution:")
    patient_class_counts = patient_df['cancer_stage'].value_counts()
    print(patient_class_counts)
    
    return duplicates.sum() == 0 and len(inconsistent_patients) == 0

def cross_validation(df):
    global history 

    X = []
    y = []
    groups = []

    for mfcc, label, patient_id in df:
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
    y_test_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    y_categorical = to_categorical(y_test_encoded, num_classes=num_classes)

    # Set up GroupKFold
    k = 4
    gkf = GroupKFold(n_splits=k)

    all_reports = []
    all_conf_matrices = []
    fold_details = []
    all_histories = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_test_encoded, groups=groups)):
        print(f"\n--- Fold {fold+1}/{k} ---")

        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold_int, y_test_fold_int = y_test_encoded[train_idx], y_test_encoded[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        y_train_fold = to_categorical(y_train_fold_int, num_classes=num_classes)
        y_test_fold = to_categorical(y_test_fold_int, num_classes=num_classes)

        # Further split train fold into train + validation
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train_fold, y_train_fold,
            test_size=0.1,
            stratify=y_train_fold_int,
            random_state=42
        )

# First verify no data leakage 
# Verify no patient leakage
        train_patients = set(df.iloc[train_idx]['patient_id'])
        test_patients = set(df.iloc[test_idx]['patient_id'])
        
        overlap = train_patients.intersection(test_patients)
        if overlap:
            print(f"CRITICAL: Patient leakage detected!")
            print(f"Overlapping patients: {list(overlap)[:5]}...")
            return None, None
        else:
            print("No patient overlap between train/test")
        
        print(f"Train: {len(train_patients)} patients, {len(X_train_fold)} samples")
        print(f"Test:  {len(test_patients)} patients, {len(X_test_fold)} samples")

        
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
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
            )
        
        lr_schedule = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.1, 
            patience=3, 
            verbose=1)

        # Train model
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val_final, y_val_final),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, lr_schedule],
            verbose=1
        )

        # Evaluate on test fold
        y_pred_prob = model.predict(X_test_fold)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true_classes = y_test_fold_int

        target_names = [f"Stage {cls}" for cls in encoder.classes_]
        report = classification_report(y_test_encoded, y_pred, target_names=target_names, output_dict=True)
        print(classification_report(y_true_classes, y_pred, target_names=target_names))

        c_matrix = confusion_matrix(y_true_classes, y_pred)
        print("Confusion Matrix:\n", c_matrix)

        all_reports.append(classification_report(y_true_classes, y_pred, output_dict=True))
        all_conf_matrices.append(c_matrix)

        fold_details.append({
            'fold': fold + 1,
            'train_patients': len(train_patients),
            'test_patients': len(test_patients),
            'train_samples': len(X_train_fold),
            'test_samples': len(X_test_fold),
            'accuracy': report['accuracy'],
            'epochs_trained': len(history.history['loss'])
        })
    
    return all_reports, all_conf_matrices, fold_details, all_histories



# Average accuracy across folds
def summarize_results(all_reports, all_conf_matrices):
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    accuracies = [report['accuracy'] for report in all_reports]

    # Trying to plot epochs
    epochs_trained = [fold['epochs_trained'] for fold in fold_details]
    
    print("Per-fold results:")
    for i, (acc, details) in enumerate(zip(accuracies, fold_details)):
        print(f"Fold {i+1}: {acc:.4f} accuracy "
              f"({details['test_patients']} patients, {details['test_samples']} samples, "
              f"{details['epochs_trained']} epochs)")

    for i, history in enumerate(all_histories):
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Fold {i+1} Performance", fontsize=14)
        
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
        
    # Overall statistics
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"\nOverall Performance:")
    print(f"Mean Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Min Accuracy:  {min(accuracies):.4f}")
    print(f"Max Accuracy:  {max(accuracies):.4f}")
    
    # Class-wise performance
    class_0_f1 = [report['0']['f1-score'] for report in all_reports]
    class_1_f1 = [report['1']['f1-score'] for report in all_reports]
    
    print(f"\nClass-wise F1-scores:")
    print(f"Class 0: {np.mean(class_0_f1):.4f} ± {np.std(class_0_f1):.4f}")
    print(f"Class 1: {np.mean(class_1_f1):.4f} ± {np.std(class_1_f1):.4f}")
    
    # Average confusion matrix
    avg_conf_matrix = np.mean(all_conf_matrices, axis=0)
    print(f"\nAverage Confusion Matrix:")
    print(np.round(avg_conf_matrix).astype(int))
    
    # Performance assessment
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

    """
    avg_accuracy = np.mean([report['accuracy'] for report in all_reports])
    print(f"\nAverage accuracy over {k} folds: {avg_accuracy:.4f}")

    # Average confusion matrix across folds
    avg_conf_matrix = np.mean(all_conf_matrices, axis=0)
    print("Average Confusion Matrix over all folds:\n", np.round(avg_conf_matrix).astype(int))
    """    
    
if __name__ == "__main__":
    df = np.load('models/binary_mfccs.npy', allow_pickle=True)
    #print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")
    
    # Verify data integrity
    print("\nStep 1: Data Integrity Check")
    is_clean = verify_data_integrity(df)
    
    if not is_clean:
        print("\nDATA QUALITY ISSUES DETECTED!")
        print("Results may be unreliable")
    
    # Step 2: Run proper cross-validation
    print(f"\nCross-Validation")
    results = cross_validation(df)
    
    if results[0] is not None:  # If no critical errors
        all_reports, all_conf_matrices, fold_details, all_histories = results
        
        # Step 3: Summarize results
        print(f"\nResults Summary")
        summarize_results(all_reports, all_conf_matrices, fold_details, all_histories)    
    else:
        print(f"\nCross-validation failed due to data leakage!")
        
    
