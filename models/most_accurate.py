import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.combine import SMOTEENN
from collections import Counter
import matplotlib.pyplot as plt


# ==================== NEED TO INCLUDE MATPLOTLIB PLOTTING =======================

def verify_data_integrity(df):
    print("=== DATA INTEGRITY CHECKS ===")
    
    # Check for duplicate samples
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

def run_proper_cross_validation(df):
    global history 
    # Limit samples per patient if specified
    MAX_SAMPLES_PER_PATIENT = None  # Set to number if you want to limit, or None to disable
    
    if MAX_SAMPLES_PER_PATIENT is not None:
        print(f"Limiting to max {MAX_SAMPLES_PER_PATIENT} samples per patient...")
        def limit_samples(group):
            return group.sample(n=min(len(group), MAX_SAMPLES_PER_PATIENT), random_state=42)
        df = df.groupby('patient_id').apply(limit_samples).reset_index(drop=True)
        print(f"Dataset size after limiting: {len(df)} samples")
    
    # Prepare features and labels
    X = df.drop(columns=['segment', 'cancer_stage', 'patient_id'])
    y = df['cancer_stage']
    groups = df['patient_id']  # Group by patient ID
    
    print(f"Total samples: {len(df)}")
    print(f"Total patients: {df['patient_id'].nunique()}")
    print(f"Features: {X.shape[1]}")
    
    # Use GroupKFold for proper patient-level splitting
    group_kfold = GroupKFold(n_splits=4)
    
    all_reports = []
    all_conf_matrices = []
    fold_details = []
    
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/4")
        print(f"{'='*50}")
        
        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        
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
        
        # Fit preprocessing ONLY on training data 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)  # Only transform, don't fit!
        
        # Encode labels (fit only on training data)
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_fold)
        y_test_encoded = le.transform(y_test_fold)
        
        num_classes = len(le.classes_)
        
        train_class_dist = Counter(y_train_encoded)
        test_class_dist = Counter(y_test_encoded)
        print(f"Train class distribution: {dict(train_class_dist)}")
        print(f"Test class distribution:  {dict(test_class_dist)}")
        
        # Apply SMOTEENN only to training data
        print("Applying SMOTEENN to training data")
        smote = SMOTEENN(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)
        
        resampled_dist = Counter(y_train_resampled)
        print(f"After SMOTEENN: {dict(resampled_dist)}")
        
        # Convert to categorical
        y_train_cat = to_categorical(y_train_resampled, num_classes=num_classes)
        y_test_cat = to_categorical(y_test_encoded, num_classes=num_classes)
        
        # Create validation split from resampled training data
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train_resampled, y_train_cat, test_size=0.1, 
            stratify=y_train_resampled, random_state=42
        )
        
        print(f"Final training set: {len(X_train_final)} samples")
        print(f"Validation set: {len(X_val_final)} samples")


        """
        Here to tweak the model
        """

        # Build model (fresh for each fold)
        model = Sequential([
            Input(shape=(X_train_final.shape[1],)),
            Dense(512, activation='relu'),
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
        
        model.summary()
        
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
            verbose=1
            )
        
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val_final, y_val_final),
            epochs=50,
            batch_size=16,
            callbacks=[early_stopping, lr_schedule],
            verbose=1
        )
        
        # Predict on test set
        y_pred_prob = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Generate classification report
        target_names = [str(cls) for cls in le.classes_]
        report = classification_report(y_test_encoded, y_pred, target_names=target_names, output_dict=True)
        
        print(f"\nFold {fold+1} Results:")
        print(classification_report(y_test_encoded, y_pred, target_names=target_names))
        
        c_matrix = confusion_matrix(y_test_encoded, y_pred)
        print("Confusion Matrix:")
        print(c_matrix)
        
        # Store results
        all_reports.append(report)
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
    
    return all_reports, all_conf_matrices, fold_details

def summarize_results(all_reports, all_conf_matrices, fold_details):
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    # Per-fold summary
    accuracies = [report['accuracy'] for report in all_reports]

    # Trying to plot epochs
    epochs_trained = [report['epochs_trained'] for report in all_reports]
    
    print("Per-fold results:")
    for i, (acc, details) in enumerate(zip(accuracies, fold_details)):
        print(f"Fold {i+1}: {acc:.4f} accuracy "
              f"({details['test_patients']} patients, {details['test_samples']} samples, "
              f"{details['epochs_trained']} epochs)")
        

# The Plotting DOES NOT go over all folds (issue where it plots the same thing 4 times)
# My own trolling stuff here 
# Need to work on the plotting of the performance 

    for i, (acc, details) in enumerate(zip(epochs_trained, accuracies, fold_details, start=1)):
        plt.plot(epochs_trained, accuracies, label=f'Fold {i}')

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

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

# MAIN EXECUTION
if __name__ == "__main__":
    # Load dataset
    print("Loading dataset")
    df = pd.read_csv("models/binary_features_log.csv")
    print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")
    
    # Step 1: Verify data integrity
    print("\nStep 1: Data Integrity Check")
    is_clean = verify_data_integrity(df)
    
    if not is_clean:
        print("\nDATA QUALITY ISSUES DETECTED!")
        print("Results may be unreliable")
    
    # Step 2: Run proper cross-validation
    print(f"\nCross-Validation")
    results = run_proper_cross_validation(df)
    
    if results[0] is not None:  # If no critical errors
        all_reports, all_conf_matrices, fold_details = results
        
        # Step 3: Summarize results
        print(f"\nResults Summary")
        summarize_results(all_reports, all_conf_matrices, fold_details)    
    else:
        print(f"\nCross-validation failed due to data leakage!")
        
    
