import pandas as pd
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from imblearn.combine import SMOTEENN

# Seed setting
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Fix history, report and etc. handling 
# Avoid global pipeline and create variables within the class 

# Could probably do something like handler.history and such for the data logging

class DataHandling:
    def __init__(self, data=r"data/binary_features_log.csv"):
        # Functions 
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.smote = SMOTEENN(random_state=SEED)
        self.data = data

        self.reports = []
        self.conf_matrices = []
        self.details = []
        self.history = []
        self.roc_aucs = []

        # Input and Output
        self.X = None
        self.y = None

        # Fold data
        self.X_train_fold = None
        self.y_train_fold = None
        self.X_test_fold = None
        self.y_test_fold = None

        # Scaled data
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_encoded = None
        self.y_test_encoded = None

        # Categorical
        self.y_train_cat = None
        self.y_test_cat = None

        # Final training and validation
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        # Resampled training data 
        self.X_train_resampled = None
        self.y_train_resampled = None

        # Data Information
        self.num_classes = None
        self.feature_cols = None
        self.groups = None
        self.duplicates = None
        self.patient_labels = None
        self.inconsistent_patients = None
        self.patient_data = None

        self.train_patients = None
        self.test_patients = None
        
    def load_data(self):
        data = pd.read_csv("data/binary_features_log.csv")
        self.X = data.drop(columns=['segment', 'cancer_stage', 'patient_id'])
        self.y = data['cancer_stage']
        self.groups = data['patient_id']
        self.duplicates = data.duplicated(subset=self.feature_cols)
        self.patient_labels = data.groupby('patient_id')['cancer_stage'].nunique()
        self.inconsistent_patients = self.patient_labels[self.patient_labels > 1]
        
        self.patient_data = data.groupby('patient_id').agg({
            'cancer_stage': lambda x: x.mode()[0]
        }).reset_index()
        print(f"\nPatient-level class distribution:")
        patient_class_counts = self.patient_data['cancer_stage'].value_counts()
        print(patient_class_counts)
        
        return self.duplicates.sum() == 0 and len(self.inconsistent_patients) == 0

    def split(self, X, y, data, train_idx, test_idx):
        self.X_train_fold = X.iloc[train_idx]
        self.y_train_fold = y.iloc[train_idx]
        self.X_test_fold = X.iloc[test_idx]
        self.y_test_fold = y.iloc[test_idx]

        self.train_patients = set(data.iloc[train_idx]['patient_id'])
        self.test_patients = set(data.iloc[test_idx]['patient_id'])

    def transform(self):
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_fold)
        self.X_test_scaled = self.scaler.transform(self.X_test_fold)
        self.y_train_encoded = self.encoder.fit_transform(self.y_train_fold)
        self.y_test_encoded = self.encoder.transform(self.y_test_fold)
        self.num_classes = len(self.encoder.classes_)
        self.X_train_resampled, self.y_train_resampled = self.smote.fit_resample(self.X_train_scaled, self.y_train_encoded)

    def put_to_categorical(self):
        self.y_train_cat = to_categorical(self.y_train_resampled, num_classes=self.num_classes)
        self.y_test_cat = to_categorical(self.y_test_encoded, num_classes=self.num_classes)
    
    def validation_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
        self.X_train_resampled, self.y_train_cat, test_size=0.1, 
        stratify=self.y_train_resampled, random_state=SEED
        )

    def get_data(self):
        return self.X, self.y, self.X_train_fold, self.y_train_fold, self.X_test_fold, self.y_test_fold

class LungCancerMLP:
    def __init__(self, X_train_final, num_classes):
        self.X_train_final = X_train_final
        self.num_classes = num_classes 
        self.model = self._buildmodel(X_train_final, num_classes)
        self.history = None
        self.accuracies = None
        self.epochs_trained = None
        self.target_names = None

    def _buildmodel(self, X_train_final, num_classes):
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
        
        return model

    def training_details(self, X_test_fold, X_train_fold, X_train_final, y_train_final, X_val_final, y_val_final, X, y, fold, epochs=50, batch_size=16):
        self.X_train_fold = X_train_fold
        self.X_test_fold = X_test_fold
        self.y_train_final = y_train_final 
        self.X_val_final = X_val_final
        self.y_val_final = y_val_final
        self.X = X
        self.y = y

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
        
        self.history = self.model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val_final, y_val_final),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, lr_schedule],
            verbose=1
        )
        
        self.details.append({
            'fold': fold + 1, 
            'train_patients': len(self.train_patients),
            'test_patients': len(self.test_patients),
            'train_samples': len(self.X_train_fold),
            'test_samples': len(self.X_test_fold),
            'accuracy': self.report['accuracy'],
            'epochs_trained': len(self.history.history['loss'])
        })

        roc_aucs.append(self.auc)
        reports.append(self.report)
        conf_matrices.append(self.c_matrix)
        history.append(self.history.history)
    
        return self.reports, self.conf_matrices, self.details, self.roc_aucs, self.history

    def evaluate(self, X_test, y_test, y_pred, y_test_encoded, train_idx, test_idx):
        self.y_pred = y_pred 
        self.y_test_encoded = y_test_encoded
        self.X_test = X_test
        self.y_test = y_test 
        self.train_idx = train_idx
        self.test_idx = test_idx

        self.target_names = [str(cls) for cls in handler.encoder.classes_]
        self.report = classification_report(
            self.y_test_encoded, 
            self.y_pred, 
            target_names=self.target_names, 
            output_dict=True
        )

        return self.model.evaluate(self.X_test, self.y_test, verbose=0), self.report
    
    def predict(self, X):
        return self.model.predict(X)
    
    def summary(self):
        accuracies = [report['accuracy'] for report in reports]
        epochs_trained = [fold['epochs_trained'] for fold in details] 
        
        print("Per-fold results:")
        for i, (acc, details) in enumerate(zip(accuracies, details)):
            print(f"Fold {i+1}: {acc:.4f} accuracy "
                f"({details['test_patients']} patients, {details['test_samples']} samples, "
                f"{details['epochs_trained']} epochs)")

        for i, history in enumerate(history):
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
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"\nOverall Performance:")
        print(f"Mean Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Min Accuracy:  {min(accuracies):.4f}")
        print(f"Max Accuracy:  {max(accuracies):.4f}")
        
        class_0_f1 = [report['0']['f1-score'] for report in reports]
        class_1_f1 = [report['1']['f1-score'] for report in reports]
        
        print(f"\nClass-wise F1-scores:")
        print(f"Class 0: {np.mean(class_0_f1):.4f} ± {np.std(class_0_f1):.4f}")
        print(f"Class 1: {np.mean(class_1_f1):.4f} ± {np.std(class_1_f1):.4f}")
        
        avg_conf_matrix = np.mean(conf_matrices, axis=0)
        print(f"\nAverage Confusion Matrix:")
        print(np.round(avg_conf_matrix).astype(int))
        
        auc_scores = [score for score in roc_aucs if not np.isnan(score)]
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)

        print(f"\nMean ROC AUC (macro): {mean_auc:.4f} ± {std_auc:.4f}")

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(auc_scores) + 1), auc_scores, marker='o', linestyle='-',
                color='blue', label='ROC AUC per Fold')
        plt.axhline(mean_auc, color='red', linestyle='--', label=f'Mean AUC = {mean_auc:.4f}')
        plt.fill_between(range(1, len(auc_scores) + 1),
                        [mean_auc - std_auc] * len(auc_scores),
                        [mean_auc + std_auc] * len(auc_scores),
                        color='red', alpha=0.2, label='±1 STD')
        plt.xticks(range(1, len(auc_scores) + 1))
        plt.xlabel("Fold")
        plt.ylabel("ROC AUC (macro)")
        plt.title("ROC AUC per Fold (Macro)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return self.model.summary()
    
def pipeline(self, handler):
        gkf = GroupKFold(n_splits=4)    

        for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)): # From datahandling class 
            handler.split(self.X, self.y, train_idx, test_idx)
            handler.transform()
            handler.put_to_categorical()
            handler.validation_split()

            model = LungCancerMLP(handler.X_train, handler.num_classes)
            model.training_details(
                handler.X_test_scaled, handler.X_train_scaled,
                handler.X_train, handler.y_train,
                handler.X_val, handler.y_val,
                handler.X, handler.y_fold
            )
            
            model.evaluate(handler.X_test_scaled, handler.y_test_cat, handler.y_test_encoded, train_idx, test_idx)

handler = DataHandling()
model = LungCancerMLP(handler.X_train_final, handler.num_classes)

model.summary()

y_pred = np.argmax(model.predict(handler.X_test_scaled), axis=1)