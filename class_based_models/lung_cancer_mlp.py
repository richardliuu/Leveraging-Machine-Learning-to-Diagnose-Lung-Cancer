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

reports = []
conf_matrices = []
details = []
histories = []
roc_aucs = [] 

class DataHandling:
    def __init__(self, data=r"data/binary_features_log.csv"):
        # Functions 
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.smote = SMOTEENN(random_state=SEED)
        self.split = train_test_split()
        self.categorical = to_categorical()
        self.data = data

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
        self.y_train_resampled

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

    def categorical(self):
        self.y_train_cat = self.categorical(self.y_train_resampled, num_classes=self.num_classes)
        self.y_test_cat = self.categorical(self.y_test_encoded, num_classes=self.num_classes)
    
    def validation_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
        self.X_train_resampled, self.y_train_cat, test_size=0.1, 
        stratify=self.y_train_resampled, random_state=SEED
        )

    def get_data(self):
        return self.X, self.y, self.X_train_fold, self.y_train_fold, self.X_test_fold, self.y_test_fold

class LungCancerMLP:
    def __init__(self, X_train, num_classes, feature_cols):
        self.X_train_final = X_train
        self.num_classes = num_classes 
        self.model = self._buildmodel(X_train, num_classes)
        self.history = None
        self.auc = roc_aucs
        self.report = reports
        self.c_matrix = conf_matrices
        self.histories = histories 
        self.details = details
        self.feature_cols = feature_cols

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

    def train(self, X_test_fold, X_train_fold, X_train_final, y_train_final, X_val_final, y_val_final, epochs=50, batch_size=16):
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
            'train_samples': len(X_train_fold),
            'test_samples': len(X_test_fold),
            'accuracy': self.report['accuracy'],
            'epochs_trained': len(self.history.history['loss'])
        })

        roc_aucs.append(self.auc)
        reports.append(self.report)
        conf_matrices.append(self.c_matrix)
        histories.append(self.history.history)
    
        return self.reports, self.conf_matrices, self.details, self.histories, self.roc_aucs, self.history

    def evaluate(self, X_test, y_test, y_pred, y_test_encoded):
        target_names = [str(cls) for cls in self.encoder.classes_]
        self.report = classification_report(
            y_test_encoded, 
            y_pred, 
            target_names=target_names, 
            output_dict=True
        )

        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def summary(self, ):
        return self.model.summary()
    
for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):

    


    

    

        
    
