import pandas as pd
import numpy as np
import logging
import time
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Data split for surrogate model

class DataHandling:
    def __init__(self, data=r"data/binary_features_log.csv"):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.smote = SMOTEENN(random_state=SEED)
        self.data = data

        self.reports = []
        self.conf_matrices = []
        self.details = []
        self.history = []
        self.roc_aucs = []
        self.predictions = []

        self.X = None
        self.y = None

        self.X_train_fold = None
        self.y_train_fold = None
        self.X_test_fold = None
        self.y_test_fold = None

        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_encoded = None
        self.y_test_encoded = None

        self.y_train_cat = None
        self.y_test_cat = None

        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        self.X_train_resampled = None
        self.y_train_resampled = None

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

