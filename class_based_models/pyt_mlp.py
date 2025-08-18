import torch.nn as nn
import shap
import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.combine import SMOTEENN

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class LoadData():
    def __init__(self):
        self.data = None

    def patient_split(self):
        self.data = pd.read_csv("data/train_data.csv")
        self.X = self.data.drop(columns=['segment', 'cancer_stage', 'patient_id'])
        self.y = self.data['cancer_stage']

        # Allows SHAP to collect the feature names for analysis 
        self.feature_cols = self.X.columns.tolist()

        self.groups = self.data['patient_id']
        # Check for duplicate samples across all feature columns
        self.duplicates = self.data.duplicated(subset=self.feature_cols)
        # Verify each patient has consistent cancer stage labels across all samples
        self.patient_labels = self.data.groupby('patient_id')['cancer_stage'].nunique()
        self.inconsistent_patients = self.patient_labels[self.patient_labels > 1]  # Patients with >1 unique label
        
        if len(self.inconsistent_patients) > 0:
            print(f"WARNING: {len(self.inconsistent_patients)} patients have inconsistent labels!")
            print("Inconsistent patients:", self.inconsistent_patients.index.tolist()[:10])
            # Show details for first inconsistent patient
            if len(self.inconsistent_patients) > 0:
                first_patient = self.inconsistent_patients.index[0]
                patient_data = self.data[self.data['patient_id'] == first_patient][['patient_id', 'cancer_stage']]
                print(f"Example - Patient {first_patient}:")
                print(patient_data['cancer_stage'].value_counts())
        
        # Calculate patient-level class distribution using mode (most frequent label per patient)
        self.patient_data = self.data.groupby('patient_id').agg({
            'cancer_stage': lambda x: x.mode()[0]  # Take most frequent label for each patient
        }).reset_index()

        print(f"\nPatient-level class distribution:")
        patient_class_counts = self.patient_data['cancer_stage'].value_counts()
        print(patient_class_counts)
        
        return self.duplicates.sum() == 0 and len(self.inconsistent_patients) == 0

class DataTransform():
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def transform(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Encoded
        self.y_train = self.encoder.fit_transform(self.y_train)
        self.y_test = self.encoder.transform(self.y_test)
        self.num_classes = len(self.encoder.classes_)
        self.X_train, self.y_train = self.smote.fit_resample(self.X_train, self.y_train)

class ValidationSplit():
    def __init__(self):
        self.X_train = X
        self.y_train = y 

    def split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=0.1, 
            stratify=self.y_train, 
            random_state=SEED
            )
        
        return self.X_train, self.X_val, self.y_train, self.y_val
        
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        self.model = None
        super(MultiLayerPerceptron, self).__init__()
        
        self.model == nn.Sequential(
            nn.Linear(128),
            nn.ReLU(),
            nn.BatchNorm1D(),
            nn.Dropout(0.3),

            nn.Linear(64),
            nn.ReLU(),
            nn.BatchNorm1D(),
            nn.Dropout(0.3),

            nn.Linear(32),
            nn.ReLU(),
            nn.BatchNorm1D(),
            
            nn.Linear(self.num_classes),
            nn.sigmoid()
        )

class Training():
    def __init__(self):
        pass

    def train():
        model.train()

class Evaluation():
    def __init__(self):
        pass

model = MultiLayerPerceptron



    


