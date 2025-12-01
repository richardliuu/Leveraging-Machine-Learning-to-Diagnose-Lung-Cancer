import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
        self.data = pd.read_csv("data/jitter_shimmerlog.csv")
        self.X = self.data.drop(columns=['chunk', 'cancer_stage', 'patient_id', 'filename', 'rolloff', 'bandwidth', "skew", "zcr", 'rms'])
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
        
        return self.duplicates.sum() == 0 and len(self.inconsistent_patients) == 0, self.X.values, self.y, self.groups, self.feature_cols

class DataTransform():
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def fit_transform(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_enc = self.encoder.fit_transform(y_train)
        return X_train_scaled, y_train_enc

    def transform(self, X_val, y_val):
        X_val_scaled = self.scaler.transform(X_val)
        y_val_enc = self.encoder.transform(y_val)
        return X_val_scaled, y_val_enc

    def num_classes(self):
        return len(self.encoder.classes_)
        
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, num_classes):
        self.model = None
        super(MultiLayerPerceptron, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            #nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            #nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            #nn.BatchNorm1d(32),
            
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X)

class Training():
    def __init__(self, model, lr=1e-3, batch_size=32, epochs=50, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train, X_val, y_val):
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.long))
        
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            train_acc = 100 * correct / total

            val_loss, val_acc = self.evaluate(val_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}] "
                  f"Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        return total_loss / len(loader), 100 * correct / total

class Evaluation:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

def run_groupkfold(X, y, groups, n_splits=4, epochs=50, batch_size=32, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gkf = GroupKFold(n_splits=n_splits)

    all_reports = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n===== Fold {fold+1} =====")

        # Train/val split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Transform per fold (no leakage!)
        transformer = DataTransform()
        X_train, y_train = transformer.fit_transform(X_train, y_train)
        X_val, y_val = transformer.transform(X_val, y_val)

        input_dim = X_train.shape[1]
        num_classes = transformer.num_classes()

        # Train model
        model = MultiLayerPerceptron(input_dim, num_classes)
        trainer = Training(model, lr=lr, batch_size=batch_size, epochs=epochs, device=device)
        trainer.train(X_train, y_train, X_val, y_val)

        # Evaluate
        evaluator = Evaluation(model, device=device)
        y_pred = evaluator.predict(X_val)

        report = classification_report(y_val, y_pred, output_dict=True)
        print(classification_report(y_val, y_pred))
        print(confusion_matrix(y_val, y_pred))
        all_reports.append(report)

    return all_reports

if __name__ == "__main__":
    # 1. Load data
    ld = LoadData()
    ok, X, y, groups, feature_cols = ld.patient_split()

    if not ok:
        print("Data quality checks failed: duplicates or inconsistent patients found.")
    else:
        print("Data passed quality checks")

        # 2. Run GroupKFold training + evaluation
        reports = run_groupkfold(
            X=X,
            y=y.values,         # make sure y is numpy
            groups=groups.values,
            n_splits=4,
            epochs=30,
            batch_size=32,
            lr=1e-3
        )

        # 3. Summarize fold reports
        print("\n===== Cross-validation summary =====")
        for i, r in enumerate(reports):
            print(f"\nFold {i+1} report:")
            print(pd.DataFrame(r).transpose())

    


