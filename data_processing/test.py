import pandas as pd
import numpy as np
import tensorflow as tf 
import random
import os
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input 
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
        self.probabilities = []

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

        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        self.X_train_resampled = None
        self.y_train_resampled = None

        self.groups = None
        self.feature_cols = None
        self.patient_labels = None
        self.inconsistent_patients = None
        self.patient_data = None

        self.train_patients = None
        self.test_patients = None
        
    def load_data(self):
        data = pd.read_csv(self.data)
        # Drop columns that are not features
        self.feature_cols = [col for col in data.columns if col not in ['segment', 'cancer_stage', 'patient_id']]
        self.X = data[self.feature_cols]
        self.y = data['cancer_stage']
        self.groups = data['patient_id']
        
        self.patient_labels = data.groupby('patient_id')['cancer_stage'].nunique()
        self.inconsistent_patients = self.patient_labels[self.patient_labels > 1]
        
        self.patient_data = data.groupby('patient_id').agg({
            'cancer_stage': lambda x: x.mode()[0]
        }).reset_index()

        print(f"\nPatient-level class distribution:")
        patient_class_counts = self.patient_data['cancer_stage'].value_counts()
        print(patient_class_counts)
        
        return len(self.inconsistent_patients) == 0

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
        
        # Encode labels as 0 or 1
        self.y_train_encoded = self.encoder.fit_transform(self.y_train_fold)
        self.y_test_encoded = self.encoder.transform(self.y_test_fold)

        # Resample training data with SMOTEENN
        self.X_train_resampled, self.y_train_resampled = self.smote.fit_resample(self.X_train_scaled, self.y_train_encoded)

    def validation_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_resampled, self.y_train_resampled, 
            test_size=0.1, stratify=self.y_train_resampled, random_state=SEED
        )

    def get_data(self):
        return self.X, self.y, self.X_train_fold, self.y_train_fold, self.X_test_fold, self.y_test_fold


class LungCancerMLP:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._buildmodel()

    def _buildmodel(self):
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(1, activation='sigmoid')  # binary classification output
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

        history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stopping, lr_schedule], verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        y_pred_prob = self.model.predict(X_test).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)

        report = classification_report(y_test, y_pred, output_dict=True)
        c_matrix = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)

        return report, c_matrix, auc

    def predict(self, X):
        y_pred_prob = self.model.predict(X).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        return y_pred, y_pred_prob


def pipeline(handler):
    gkf = GroupKFold(n_splits=4)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        print(f"\n=== Fold {fold+1} ===")

        # Split and preprocess
        data_df = pd.read_csv(handler.data)
        handler.split(handler.X, handler.y, data_df, train_idx, test_idx)
        handler.transform()
        handler.validation_split()

        # === Build MLP model ===
        model = Sequential([
            Input(shape=(handler.X_train.shape[1],)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')  # Binary classification
        ])

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        es = EarlyStopping(patience=10, restore_best_weights=True)

        history = model.fit(
            handler.X_train,
            handler.y_train,
            validation_data=(handler.X_val, handler.y_val),
            epochs=100,
            batch_size=32,
            callbacks=[es],
            verbose=0
        )

        # === Platt Scaling ===
        # Get raw probabilities from MLP on validation data
        val_probs = model.predict(handler.X_val).flatten()

        # Fit a logistic regression model for calibration
        platt_model = LogisticRegression()
        platt_model.fit(val_probs.reshape(-1, 1), handler.y_val)

        # Predict on test set
        raw_probs = model.predict(handler.X_test_scaled).flatten()
        calibrated_probs = platt_model.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        y_pred = (calibrated_probs >= 0.5).astype(int)

        # === Evaluation ===
        report = classification_report(handler.y_test_encoded, y_pred, output_dict=True)
        c_matrix = confusion_matrix(handler.y_test_encoded, y_pred)
        auc = roc_auc_score(handler.y_test_encoded, calibrated_probs)

        print(y_pred)
        print(calibrated_probs)

        handler.predictions.append(y_pred)
        handler.probabilities.append(calibrated_probs)

        print(f"Confusion Matrix:\n{c_matrix}")
        print(f"ROC AUC: {auc:.4f}")
        print("Classification Report:\n", classification_report(handler.y_test_encoded, y_pred))

        # === Save to CSV ===
        results_df = handler.X_test_fold.copy()
        results_df['true_label'] = handler.y_test_fold.values
        results_df['predicted_label'] = y_pred
        results_df['mlp_prob'] = calibrated_probs
        results_df['patient_id'] = data_df.iloc[test_idx]['patient_id'].values
        results_df['segment'] = data_df.iloc[test_idx]['segment'].values
        results_df.to_csv(f'results_fold_{fold+1}.csv', index=False)

handler = DataHandling()
if handler.load_data():
    pipeline(handler)
else:
    print("Data has inconsistent patients. Please fix the data before proceeding.")
