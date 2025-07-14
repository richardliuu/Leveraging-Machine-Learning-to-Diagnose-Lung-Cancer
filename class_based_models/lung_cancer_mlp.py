import pandas as pd
import numpy as np
import tensorflow as tf 
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

# Does not include making predictions outside of Cross Validation or Plotting/Summaries 
# Appends data 
# Look over generalizability because val accuracy is quite high 

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

class LungCancerMLP:
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes 
        self.model = self._buildmodel()

    def _buildmodel(self):
        model = Sequential([
            Input(shape=(self.input_dim,)),
            #Dense(512, activation='relu'),
            #BatchNormalization(),
            #Dropout(0.4),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights = True)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose = 1 )

        history = self.model.fit(
            X_train, y_train, validation_data = (X_val, y_val), 
            epochs = epochs, batch_size = batch_size, 
            callbacks = [early_stopping, lr_schedule], verbose = 1 
        )

        return history 
        
    def evaluate(self, X_test, y_test, encoder):
        preds = np.argmax(self.model.predict(X_test), axis=1)

        report = classification_report(
            y_test, preds, 
            target_names=[str(cls) for cls in encoder.classes_],
            output_dict=True 
        )

        c_matrix = confusion_matrix(y_test, preds)

        auc = roc_auc_score(
            to_categorical(y_test, num_classes=self.num_classes),
            to_categorical(preds, num_classes=self.num_classes),
            multi_class='ovr'
        )

        return report, c_matrix, auc 
    
    def predict(self, X):
        y_pred_prob = self.model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        return y_pred 

def pipeline(handler):
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        handler.split(handler.X, handler.y, pd.read_csv(handler.data), train_idx, test_idx)
        handler.transform()
        handler.put_to_categorical()
        handler.validation_split()

        model = LungCancerMLP(
            num_classes=handler.num_classes,
            input_dim=handler.X_train.shape[1],
        )

        history = model.train(
            handler.X_train, handler.y_train, handler.X_val, handler.y_val
        )

        report, c_matrix, auc = model.evaluate(
            handler.X_test_scaled, handler.y_test_encoded, handler.encoder
        )

        y_pred = model.predict(handler.X_test_scaled)

        handler.predictions.append(y_pred)
        handler.reports.append(report)
        handler.conf_matrices.append(c_matrix)
        handler.roc_aucs.append(auc)
        handler.history.append(history.history)
        handler.details.append({
            "fold": fold + 1,
            "train_samples": len(handler.X_train_fold),
            "test_samples": len(handler.X_test_fold),
            "accuracy": report['accuracy'],
            "epochs_trained": len(history.history['loss']),
        })

        # Logging 
        accuracies = [report['accuracy'] for report in handler.reports]
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print(f"\nOverall Performance:")
        print(f"Mean Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Min Accuracy:  {min(accuracies):.4f}")
        print(f"Max Accuracy:  {max(accuracies):.4f}")
        
        class_0_f1 = [report['0']['f1-score'] for report in handler.reports]
        class_1_f1 = [report['1']['f1-score'] for report in handler.reports]
        
        print(f"\nClass-wise F1-scores:")
        print(f"Class 0: {np.mean(class_0_f1):.4f} ± {np.std(class_0_f1):.4f}")
        print(f"Class 1: {np.mean(class_1_f1):.4f} ± {np.std(class_1_f1):.4f}")
        
        print(c_matrix)

handler = DataHandling()
handler.load_data()
pipeline(handler)
