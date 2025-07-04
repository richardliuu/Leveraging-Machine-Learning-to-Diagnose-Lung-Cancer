import pandas as pd
import numpy as np
import tensorflow as tf 
import random
import os
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
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

class DataHandling:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.smote = SMOTEENN()
        self.data = r"data/binary_mfccs.npy"

        self.history = []
        self.auc = []
        self.report = []
        self.c_matrix = []
        self.details = []

        self.X = None
        self.y = None
        self.groups = None
        self.y_test_fold = None 
        self.X_test_fold = None 
        self.y_encoded = None 
        self.X_train_fold = None 
        self.X_test_fold = None 
        self.y_train_fold_int = None
        self.y_test_fold_int = None 
        self.train_patients = None 
        self.val_patients = None
        self.test_patients = None 

        self.groups_train = None
        self.groups_test = None 

        self.num_classes = None
        self.patient_labels = None 
        self.inconsistent_patients = None 

        self.rows = None 
        self.train_idx = None
        self.test_idx = None 

    def load_data(self):
        data_array = np.load(self.data, allow_pickle=True)
        self.rows = [{'patient_id': pid, 'cancer_stage': label} for _, label, pid in data_array]
        data = pd.DataFrame(self.rows)
        self.patient_labels = data.groupby('patient_id')['cancer_stage'].nunique()
        self.inconsistent_patients = self.patient_labels[self.patient_labels > 1]

        return len(self.inconsistent_patients) == 0
    
    def transform(self, data_array):
        X, y, groups = [], [], []

        for mfcc, label, patient_id in data_array:
            mfcc = mfcc[:60] if mfcc.shape[0] >= 60 else np.pad(mfcc, ((0, 60 - mfcc.shape[0]), (0, 0)))
            X.append(mfcc)
            y.append(label)
            groups.append(patient_id)     

        self.X = np.array(self.X)[..., np.newaxis]  
        self.X = self.X / np.max(np.abs(self.X))

        self.X = X
        self.y = np.array(y)
        self.groups = np.array(groups)

    def data_split(self, encoder, train_idx, test_idx):

        self.train_idx = train_idx
        self.test_idx = test_idx

        self.y_encoded = encoder.fit_transform(self.y)
        self.X_train_fold = self.X[self.train_idx]
        self.X_test_fold = self.X[self.test_idx]
        self.y_train_fold_int = self.y_encoded[self.train_idx]
        self.y_test_fold_int = self.y_encoded[self.test_idx]
        self.groups_train = self.groups[self.train_idx]
        self.groups_test = self.groups[self.test_idx]

        self.y_train_fold = to_categorical(self.y_train_fold_int, num_classes=self.num_classes)
        self.y_test_fold = to_categorical(self.y_test_fold_int, num_classes=self.num_classes)

    def validation_split(self):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        val_train_idx, val_idx = next(gss.split(self.X_train_fold, self.y_train_fold_int, groups=self.groups_train))

        self.X_train_final = self.X_train_fold[val_train_idx]
        self.X_val_final = self.X_train_fold[val_idx]
        self.y_train_final = self.y_train_fold[val_train_idx]
        self.y_val_final = self.y_train_fold[val_idx]

        self.train_patients = set(self.groups_train[val_train_idx])
        self.val_patients = set(self.groups_train[val_idx])
        self.test_patients = set(self.groups_test)

    def get_data(self):
        return self.y_train_fold, self.y_test_fold, self.train_patients, self.val_patients, self.test_patients

class LungCancerCNN:
    def __init__(self, X_train_final, num_classes):
        self.X_train_final = X_train_final
        self.num_classes = num_classes 
        self.model = self._buildmodel()
        self.target_names = None

    def _buildmodel(self):
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
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        
        return model
    
    def train(self, X_train_final, y_train_final, X_val_final, y_val_final, epochs=50, batch_size=16):
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

        history = self.model.fit(
            X_train_final, y_train_final,
            validation_data=[X_val_final, y_val_final],
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stopping, lr_schedule], verbose=1
        )

        return history 

    def evaluate(self, y_test, preds, encoder):
        preds = np.argmax(self.model.predict(self.X_test_fold), axis=1)

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

def pipeline(handler):
    gkf = GroupKFold(n_splits=5)

    print("X is None", handler.X is None)
        
    print("Y is None", handler.y is None)
        
    print("groups is None", handler.groups is None)

    for fold, (train_idx, test_idx) in (gkf.split(handler.X, handler.y, handler.groups)):
        handler.data_split(train_idx, test_idx)
        handler.transform()
        handler.validation_split()

        model = LungCancerCNN(
        )

        history = model.train(
            handler.X_train, handler.y_train, handler.X_val, handler.y_val
        )

        report, c_matrix, auc = model.evaluate(
            handler.X_test_scaled, handler.y_test_encoded, handler.encoder
        )

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
# Requires a positional argument (data_array) but its not defined 
handler.load_data()
pipeline(handler)
