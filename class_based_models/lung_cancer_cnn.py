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
    def __init__(self, data=r"binary_mfccs.npy"):
        self.encoder = LabelEncoder()
        self.smote = SMOTEENN()

        self.history = []
        self.auc = []
        self.report = []
        self.c_matrix = []
        self.details = []
        self.data = data 

        self.X = []
        self.y = [] 
        self.groups = [] 

# Pipeline 
    def data_split():
        pass 
        # Data Split 

    def transform():
        pass
        # Transform 

    def put_to_categorical():
        pass
        # Categorical

    def validation_split():
        pass 
        # Validation Split 

class LungCancerCNN:
    def __init__(self, X_train_final, num_classes):
        self.X_train_final = X_train_final
        self.num_classes = num_classes 
        self.model = self._buildmodel()
        self.target_names = None

    def _buildmodel(self, X_train_final, num_classes):
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
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model
    
    # Need to insert the training data in the params
    def train(self,  ,epochs=50, batch_size=16):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

        history = self.model.fit(
            
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stopping, lr_schedule], verbose=1
        )

        return history 


    def training_details(self, X_test_fold, X_train_fold, X_train_final, y_train_final, X_val_final, y_val_final, X, y, fold, epochs=50, batch_size=16):
        self.X_test_fold = X_train_fold
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
            'fold': fold + 1, # Create a variable/sort it out 
            'train_patients': len(self.train_patients),
            'test_patients': len(self.test_patients),
            'train_samples': len(self.X_train_fold),
            'test_samples': len(self.X_test_fold),
            'accuracy': self.report['accuracy'],
            'epochs_trained': len(self.history.history['loss'])
        })
    
        return self.reports, self.conf_matrices, self.details, self.histories, self.roc_aucs, self.history
        
def pipeline(handler):
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        handler.split(handler.X, handler.y, pd.read_csv(handler.data), train_idx, test_idx)
        handler.transform()
        handler.put_to_categorical()
        handler.validation_split()

        model = LungCancerCNN(
            num_classes=handler.num_classes,
            input_dim=handler.X_train.shape[1],
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

handler = DataHandling()
if handler.load_data(): 
    pipeline(handler)
else:
    print("Error with duplicate data or inconsistent patients")