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
        # Instantiated all variables used in the CNN models 
        pass


# Need to modify this CNN to MLP code 
# Model and Data handling is slightly different 
class LungCancerCNN:
    def __init__(self, X_train_final, num_classes):
        self.X_train_final = X_train_final
        self.num_classes = num_classes 
        self.model = self._buildmodel(X_train_final, num_classes)
        self.history = None
        self.auc = roc_aucs
        self.report = reports
        self.c_matrix = conf_matrices
        self.histories = histories 
        self.details = details
        self.accuracies = None
        self.epochs_trained = None
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

        roc_aucs.append(self.auc)
        reports.append(self.report)
        conf_matrices.append(self.c_matrix)
        histories.append(self.history.history)
    
        return self.reports, self.conf_matrices, self.details, self.histories, self.roc_aucs, self.history
        
    
