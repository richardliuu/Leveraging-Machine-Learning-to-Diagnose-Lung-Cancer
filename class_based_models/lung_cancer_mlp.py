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

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

df = pd.read_csv("data/binary_features_log.csv")

X = df.drop(columns=['segment', 'cancer_stage', 'patient_id'])
y = df['cancer_stage']
groups = df['patient_id']  

group_kfold = GroupKFold(n_splits=4)

all_reports = []
all_conf_matrices = []
fold_details = []
all_histories = []
all_roc_aucs = [] 

for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
    
    # Split data
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]
    
    # Verify no patient leakage
    train_patients = set(df.iloc[train_idx]['patient_id'])
    test_patients = set(df.iloc[test_idx]['patient_id'])

    # Fit preprocessing ONLY on training data 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)
    
    # Encode labels (fit only on training data)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_fold)
    y_test_encoded = le.transform(y_test_fold)
    
    num_classes = len(le.classes_)
    
    # Apply SMOTEENN only to training data
    smote = SMOTEENN(random_state=SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train_resampled, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_encoded, num_classes=num_classes)
    
    # Create validation split from resampled training data
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_resampled, y_train_cat, test_size=0.1, 
        stratify=y_train_resampled, random_state=SEED
    )

class LungCancerMLP:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes 
        self._buildmodel()
        self.history = None

    def data_split():

    def _buildmodel(self):
        self.model = model = Sequential([
            Input(shape=(self.X_train_final.shape[1],)),
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
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

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
        
        self.history = self.model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val_final, y_val_final),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, lr_schedule],
            verbose=1
        )
        return self.history
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def summary(self):
        return self.model.summary()
    

    

        
    
