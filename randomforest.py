from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import xgboost
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

# Multilayer Perceptron (MLP) implementation

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt 
from imblearn.combine import SMOTEENN
from collections import Counter

df = pd.read_csv("voice_features_log.csv")

X = df.drop(columns=['segment', 'cancer_stage', 'patient_id'])
y = df['cancer_stage']  

# Label encoding 
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))
y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data
"""
Train = training data
Val = validate; does training need to continue 
Test = when the model is trained, what is it's performance 
"""

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Apply SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

xgb = XGBClassifier(
    n_estimators = 100,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    early_stopping_rounds=10,
)

xgb.fit(
    X_train_resampled,
    y_train_resampled,
    eval_set=[(X_val, y_val)],
    verbose=True
)

y_pred = xgb.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=[str(label) for label in le.classes_]))


c_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", c_matrix)

ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=le.classes_).plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.show()