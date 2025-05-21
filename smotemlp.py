# Multilayer Perceptron (MLP) implementation

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE 

# Needs some fix because TensorFlow is outdated or something 

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

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_categorical, test_size=0.3, stratify=y_encoded, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=np.argmax(y_temp, axis=1), random_state=42)

y_train_int = np.argmax(y_train, axis=1)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

if len(y_train_resampled) == 1:
    y_train_resampled = to_categorical(y_train_resampled, num_classes=num_classes)

y_train_resampled_int = np.argmax(y_train_resampled, axis=1)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Replace the X_train data with the SMOTE resampled

history = model.fit(X_train_resampled, y_train_resampled,
                    validation_data = (X_val, y_val),
                    epochs=50, 
                    batch_size=32,
                    verbose=1)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

target_names = [str(label) for label in le.classes_]

print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

# Checking for SMOTE Functionality 

from collections import Counter

print("Before Smote:", Counter(y_train_int))
print("After Smote:", Counter(y_train_resampled_int))

plt.plot(history.history['accuracy'], label="Training Accuracy" )
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Performance")
plt.show()

# Creating a confusion matrix 

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

c_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:\n", c_matrix)

display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=target_names)
display.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

def calculate_tp_fp_fn_tn(c_matrix):
    results = []
    for i in range(len(c_matrix)):
        TP = c_matrix[i, i]
        FP = c_matrix[:, i].sum() - TP
        FN = c_matrix[i, :].sum() - TP
        TN = c_matrix.sum() - (TP + FP + FN)
        results.append({'Class': target_names[i], 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN})
    return pd.DataFrame(results)

metrics_df = calculate_tp_fp_fn_tn(c_matrix)
print(metrics_df)




