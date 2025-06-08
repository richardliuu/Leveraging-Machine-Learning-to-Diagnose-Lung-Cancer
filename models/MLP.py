# Multilayer Perceptron (MLP) implementation

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report 
import tensorflow as tf 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
import matplotlib.pyplot as plt 

# ===== Note to clean up the prints in the terminal and go back to the final model (clean code) 

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

history = model.fit(X_train, y_train,
                    validation_data = (X_val, y_val),
                    epochs=4, 
                    batch_size=32,
                    verbose=1)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

target_names = [str(label) for label in le.classes_]

print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

plt.plot(history.history['accuracy'], label="Training Accuracy" )
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Performance")
plt.show()

