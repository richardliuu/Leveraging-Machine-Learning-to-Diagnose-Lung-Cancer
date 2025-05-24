# Taking the .npy file and seeing its success 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# Loading the data

data = np.load('all_mfccs.npy', allow_pickle=True)

X = []
y = []

for mfcc, label, _ in data:
    if mfcc.shape[0] >= 60:
        mfcc = mfcc[:60]
    else:
        pad_width = 60 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))

    X.append(mfcc)
    y.append(label)

X = np.array(X)
y = np.array(y)


# Scale and normalize
# Turns it into a smaller number (easier to use)

X = X / np.max(np.abs(X))

# Encoding 

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# SMOTE 

X_train, X_temp, y_train, y_temp = train_test_split(X, y_categorical, test_size=0.3, stratify=y_encoded, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=np.argmax(y_temp, axis=1), random_state=42)

# Skipping SMOTE because it doesn't work with 3D data like how CNN's require 
"""
y_train_int = np.argmax(y_train, axis=1)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

if len(y_train_resampled) == 1:
    y_train_resampled = to_categorical(y_train_resampled, num_classes=num_classes)

y_train_resampled_int = np.argmax(y_train_resampled, axis=1)
"""
# Model

model = Sequential([
    Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(60, 13, 1)),
    MaxPooling2D(pool_size=(2, 2)), 
    Dropout(0.3),

    Conv2D(64, kernel_size = (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_data = (X_val, y_val),
                    epochs=100, 
                    batch_size=32, 
                    verbose=1, 
                    )

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


target_names = encoder.classes_
target_names = [f"Stage {i}" for i in encoder.inverse_transform(np.arange(y_categorical.shape[1]))]


# Graphing

print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

c_matrix = confusion_matrix(y_true_classes, y_pred_classes)
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

plt.plot(history.history['accuracy'], label="Training Accuracy" )
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("CNN Model Performance")
plt.show()
