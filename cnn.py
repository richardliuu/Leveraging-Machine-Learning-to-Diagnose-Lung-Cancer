# Taking the .npy file and seeing its success 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Loading the data

data = np.load('all_mfccs.npy', allow_pickle=True)

X = []
y = []

for sample in data: 
    array_2D = sample[0]


# Scale and normalize

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# SMOTE 

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_categorical, test_size=0.3, stratify=y_encoded, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=np.argmax(y_temp, axis=1), random_state=42)

y_train_int = np.argmax(y_train, axis=1)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

if len(y_train_resampled) == 1:
    y_train_resampled = to_categorical(y_train_resampled, num_classes=num_classes)

y_train_resampled_int = np.argmax(y_train_resampled, axis=1)

# Model

model = Sequential([
    Conv2D(),
])

history = model.fit(

)

# Graphing

print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

plt.show()
