# Taking the .npy file and seeing its success 
import numpy as np 

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Loading the data

data = np.load('all_mfccs.npy', allow_pickle=True)

X = []
y = []

for sample in data: 
    array_2D = sample[0]

# Model

model = Sequential([

])

# Graphing

import matplotlib.pyplot as plt 
