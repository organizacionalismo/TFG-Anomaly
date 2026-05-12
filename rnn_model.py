import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from preprocessing import *
from evaluation import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

X, y, class_dict = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify = y)

# array de 48 dimensiones del consumo
# tipo de ataque 
# salida

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(48, 1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)), # segunda capa LSTM
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), # Dado que la salida es un array de 9 elements
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    batch_size=32,
    verbose=1
)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

plot_training_curves(history)
plot_confusion_matrix(y_test, y_pred)
plot_errors_class(y_test, y_pred)
plot_global_metrics(y_test, y_pred)
plot_class_metrics(y_test, y_pred)