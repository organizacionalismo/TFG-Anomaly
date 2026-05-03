import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocesamiento import cargar_datos

X, y, tipos_ataques = cargar_datos()

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
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Dado que la salida es un array de 9 elements
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


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Obtener predicciones
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print(f"Predicciones shape: {y_pred.shape}")
print(f"Primeras 5 predicciones: {y_pred[:5]}")
print(f"Primeras 5 reales: {y_test[:5]}")

# MATRIZ DE CONFUSIÓN
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True,  # Mostrar números
    fmt='d',     # Formato entero
    cmap='Blues',
    xticklabels=range(9),  # O usa le.classes_ si tienes LabelEncoder
    yticklabels=range(9),
    cbar_kws={'label': 'Cantidad'}
)
plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
plt.ylabel('Etiqueta Real', fontsize=12)
plt.xlabel('Etiqueta Predicha', fontsize=12)
plt.tight_layout()
plt.show()

# REPORTE DE CLASIFICACIÓN COMPLETO
print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN")
print("="*60)
print(classification_report(
    y_test, 
    y_pred,
    target_names=[f'Clase {i}' for i in range(9)],  # Nombres de clases
    digits=4  # 4 decimales
))

# MÉTRICAS GLOBALES
print("\n" + "="*60)
print("MÉTRICAS GLOBALES")
print("="*60)

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy:           {accuracy:.4f}")
print(f"\nPrecision (macro):  {precision_macro:.4f}")
print(f"Precision (weighted): {precision_weighted:.4f}")
print(f"\nRecall (macro):     {recall_macro:.4f}")
print(f"Recall (weighted):  {recall_weighted:.4f}")
print(f"\nF1-Score (macro):   {f1_macro:.4f}")
print(f"F1-Score (weighted): {f1_weighted:.4f}")

# MÉTRICAS POR CLASE
print("\n" + "="*60)
print("MÉTRICAS POR CLASE")
print("="*60)

for i in range(9):
    # Filtrar para cada clase
    mask_true = y_test == i
    mask_pred = y_pred == i
    
    tp = np.sum((y_test == i) & (y_pred == i))  # True Positives
    fp = np.sum((y_test != i) & (y_pred == i))  # False Positives
    fn = np.sum((y_test == i) & (y_pred != i))  # False Negatives
    tn = np.sum((y_test != i) & (y_pred != i))  # True Negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nClase {i}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Support:   {np.sum(mask_true)}")

# VISUALIZACIÓN DE ERRORES POR CLASE
errors_per_class = []
for i in range(9):
    mask = y_test == i
    correct = np.sum(y_pred[mask] == i)
    total = np.sum(mask)
    error_rate = 1 - (correct / total) if total > 0 else 0
    errors_per_class.append(error_rate * 100)

plt.figure(figsize=(10, 6))
plt.bar(range(9), errors_per_class, color='coral', alpha=0.7)
plt.xlabel('Clase', fontsize=12)
plt.ylabel('Tasa de Error (%)', fontsize=12)
plt.title('Tasa de Error por Clase', fontsize=14, fontweight='bold')
plt.xticks(range(9))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()