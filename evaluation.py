import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
ACCURACY_RANGE = (0.0, 1.0)
LOSS_RANGE = (0.0, 2.5)
ERROR_RANGE = (0, 100)
CONFUSION_RANGE = (0, 126)

def plot_training_curves(history, save_path=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ACCURACY
    ax1.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
    ax1.axvline(x=np.argmax(val_acc), color='r', linestyle='--', 
                label=f'Best epoch: {np.argmax(val_acc)+1}', alpha=0.5)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(ACCURACY_RANGE)
    
    # LOSS
    ax2.plot(epochs_range, loss, label='Training Loss', linewidth=2)
    ax2.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
    ax2.axvline(x=np.argmin(val_loss), color='r', linestyle='--', 
                label=f'Best epoch: {np.argmin(val_loss)+1}', alpha=0.5)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(LOSS_RANGE)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"MEJOR EPOCH")
    print(f"{'='*60}")
    print(f"Mejor Validation Accuracy: {np.max(val_acc):.4f} (Epoch {np.argmax(val_acc)+1})")
    print(f"Mejor Validation Loss:     {np.min(val_loss):.4f} (Epoch {np.argmin(val_loss)+1})")

def plot_confusion_matrix(y_test, y_pred, class_names=None, save_path=None):
    cm = confusion_matrix(y_test, y_pred)

    if class_names is None:
        class_names = range(len(cm))
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm, 
        annot=True,  
        fmt='d',     
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Número de aciertos'},
        vmin=CONFUSION_RANGE[0],
        vmax=CONFUSION_RANGE[1]
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_errors_class(y_test, y_pred, class_names, save_path=None):
    errors_per_class = []
    for i in range(len(class_names)):
        mask = y_test == i
        correct = np.sum(y_pred[mask] == i)
        total = np.sum(mask)
        error_rate = 1 - (correct / total) if total > 0 else 0
        errors_per_class.append(error_rate * 100)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_names)), errors_per_class, color='coral', alpha=0.7)
    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Tasa de Error (%)', fontsize=12)
    plt.title('Tasa de Error por Clase', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylim(ERROR_RANGE)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_global_metrics(y_test, y_pred, save_path=None):
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision (macro)', 'Precision (weighted)', 
                    'Recall (macro)', 'Recall (weighted)', 'F1-Score (macro)', 'F1-Score (weighted)'],
        'Value': [accuracy, precision_macro, precision_weighted, 
                  recall_macro, recall_weighted, f1_macro, f1_weighted]
    }
    df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Crear tabla
    table = ax.table(
        cellText=[[m, f'{v:.4f}'] for m, v in zip(df['Metric'], df['Value'])],
        colLabels=['Metric', 'Value'],
        cellLoc='left',
        loc='center',
        colWidths=[0.6, 0.2]
    )
    
    # Estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(df) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('Métricas Globales del Modelo', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_class_metrics(y_test, y_pred, class_names=None, save_path=None):
    n_classes = len(np.unique(y_test))
    
    if class_names is None:
        class_names = range(9)
    
    # Calcular métricas por clase
    metrics_per_class = []
    
    for i in range(n_classes):
        tp = np.sum((y_test == i) & (y_pred == i))
        fp = np.sum((y_test != i) & (y_pred == i))
        fn = np.sum((y_test == i) & (y_pred != i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_test == i)
        
        metrics_per_class.append([class_names[i], precision, recall, f1, support])
        print(f"class {i} errors appended")
    
    # Crear DataFrame
    df = pd.DataFrame(metrics_per_class, 
                      columns=['Clase', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, n_classes * 0.5)))
    
    # ===== SUBPLOT 1: TABLA =====
    ax1.axis('tight')
    ax1.axis('off')
    
    table = ax1.table(
        cellText=[[c, f'{p:.4f}', f'{r:.4f}', f'{f:.4f}', f'{int(s)}'] 
                  for c, p, r, f, s in metrics_per_class],
        colLabels=['Clase', 'Precision', 'Recall', 'F1-Score', 'Support'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.2, 0.2, 0.2, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Colorear header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear celdas según valor
    for i in range(1, n_classes + 1):
        # Precision
        precision_val = metrics_per_class[i-1][1]
        if precision_val < 0.5:
            table[(i, 1)].set_facecolor('#F4C7C3')  # Rojo claro
        elif precision_val < 0.75:
            table[(i, 1)].set_facecolor('#FCE8B2')  # Amarillo claro
        else:
            table[(i, 1)].set_facecolor('#B7E1CD')  # Verde claro
        
        # Recall
        recall_val = metrics_per_class[i-1][2]
        if recall_val < 0.5:
            table[(i, 2)].set_facecolor('#F4C7C3')
        elif recall_val < 0.75:
            table[(i, 2)].set_facecolor('#FCE8B2')
        else:
            table[(i, 2)].set_facecolor('#B7E1CD')
        
        # F1-Score
        f1_val = metrics_per_class[i-1][3]
        if f1_val < 0.5:
            table[(i, 3)].set_facecolor('#F4C7C3')
        elif f1_val < 0.75:
            table[(i, 3)].set_facecolor('#FCE8B2')
        else:
            table[(i, 3)].set_facecolor('#B7E1CD')
    
    ax1.set_title('Métricas por Clase', fontsize=14, fontweight='bold', pad=20)
    
    # ===== SUBPLOT 2: HEATMAP =====
    metrics_matrix = df[['Precision', 'Recall', 'F1-Score']].values
    
    sns.heatmap(
        metrics_matrix.T,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=class_names,
        yticklabels=['Precision', 'Recall', 'F1-Score'],
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Score'},
        ax=ax2
    )
    ax2.set_title('Heatmap de Métricas por Clase', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Clase', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"MÉTRICAS POR CLASE")
    print(f"{'='*60}")
    print(df.to_string(index=False))