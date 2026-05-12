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

CLASS_NAMES = ("avg", "const50", "const150", "max_avg", "min_avg", "normal", "rsa01_08", "rsa2_5", "swap")

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

def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm, 
        annot=True,  
        fmt='d',     
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Number of predictions'},
        vmin=0.0,
        vmax=np.max(cm)
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.title('Confusion matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Real Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_errors_class(y_test, y_pred, save_path=None):
    errors_per_class = []
    for i in range(len(CLASS_NAMES)):
        mask = y_test == i
        correct = np.sum(y_pred[mask] == i)
        total = np.sum(mask)
        error_rate = 1 - (correct / total) if total > 0 else 0
        errors_per_class.append(error_rate * 100)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(CLASS_NAMES)), errors_per_class, color='coral', alpha=0.7)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.title('Class Error Rate', fontsize=14, fontweight='bold')
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
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
    
    table = ax.table(
        cellText=[[m, f'{v:.4f}'] for m, v in zip(df['Metric'], df['Value'])],
        colLabels=['Metric', 'Value'],
        cellLoc='left',
        loc='center',
        colWidths=[0.6, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(df) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#E0CBCB")
    
    plt.title('Global Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_class_metrics(y_test, y_pred, save_path=None):
    n_classes = len(np.unique(y_test))
    
    metrics_per_class = []
    
    for i in range(n_classes):
        tp = np.sum((y_test == i) & (y_pred == i))
        fp = np.sum((y_test != i) & (y_pred == i))
        fn = np.sum((y_test == i) & (y_pred != i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_test == i)
        
        metrics_per_class.append([CLASS_NAMES[i], precision, recall, f1, support])
        print(f"class {i} errors appended")
    
    # create DataFrame
    df = pd.DataFrame(metrics_per_class, 
                      columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    
    fig, ax = plt.subplots(figsize=(10, max(6, n_classes * 0.5)))
    
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=[[c, f'{p:.4f}', f'{r:.4f}', f'{f:.4f}', f'{int(s)}'] 
                  for c, p, r, f, s in metrics_per_class],
        colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.2, 0.2, 0.2, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, n_classes + 1):
        for j in range(1, 4):
            if metrics_per_class[i - 1][j] < 0.5: 
                table[(i, j)].set_facecolor('#F4C7C3')  # Rojo claro
            elif metrics_per_class[i - 1][j] < 0.75:
                 table[(i, j)].set_facecolor('#FCE8B2')  # Amarillo claro
            else:
                table[(i, j)].set_facecolor('#B7E1CD')  # Verde claro
    
    ax.set_title('Metrics per Class', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"MÉTRICAS POR CLASE")
    print(f"{'='*60}")
    print(df.to_string(index=False))