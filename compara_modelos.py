
import matplotlib.pyplot as plt

prototipos = {
    'Prototipo 1\n(LSTM simple)': {
        'accuracy': 0.755,
        'f1_macro': 0.753,
        'errors_per_class': [16, 37, 54, 11, 0, 27, 57, 9, 8]
    },
    'Prototipo 2\n(2 LSTM)': {
        'accuracy': 0.782,
        'f1_macro': 0.780,
        'errors_per_class': [14, 32, 48, 9, 0, 22, 45, 7, 5]
    },
    'Prototipo 3\n(2 LSTM + dropout)': {
        'accuracy': 0.810,
        'f1_macro': 0.808,
        'errors_per_class': [10, 25, 35, 7, 0, 18, 32, 5, 3]
    }
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (nombre, datos) in enumerate(prototipos.items()):
    ax = axes[idx]
    
    # Gráfica de barras de error por clase
    ax.bar(range(9), datos['errors_per_class'], 
           color='coral', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Clase', fontsize=11)
    ax.set_ylabel('Tasa de Error (%)', fontsize=11)
    ax.set_title(f'{nombre}\nAccuracy: {datos["accuracy"]:.3f} | F1: {datos["f1_macro"]:.3f}',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(9))
    
    # ESCALA FIJA EN TODAS
    ax.set_ylim(0, 100)
    
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparacion_prototipos.png', dpi=300, bbox_inches='tight')
plt.show()