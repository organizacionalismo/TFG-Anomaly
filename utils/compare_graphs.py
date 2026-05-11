import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

save_path = None

dct = Path("datasets\day_0_420_user_1143")

df1 = pd.read_csv(dct / "days_0_1_Normal.csv", header=None)
df2 = pd.read_csv(dct / "days_0_1_FDI50.csv", header=None)

data1 = df1[1]
data2 = df2[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

plt.ylim(0, 5)
plt.xlabel("Registros")
plt.ylabel("Valores")
plt.title("Consumo del día 0_1")


ax1.plot(range(len(data1)), data1, label='Consumption', linewidth=2)
ax1.set_xlabel('Timestamps', fontsize=12)
ax1.set_ylabel('Values', fontsize=12)
ax1.set_title('Normal Consumption Distribution', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.0, 6.0)

# LOSS
ax2.plot(range(len(data2)), data2, label='Consumption', linewidth=2)
ax2.set_xlabel('Timestamps', fontsize=12)
ax2.set_ylabel('Values', fontsize=12)
ax2.set_title('FDI150 Attack Distribution', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.0, 6.0)

plt.tight_layout()

if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()

