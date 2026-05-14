import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

CLASS_NAMES = ("avg", "const50", "const150", "max_avg", "min_avg", "normal", "rsa01_08", "rsa2_5", "swap")

save_path = "..\outputs\comparison_normal_between_seasons.png"

dct = Path("datasets\day_0_420_user_1143")

df1 = pd.read_csv(dct / "days_40_41_Normal.csv", header=None)
df2 = pd.read_csv(dct / "days_222_223_Normal.csv", header=None)
# df2 = pd.read_csv(dct / "days_40_41_RSA0.10.8.csv", header=None)
# df2 = pd.read_csv(dct / "days_40_41_Avg.csv", header=None)

data1 = df1[1]
data2 = df2[1]

max_1 = np.max(data1) 
max_2 = np.max(data2)
ylim = max(max_1, max_2) * 1.1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(len(data1)), data1, label='Consumption', linewidth=2)
ax1.set_xlabel('Timestamps', fontsize=12)
ax1.set_ylabel('Values', fontsize=12)
ax1.set_title('Normal Consumption', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.0, ylim)

ax2.plot(range(len(data2)), data2, label='Consumption', linewidth=2)
ax2.set_xlabel('Timestamps', fontsize=12)
ax2.set_ylabel('Values', fontsize=12)
ax2.set_title('Normal Consumption (6 months later)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.0, ylim)

plt.tight_layout()

if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()

