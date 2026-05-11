import os

import numpy as np

from pathlib import Path

dct = Path("datasets\day_0_420_user_1143")

num_files = {}
print("Numero de CSVs por tipo de ataque:")
for file in dct.glob("*.csv"):
    parts = file.stem.split("_")
    if parts[3] not in num_files:
        num_files[parts[3]] = 1
    else:
        num_files[parts[3]] += 1

for k, v in num_files.items():
    print(f"{k} -> {v} archivos")
