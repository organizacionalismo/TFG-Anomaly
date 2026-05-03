import os

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

def cargar_datos():
    # y = ["avg", "fdi50", "fdi150", "max_avg", "min_avg", "normal", "rsa01_08", "rsa2_5", "swap"]

    dct = Path("datasets\day_0_420_user_1143")

    X = []
    y = []
    dicc_clases = {}

    for arch in dct.glob("*.csv"):
            df = pd.read_csv(arch, header=None)

            values = df[1].values.reshape(48, 1)

            X.append(values)

            partes = arch.stem.split('_')
            nombre = partes[-1] # nombre del tipo
            if (nombre not in dicc_clases):
                dicc_clases[nombre] = len(dicc_clases)
            
            y.append(dicc_clases[nombre])
            # print(f"Archivo: {arch.name} → Clase: {nombre} (ID: {dicc_clases[nombre]})")

    X = np.array(X)
    y = np.array(y)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X min: {X.min()}")
    print(f"X max: {X.max()}")
    print(f"X mean: {X.mean()}")
    print(f"X std: {X.std()}")
    print(f"Clases: {dicc_clases}")

    return X, y, dicc_clases