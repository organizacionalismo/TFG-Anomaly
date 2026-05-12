import os

import numpy as np
import pandas as pd

from pathlib import Path

CLASS_MAP = {
    'Avg': 0,
    'FDI50': 1,
    'FDI150': 2,
    'MaxAvg': 3,
    'MinAvg': 4,
    'Normal': 5,
    'RSA0.10.8': 6,
    'RSA2.05.0': 7,
    'Swap': 8
}

def load_data():
    dct = Path("datasets\day_0_420_user_1143")

    X = []
    y = []

    for file in dct.glob("*.csv"):
            df = pd.read_csv(file, header=None)

            values = df[1].values.reshape(48, 1)

            X.append(values)

            parts = file.stem.split('_')
            name = parts[-1]
            
            y.append(CLASS_MAP[name])

    X = np.array(X)
    y = np.array(y)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X min: {X.min()}")
    print(f"X max: {X.max()}")
    print(f"X mean: {X.mean()}")
    print(f"X std: {X.std()}")
    print(f"Classes mapped: {CLASS_MAP}")

    return X, y, CLASS_MAP