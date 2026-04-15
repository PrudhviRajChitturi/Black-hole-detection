import os
import numpy as np
import tensorflow as tf

def load_npy_dataset(data_dir):
    X, y = [], []

    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith("x_") and fname.endswith(".npy"):
            idx = fname.split("_")[1].split(".")[0]

            x = np.load(os.path.join(data_dir, f"x_{idx}.npy"))
            label = np.load(os.path.join(data_dir, f"y_{idx}.npy"))

            X.append(x)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def normalize_inputs(X):
    # Normalize first three channels separately
    for c in range(3):
        mean = X[..., c].mean()
        std = X[..., c].std() + 1e-6
        X[..., c] = (X[..., c] - mean) / std

    # Mask channel stays as-is
    return X
