import tensorflow as tf
import numpy as np
from cnn.dataset import load_npy_dataset, normalize_inputs
from cnn.model import build_resnet
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load data
X_pos, y_pos = load_npy_dataset("data/train/positive")
X_neg, y_neg = load_npy_dataset("data/train/negative")

X = np.concatenate([X_pos, X_neg], axis=0)
y = np.concatenate([y_pos, y_neg], axis=0)

X, y = shuffle(X, y, random_state=42)

X = normalize_inputs(X)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0])
print("Validation size:", X_val.shape[0])

# Data augmentation (IMPORTANT for small data)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.1)
])

# Model
model = build_resnet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=5,
        restore_best_weights=True
    )
]

# Train
history = model.fit(
    data_augmentation(X_train),
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16
)

model.save("bh_lens_resnet.keras")
