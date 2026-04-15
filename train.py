import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from cnn.dataset import load_npy_dataset, normalize_inputs
from cnn.model import build_resnet


# -------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------

X_pos, y_pos = load_npy_dataset("data/train/positive")
X_neg, y_neg = load_npy_dataset("data/train/negative")

X = np.concatenate([X_pos, X_neg], axis=0)
y = np.concatenate([y_pos, y_neg], axis=0)

# Shuffle before split
X, y = shuffle(X, y, random_state=42)

# Normalize inputs
X = normalize_inputs(X)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# -------------------------------------------------
# 2. Train / Validation Split
# -------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Validation size:", X_val.shape[0])

# -------------------------------------------------
# 3. Data Augmentation (inside model)
# -------------------------------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.1)
])

# -------------------------------------------------
# 4. Build Model
# -------------------------------------------------

base_model = build_resnet(input_shape=X_train.shape[1:])

inputs = tf.keras.Input(shape=X_train.shape[1:])
x = data_augmentation(inputs)
outputs = base_model(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# -------------------------------------------------
# 5. Callbacks
# -------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=6,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# -------------------------------------------------
# 6. Train
# -------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------------
# 7. Save Model
# -------------------------------------------------

model.save("bh_lens_resnet.keras")

print("Model saved successfully.")
