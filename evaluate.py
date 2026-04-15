import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from cnn.dataset import load_npy_dataset, normalize_inputs

# Load model
model = tf.keras.models.load_model("bh_lens_resnet_new.keras")

# Load test data
X_pos, y_pos = load_npy_dataset("data/test/positive")
X_neg, y_neg = load_npy_dataset("data/test/negative")

X = np.concatenate([X_pos, X_neg], axis=0)
y = np.concatenate([y_pos, y_neg], axis=0)

X = normalize_inputs(X)

print("Test shape:", X.shape)

# Predict
probs = model.predict(X).flatten()

# Metrics
auc = roc_auc_score(y, probs)
preds = (probs > 0.5).astype(int)
cm = confusion_matrix(y, preds)

print("Test AUC:", auc)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(y, preds))    

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y, probs)

# Youden’s J statistic
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]

print("Best threshold:", best_threshold)

preds = (probs > best_threshold).astype(int)
cm = confusion_matrix(y, preds)
print(cm)

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y, probs)
ap = average_precision_score(y, probs)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"PR Curve (AP={ap:.3f})")
plt.show()

print("Average Precision:", ap)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

plt.hist(probs[y==1], bins=20, alpha=0.6, label="Positive")
plt.hist(probs[y==0], bins=20, alpha=0.6, label="Negative")
plt.legend()
plt.show()

model = tf.keras.models.load_model("bh_lens_resnet2.keras")

for layer in model.layers:
    print(layer.name, getattr(layer, "output_shape", "no output_shape"))


from gradcam import make_gradcam_heatmap
import cv2
import matplotlib.pyplot as plt
import numpy as np

last_conv_layer_name = "conv2d_8"

# Pick a test sample
idx = 10
x_sample = X[idx]
img = np.expand_dims(x_sample, axis=0)

heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)

heatmap = cv2.resize(heatmap, (128, 128))

plt.figure(figsize=(6,6))
plt.imshow(x_sample[:,:,0], cmap='gray')  # observed channel
plt.imshow(heatmap, cmap='jet', alpha=0.4)
plt.title("Grad-CAM")
plt.colorbar()
plt.show()


