import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize

from ring_detection.RingD import detect_einstein_rings_jwst
from cnn.dataset import normalize_inputs
from statistical.mcmc_runner import run_mcmc_estimation  # adjust to your file


# -------------------------------------------------
# 1. Load CNN
# -------------------------------------------------

model = tf.keras.models.load_model("bh_lens_resnet_new.keras")

THRESHOLD = 0.05   # use your tuned threshold


# -------------------------------------------------
# 2. Helper: Crop Around Ring
# -------------------------------------------------

def crop_candidate(image, x, y, r, size=128):

    H, W = image.shape

    half = int(1.5 * r)

    x1 = max(x - half, 0)
    x2 = min(x + half, W)

    y1 = max(y - half, 0)
    y2 = min(y + half, H)

    crop = image[y1:y2, x1:x2]

    crop_resized = resize(crop, (size, size), mode="reflect")

    return crop_resized


# -------------------------------------------------
# 3. Main Pipeline
# -------------------------------------------------

def run_pipeline(image):
    
    '''from skimage.transform import resize

    MAX_SIZE = 800

    H, W = image.shape

    if max(H, W) > MAX_SIZE:
        scale = MAX_SIZE / max(H, W)
        new_shape = (int(H * scale), int(W * scale))
        image = resize(image, new_shape, mode="reflect")'''

    rings = detect_einstein_rings_jwst(image)

    confirmed = []

    for i, ring in enumerate(rings):

        x0, y0, r0 = ring["x"], ring["y"], ring["r"]

        crop = crop_candidate(image, x0, y0, r0)

        # Build 3-channel input like training
        model_img = np.zeros_like(crop)
        residual = crop - model_img

        x_input = np.stack([crop, model_img, residual], axis=-1)

        x_input = normalize_inputs(np.expand_dims(x_input, axis=0))

        prob = model.predict(x_input)[0][0]

        if prob > THRESHOLD:

            # Run MCMC
            stats = run_mcmc_estimation(crop)

            confirmed.append({
                "x": x0,
                "y": y0,
                "r": r0,
                "prob": prob,
                "stats": stats
            })

    return confirmed


# -------------------------------------------------
# 4. Visualization
# -------------------------------------------------

def visualize_results(image, results):
    plt.figure(figsize=(12,8))
    plt.imshow(image, origin="lower", cmap="gray")
    ax = plt.gca()
    
    print(f"Image shape: {image.shape}")  # Debug: check dimensions
    print(f"Results length: {len(results)}")  # Debug: empty?
    
    if not results:
        print("No results to plot!")
        plt.title("No Detections")
        plt.show()
        return
    
    for idx, obj in enumerate(results):
        print(f"Candidate {idx}: x={obj['x']}, y={obj['y']}, r={obj['r']}")  # DEBUG VALUES
        
        # Scale if normalized coords [0,1]
        x, y, r = obj['x'], obj['y'], obj['r']
        if max(x, y, r) <= 1.0:  # Normalized?
            h, w = image.shape
            x, y, r = x * w, y * h, r * max(h, w) * 0.01  # Scale up
        
        circle = plt.Circle((x, y), r, color="red", fill=False, linewidth=3)
        ax.add_patch(circle)
        
        plt.text(x, y - r - 10, f"{idx+1}", color="yellow", fontsize=14,
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.8),
                ha='center', va='top', weight='bold')
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Fix origin="lower" coordinate flip
    
    plt.title("Black Hole Candidates (Debug Mode)")
    plt.tight_layout()
    plt.show()

