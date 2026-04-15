import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.feature import peak_local_max
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip


# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def preprocess_image(image, sigma=2.0):
    return gaussian_filter(image, sigma=sigma)


def normalize_image(image):
    """
    Robust normalization for JWST-like images.
    """
    img = image.copy()
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-8)
    return img

def subtract_background(image, box_size=64):
    """
    Background subtraction.

    - For small simulated images: median subtraction
    - For large real images: Background2D
    """

    # Small images → simple & robust
    if image.shape[0] < box_size or image.shape[1] < box_size:
        return image - np.median(image)

    # Large images → adaptive background
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()

    bkg = Background2D(
        image,
        box_size=(box_size, box_size),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
        exclude_percentile=90  # allow signal-dominated fields
    )

    return image - bkg.background

# -------------------------------------------------
# Ridge detection
# -------------------------------------------------

def detect_directed_ridges(image, sigma=2.0, curvature_thresh=0.0):
    H_elems = hessian_matrix(
        image,
        sigma=sigma,
        order="rc",
        use_gaussian_derivatives=False
    )

    l1, l2 = hessian_matrix_eigvals(H_elems)

    ridge_mask = (l1 < curvature_thresh) & (np.abs(l1) > np.abs(l2))

    gx, gy = np.gradient(image)
    norm = np.sqrt(gx**2 + gy**2) + 1e-8
    gx /= norm
    gy /= norm

    return ridge_mask, gx, gy



# -------------------------------------------------
# Directed Hough Transform
# -------------------------------------------------

def directed_circle_hough(ridge_mask, gx, gy, r_min, r_max, image_shape):
    H, W = image_shape
    R = r_max - r_min + 1
    accumulator = np.zeros((H, W, R), dtype=np.float32)

    ys, xs = np.where(ridge_mask)

    for y, x in zip(ys, xs):
        dx, dy = gx[y, x], gy[y, x]

        for r in range(r_min, r_max + 1):
            cx = int(x - r * dx)
            cy = int(y - r * dy)

            if 0 <= cx < W and 0 <= cy < H:
                accumulator[cy, cx, r - r_min] += 1.0

    return accumulator


def normalize_parameter_space(accumulator, r_min):
    acc_norm = np.zeros_like(accumulator)

    for i in range(accumulator.shape[2]):
        r = r_min + i
        sigma = 0.05 * r + 0.25
        smoothed = gaussian_filter(accumulator[:, :, i], sigma=sigma)
        acc_norm[:, :, i] = smoothed / (r + 1e-6)

    return acc_norm


# -------------------------------------------------
# Ring extraction
# -------------------------------------------------

def extract_ring_pixels(ridge_mask, ring, tolerance=2):
    x0, y0, r0 = ring["x"], ring["y"], ring["r"]

    ys, xs = np.where(ridge_mask)
    pts = []

    for y, x in zip(ys, xs):
        d = np.sqrt((x - x0)**2 + (y - y0)**2)
        if abs(d - r0) <= tolerance:
            pts.append([x, y])

    return np.array(pts)


def detect_einstein_rings_jwst(
    image,
    sigma=2.5,
    r_min=10,
    r_max=200,
    vote_frac=0.01,
    arc_tol=2
):
    """
    PURE FUNCTION:
    Input  : 2D numpy image
    Output : list of ring dictionaries
    """

    # --- Preprocessing ---
    img = subtract_background(image)
    img = normalize_image(img)
    img = gaussian_filter(img, sigma=1.0)

    # --- Ridge detection ---
    ridge_mask, gx, gy = detect_directed_ridges(img, sigma=sigma)

    # --- Directed Hough ---
    acc = directed_circle_hough(
        ridge_mask, gx, gy, r_min, r_max, img.shape
    )

    acc_norm = normalize_parameter_space(acc, r_min)

    threshold = vote_frac * np.max(acc_norm)

    peaks = peak_local_max(
        acc_norm,
        threshold_abs=threshold,
        footprint=np.ones((3, 3, 3))
    )

    rings = []

    for y, x, r_idx in peaks:
        r = r_min + r_idx

        ring = {
            "x": int(x),
            "y": int(y),
            "r": int(r),
            "score": float(acc_norm[y, x, r_idx])
        }

        ring["pixels"] = extract_ring_pixels(
            ridge_mask, ring, tolerance=arc_tol
        )

        ring["arc_fraction"] = (
            len(ring["pixels"]) / (2 * np.pi * r + 1e-6)
        )

        rings.append(ring)

    return rings


# -------------------------------------------------
# Test block (SAFE)
# -------------------------------------------------

if __name__ == "__main__":
    from astropy.io import fits
    import matplotlib.pyplot as plt

    with fits.open("jwst_image.fits") as hdul:
        image = hdul[1].data.astype(float)

    rings = detect_einstein_rings_jwst(image)

    print(f"Detected {len(rings)} candidate rings")

    plt.imshow(image, origin="lower", cmap="gray")
    for ring in rings:
        c = plt.Circle((ring["x"], ring["y"]), ring["r"],
                       color="red", fill=False)
        plt.gca().add_patch(c)
    plt.show()
