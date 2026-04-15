import numpy as np


def generate_ring_mask(shape, rings, thickness=2):
    """
    Generate a binary mask for detected Einstein rings.

    Parameters
    ----------
    shape : tuple
        Image shape (H, W)
    rings : list of dict
        Output from detect_einstein_rings_jwst
    thickness : int
        Ring thickness for geometric fallback

    Returns
    -------
    mask : 2D numpy array
        Binary mask
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    for ring in rings:
        # --- Preferred: use detected pixels ---
        if "pixels" in ring and len(ring["pixels"]) > 0:
            for x, y in ring["pixels"]:
                if 0 <= y < H and 0 <= x < W:
                    mask[int(y), int(x)] = 1.0

        # --- Fallback: idealized circular ring ---
        else:
            x0, y0, r0 = ring["x"], ring["y"], ring["r"]
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - x0)**2 + (yy - y0)**2)
            ring_region = np.logical_and(
                dist >= r0 - thickness,
                dist <= r0 + thickness
            )
            mask[ring_region] = 1.0

    return mask
