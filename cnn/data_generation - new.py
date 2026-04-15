import os
import numpy as np

from cnn.simulators import simulate_lens_case
from cnn.parameter_sampling import (
    sample_lens_params,
    sample_source_params,
    sample_lens_light_params
)
from cnn.false_positive_generators import (
    false_positive_no_lens,
    false_positive_wrong_lens,
    false_positive_noise_psf
)


def save_sample(x, y, idx, save_dir):
    np.save(os.path.join(save_dir, f"x_{idx}.npy"), x)
    np.save(os.path.join(save_dir, f"y_{idx}.npy"), y)


def generate_positive_samples(n_samples, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_samples):

        kwargs_lens = sample_lens_params()
        kwargs_source = sample_source_params()
        kwargs_lens_light = sample_lens_light_params()

        data = simulate_lens_case(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light
        )

        observed = data["observed"]
        model = data["model"]
        source = data["source"]

        residual = observed - model

        # --- TRUE PHYSICS MASK ---
        threshold = 0.05 * np.max(source)
        mask = (source > threshold).astype(np.float32)

        x = np.stack([observed, model, residual, mask], axis=-1)

        save_sample(x, 1, i, save_dir)

        print(f"[{i}] positive generated")


def generate_negative_samples(n_samples, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    generators = [
        false_positive_no_lens,
        false_positive_wrong_lens,
        false_positive_noise_psf
    ]

    for i in range(n_samples):

        gen = np.random.choice(generators)
        data = gen()

        observed = data["observed"]
        model = data["model"]

        residual = observed - model

        mask = np.zeros_like(observed)

        x = np.stack([observed, model, residual, mask], axis=-1)

        save_sample(x, 0, i, save_dir)

        print(f"[{i}] negative generated")
