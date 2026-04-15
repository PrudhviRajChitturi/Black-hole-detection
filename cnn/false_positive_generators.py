import numpy as np
from cnn.simulators import simulate_lens_case
from cnn.parameter_sampling import (
    sample_source_params,
    sample_lens_light_params,
    sample_lens_params
)

def false_positive_no_lens():
    """
    No lens mass, only source + lens light
    """
    kwargs_lens = []  # NO lens
    kwargs_source = sample_source_params()
    kwargs_lens_light = sample_lens_light_params()

    data = simulate_lens_case(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light
    )
    return data

def false_positive_wrong_lens():
    """
    Lens exists but parameters are wrong → bad residuals
    """
    kwargs_lens = sample_lens_params()

    # Intentionally break physics
    kwargs_lens[0]['theta_E'] *= np.random.uniform(0.2, 0.5)
    kwargs_lens[0]['center_x'] += np.random.uniform(0.2, 0.5)
    kwargs_lens[0]['center_y'] += np.random.uniform(0.2, 0.5)

    kwargs_source = sample_source_params()
    kwargs_lens_light = sample_lens_light_params()

    data = simulate_lens_case(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light
    )
    return data

def false_positive_noise_psf():
    """
    Strong noise / PSF mismatch
    """
    kwargs_lens = sample_lens_params()
    kwargs_source = sample_source_params()
    kwargs_lens_light = sample_lens_light_params()

    data = simulate_lens_case(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
        background_rms=np.random.uniform(0.01, 0.03)  # stronger noise
    )
    return data

