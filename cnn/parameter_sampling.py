import numpy as np

def sample_lens_params():
    return [
        {
            'theta_E': np.random.uniform(0.8, 1.5),
            'center_x': np.random.uniform(-0.2, 0.2),
            'center_y': np.random.uniform(-0.2, 0.2),
            'e1': np.random.uniform(-0.4, 0.4),
            'e2': np.random.uniform(-0.4, 0.4)
        },
        {
            'gamma1': np.random.normal(-0.2, 0.2),
            'gamma2': np.random.normal(-0.2, 0.2)
        }
    ]


def sample_source_params():
    return [{
        'amp': np.random.uniform(5, 30),
        'R_sersic': np.random.uniform(0.05, 0.5),
        'n_sersic': np.random.uniform(0.5, 4),
        'e1': np.random.uniform(-0.5, 0.5),
        'e2': np.random.uniform(-0.5, 0.5),
        'center_x': np.random.uniform(-0.3, 0.3),
        'center_y': np.random.uniform(-0.3, 0.3)
    }]


def sample_lens_light_params():
    return [{
        'amp': np.random.uniform(5, 30),
        'R_sersic': np.random.uniform(0.3, 1.0),
        'n_sersic': np.random.uniform(1, 4),
        'e1': np.random.uniform(-0.3, 0.3),
        'e2': np.random.uniform(-0.3, 0.3),
        'center_x': np.random.uniform(-0.1, 0.1),
        'center_y': np.random.uniform(-0.1, 0.1)
    }]
