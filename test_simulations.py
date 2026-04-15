from cnn.simulators import simulate_lens_case
import matplotlib.pyplot as plt
import numpy as np

# ---- Define lens parameters ----
kwargs_spemd = {
    'theta_E': 0.66,
    'center_x': 0.05,
    'center_y': 0.0,
    'e1': 0.07,
    'e2': -0.03
}

kwargs_shear = {
    'gamma1': 0.0,
    'gamma2': -0.05
}

kwargs_lens = [kwargs_spemd, kwargs_shear]

# ---- Define source parameters ----
kwargs_sersic = {
    'amp': 16,
    'R_sersic': 0.1,
    'n_sersic': 1,
    'e1': -0.1,
    'e2': 0.1,
    'center_x': 0.1,
    'center_y': 0.0
}

kwargs_source = [kwargs_sersic]

# ---- Define lens light parameters ----
kwargs_sersic_lens = {
    'amp': 16,
    'R_sersic': 0.6,
    'n_sersic': 2,
    'e1': -0.1,
    'e2': 0.1,
    'center_x': 0.05,
    'center_y': 0.0
}

kwargs_lens_light = [kwargs_sersic_lens]

# ---- Run simulation ----
sample = simulate_lens_case(
    kwargs_lens=kwargs_lens,
    kwargs_source=kwargs_source,
    kwargs_lens_light=kwargs_lens_light,
    print("theta_E:", kwargs_lens[0]["theta_E"])
print("deltaPix:", deltaPix)
print("Einstein radius (pixels):", kwargs_lens[0]["theta_E"] / deltaPix)
)
    
    


# ---- Visual sanity check ----
plt.imshow(np.log10(sample["observed"]), origin="lower", cmap="gray")
plt.colorbar()
plt.title("Simulated Einstein Ring")
plt.show()
