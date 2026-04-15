import numpy as np
import matplotlib.pyplot as plt

from cnn.simulators import simulate_lens_case
from statistical.mcmc_runner import run_mcmc_estimation

# ---- Generate synthetic lens ----
kwargs_spemd = {
    'theta_E': 0.8,
    'center_x': 0.0,
    'center_y': 0.0,
    'e1': 0.05,
    'e2': -0.02
}

kwargs_shear = {
    'gamma1': 0.01,
    'gamma2': 0.02
}

kwargs_lens = [kwargs_spemd, kwargs_shear]

kwargs_source = [{
    'amp': 15,
    'R_sersic': 0.1,
    'n_sersic': 1,
    'e1': 0.1,
    'e2': 0.05,
    'center_x': 0.1,
    'center_y': 0.0
}]

kwargs_lens_light = [{
    'amp': 10,
    'R_sersic': 0.5,
    'n_sersic': 2,
    'e1': 0.0,
    'e2': 0.0,
    'center_x': 0.0,
    'center_y': 0.0
}]

sample = simulate_lens_case(
    kwargs_lens,
    kwargs_source,
    kwargs_lens_light
)

image = sample["observed"]

# ---- Run PSO + MCMC ----
result = run_mcmc_estimation(image)

print("\n==== MCMC RESULT ====")
print("Einstein radius:", result["theta_E"])
print("Estimated lens mass:")

# ---- Show image ----
plt.imshow(np.log10(image), origin="lower", cmap="gray")
plt.title("Synthetic Einstein Ring")
plt.show()