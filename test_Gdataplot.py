import numpy as np
import matplotlib.pyplot as plt

x = np.load("data/synthetic/negative/x_15.npy")
plt.imshow(np.log10(x[...,1]), origin="lower")
plt.show()
