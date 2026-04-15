import numpy as np
import matplotlib.pyplot as plt

x = np.load("data/train/positive/x_5.npy")
y = np.load("data/train/positive/x_12.npy")
z = np.load("data/train/positive/x_18.npy")
a = np.load("data/train/positive/x_27.npy")
b = np.load("data/train/positive/x_31.npy")
c = np.load("data/train/positive/x_38.npy")
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(x[...,0], cmap="gray")
plt.colorbar()
plt.title("Observed1")

plt.subplot(1,2,2)
plt.imshow(x[...,2], cmap="hot")
plt.colorbar()
plt.title("Ring mask")

print(x.shape)
print("Unique values in channel 3:", np.unique(x[...,2])[:20])

plt.show()

plt.subplot(1,2,1)
plt.imshow(y[...,0], cmap="gray")
plt.colorbar()
plt.title("Observed2")

plt.subplot(1,2,2)
plt.imshow(y[...,2], cmap="hot")
plt.colorbar()
plt.title("Ring mask")

plt.show()

plt.subplot(1,2,1)
plt.imshow(z[...,0], cmap="gray")
plt.colorbar()
plt.title("Observed3")

plt.subplot(1,2,2)
plt.imshow(z[...,2], cmap="hot")
plt.colorbar()
plt.title("Ring mask")

plt.show()

plt.subplot(1,2,1)
plt.imshow(a[...,0], cmap="gray")
plt.colorbar()
plt.title("Observed4")

plt.subplot(1,2,2)
plt.imshow(a[...,2], cmap="hot")
plt.colorbar()
plt.title("Ring mask")

plt.show()

plt.subplot(1,2,1)
plt.imshow(b[...,0], cmap="gray")
plt.colorbar()
plt.title("Observed5")

plt.subplot(1,2,2)
plt.imshow(b[...,2], cmap="hot")
plt.colorbar()
plt.title("Ring mask")

plt.show()

plt.subplot(1,2,1)
plt.imshow(c[...,0], cmap="gray")
plt.colorbar()
plt.title("Observed6")

plt.subplot(1,2,2)
plt.imshow(c[...,2], cmap="hot")
plt.colorbar()
plt.title("Ring mask")

plt.show()