from cnn.data_generation import (
    generate_positive_samples,
    generate_negative_samples
)
import matplotlib.pyplot as plt


generate_positive_samples(1500, "data/train/positive")
generate_negative_samples(1500, "data/train/negative")


generate_positive_samples(300, "data/test/positive")
generate_negative_samples(300, "data/test/negative")

