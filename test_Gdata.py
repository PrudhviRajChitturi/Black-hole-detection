from cnn.data_generation import generate_dataset

generate_dataset(
    n_samples=20,
    save_dir="data/synthetic/positive",
    label=1
)

generate_dataset(
    n_samples=20,
    save_dir="data/synthetic/negative",
    label=0
)
    