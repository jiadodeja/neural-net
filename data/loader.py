import numpy as np
from urllib.request import urlretrieve
import os
import gzip

def download_mnist(save_dir):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        dest = os.path.join(save_dir, file)
        if not os.path.exists(dest):
            print(f"Downloading {file}...")
            urlretrieve(base_url + file, dest)
            print(f"Saved to {dest}")
        else:
            print(f"Already exists: {file}")

def load_images(path):
    with gzip.open(path, 'rb') as f:
        f.read(16)  # skip header
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 784).astype(np.float32) / 255.0

def load_labels(path):
    with gzip.open(path, 'rb') as f:
        f.read(8)  # skip header
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)

def load_mnist(data_dir="data/mnist"):
    download_mnist(data_dir)

    X_train = load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test  = load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test  = load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    print(f"Training: {X_train.shape[0]} images")
    print(f"Testing:  {X_test.shape[0]} images")

    return X_train, y_train, X_test, y_test
