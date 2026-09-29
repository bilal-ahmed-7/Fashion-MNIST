import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import logging
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def prepare():
    os.makedirs("data/raw", exist_ok=True)
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    np.savez_compressed("data/raw/fashion_mnist.npz", 
                        x_train=x_train, y_train=y_train, 
                        x_test=x_test, y_test=y_test)
    print("Raw data saved to data/raw/fashion_mnist.npz")

if __name__ == "__main__":
    prepare()