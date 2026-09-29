import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import logging
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["preprocess"]

    os.makedirs("data/processed", exist_ok=True)
    raw_data = np.load("data/raw/fashion_mnist.npz")
    x_train_full, y_train_full = raw_data["x_train"], raw_data["y_train"]
    x_test, y_test = raw_data["x_test"], raw_data["y_test"]

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, 
        test_size=params["test_size"], 
        random_state=params["seed"]
    )

    np.savez_compressed("data/processed/data.npz", 
                        x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val,
                        x_test=x_test, y_test=y_test)
    print("Processed data saved to data/processed/data.npz")

if __name__ == "__main__":
    preprocess()