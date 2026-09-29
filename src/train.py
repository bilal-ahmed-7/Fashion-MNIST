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
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

def train():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    os.makedirs("models", exist_ok=True)
    data = np.load("data/processed/data.npz")
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]

    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(params["dense_units"], activation="relu"),
        layers.Dropout(params["dropout_rate"]),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"]
    )

    model.save("models/model.keras")
    pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
    print("Model saved to models/model.keras and training history to models/history.csv")

if __name__ == "__main__":
    train()