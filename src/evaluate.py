import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate():
    data = np.load("data/processed/data.npz")
    x_test, y_test = data["x_test"], data["y_test"]

    model = tf.keras.models.load_model("models/model.h5")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print(f"Metrics written to metrics.json. Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    evaluate()