"""Trains the printed-vs-handwriting binary CNN on the synthetic dataset
produced by generate_dataset.py.

Gate: must clear >90% accuracy on the held-out synthetic test set before
export.py / app integration proceeds (see the approved plan).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_H = 64
IMG_W = 128


def load_split(name: str):
    data = np.load(f"{name}.npz")
    images = data["images"].astype("float32") / 255.0
    labels = data["labels"].astype("float32")
    return images, labels


def build_model() -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(IMG_H, IMG_W, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    return model


def main():
    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    test_x, test_y = load_split("test")
    print(f"train={train_x.shape} val={val_x.shape} test={test_x.shape}")

    model = build_model()
    model.summary()
    total_params = model.count_params()
    print(f"Total params: {total_params}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=40,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    print("\nEvaluating on held-out test set...")
    results = model.evaluate(test_x, test_y, verbose=0, return_dict=True)
    for k, v in results.items():
        print(f"  test {k}: {v:.4f}")

    model.save("handwriting_classifier.keras")
    print("\nSaved handwriting_classifier.keras")

    if results["accuracy"] < 0.90:
        print(f"\nGATE FAILED: test accuracy {results['accuracy']:.4f} < 0.90 — do not proceed to export/integration.")
    else:
        print(f"\nGATE PASSED: test accuracy {results['accuracy']:.4f} >= 0.90.")


if __name__ == "__main__":
    main()
