"""Trains a small U-Net to predict, per-pixel, which ink is handwriting (to be erased) vs
printed text/background — replaces the hand-tuned redness+dilate+connected-component erase-
region heuristic in Inpainter.kt/.mm, which still missed parts of circles/ticks and only
worked for red ink.

Gate: mean IoU > 0.6 on the held-out synthetic test set (segmentation masks are inherently
fuzzier at the boundary than a classification label, so this is a looser bar than train.py's
90% accuracy gate — a perfect pixel boundary isn't the goal, "roughly the right ink blob" is).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEG_SIZE = 128


def load_split(name: str):
    data = np.load(f"seg_{name}.npz")
    images = data["images"].astype("float32") / 255.0
    masks = (data["masks"].astype("float32") / 255.0 > 0.5).astype("float32")
    return images, masks


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_model() -> keras.Model:
    inputs = keras.Input(shape=(SEG_SIZE, SEG_SIZE, 1))

    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D(2)(c3)

    b = conv_block(p3, 128)

    u3 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(b)
    u3 = layers.Concatenate()([u3, c3])
    d3 = conv_block(u3, 64)

    u2 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 32)

    u1 = layers.Conv2DTranspose(16, 2, strides=2, padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = conv_block(u1, 16)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)

    model = keras.Model(inputs, outputs)
    return model


def iou_metric(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2, 3])
    union = tf.reduce_sum(tf.maximum(y_true, y_pred_bin), axis=[1, 2, 3])
    return tf.reduce_mean((intersection + 1e-6) / (union + 1e-6))


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def combined_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def main():
    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    test_x, test_y = load_split("test")
    print(f"train={train_x.shape} val={val_x.shape} test={test_x.shape}")

    model = build_model()
    model.summary()
    print(f"Total params: {model.count_params()}")

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=combined_loss, metrics=[iou_metric])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_iou_metric", mode="max", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=40,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    print("\nEvaluating on held-out test set...")
    results = model.evaluate(test_x, test_y, verbose=0, return_dict=True)
    for k, v in results.items():
        print(f"  test {k}: {v:.4f}")

    model.save("ink_segmenter.keras")
    print("\nSaved ink_segmenter.keras")

    if results["iou_metric"] < 0.6:
        print(f"\nGATE FAILED: test IoU {results['iou_metric']:.4f} < 0.60")
    else:
        print(f"\nGATE PASSED: test IoU {results['iou_metric']:.4f} >= 0.60")


if __name__ == "__main__":
    main()
