"""Exports the trained Keras model to TFLite (Android) and Core ML (iOS).

Run after train.py has produced handwriting_classifier.keras and cleared the
>90% test-accuracy gate.
"""
from __future__ import annotations

import coremltools as ct
import tensorflow as tf
from tensorflow import keras

IMG_H = 64
IMG_W = 128


def export_tflite(model: keras.Model, out_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {out_path} ({len(tflite_model) / 1024:.1f} KB)")


def export_coreml(model: keras.Model, out_path: str):
    # coremltools' TF2/Keras loader targets TF<=2.12's legacy tf.keras API and chokes on
    # Keras 3's Sequential (no `output_names` attribute), and a plain SavedModel export
    # carries multiple concrete functions/signatures which the loader also rejects.
    # Wrapping the model call in our own single tf.function sidesteps both: coremltools
    # accepts a list of exactly one concrete function directly.
    call_fn = tf.function(lambda x: model(x, training=False))
    concrete_fn = call_fn.get_concrete_function(tf.TensorSpec([1, IMG_H, IMG_W, 1], tf.float32, name="input"))

    mlmodel = ct.convert(
        [concrete_fn],
        source="tensorflow",
        inputs=[ct.ImageType(name="input", shape=(1, IMG_H, IMG_W, 1), scale=1 / 255.0, color_layout="G")],
        classifier_config=None,
        minimum_deployment_target=ct.target.iOS15,
    )
    mlmodel.short_description = "Printed vs handwriting classifier for word-crop images."
    mlmodel.save(out_path)
    print(f"Saved {out_path}")


def main():
    model = keras.models.load_model("handwriting_classifier.keras")

    export_tflite(model, "handwriting_classifier.tflite")
    export_coreml(model, "HandwritingClassifier.mlpackage")


if __name__ == "__main__":
    main()
