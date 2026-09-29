"""Exports ml/ink_segmenter_v2.keras to Core ML (iOS) and TFLite (Android) and verifies both
numerically against the Keras model.

  Core ML : ios/Runner/InkSegmenter.mlpackage
            mlprogram, iOS15+, FLOAT16 compute (ANE/GPU friendly; --fp32-coreml for exact),
            input  "input" MultiArray float32 [1,256,256,3] (RGB raw 0..255)
                               output "probs" MultiArray float32 [1,256,256,2] (sigmoid: print, hw)
  TFLite  : android/app/src/main/assets/ink_segmenter.tflite
            float32 weights by default (exact; --fp16-tflite for float16 weights, ~half size),
            input [1,256,256,3], output [1,256,256,2],
            signature "serving_default" with input key "input", output key "probs".

Core ML conversion trick (same as export.py): coremltools' TF loader chokes on Keras 3 models,
so we trace the model call inside a single tf.function and hand coremltools that one concrete
function.

Usage: venv/bin/python export_ink_seg.py [--fp32-coreml]
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import coremltools as ct
import keras
import numpy as np
import tensorflow as tf

ML_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ML_DIR)
KERAS_PATH = os.path.join(ML_DIR, "ink_segmenter_v2.keras")
COREML_PATH = os.path.join(ROOT, "ios", "Runner", "InkSegmenter.mlpackage")
TFLITE_PATH = os.path.join(ROOT, "android", "app", "src", "main", "assets", "ink_segmenter.tflite")
TILE = 256


def dir_size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


def export_tflite(model, fp16=False):
    # NB: from_concrete_functions() on a tf.Module wrapping a Keras 3 model silently drops the
    # variables (tiny file, all-NaN output) — from_keras_model() is the path that works. A dict
    # output gives the signature output key "probs"; the input key is the Keras input name.
    wrapped = keras.Model(model.inputs, {"probs": model.outputs[0]})
    conv = tf.lite.TFLiteConverter.from_keras_model(wrapped)
    if fp16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    data = conv.convert()
    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    with open(TFLITE_PATH, "wb") as f:
        f.write(data)
    print(f"saved {TFLITE_PATH} ({len(data) / 1024:.0f} KB, {'float16' if fp16 else 'float32'} weights)")


def export_coreml(model, fp32=False):
    fn = tf.function(lambda x: model(x, training=False))
    cf = fn.get_concrete_function(tf.TensorSpec([1, TILE, TILE, 3], tf.float32, name="input"))
    ml = ct.convert(
        [cf],
        source="tensorflow",
        inputs=[ct.TensorType(name="input", shape=(1, TILE, TILE, 3), dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32 if fp32 else ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
    )
    spec = ml.get_spec()
    out_name = spec.description.output[0].name
    if out_name != "probs":
        ct.utils.rename_feature(spec, out_name, "probs")
        ml = ct.models.MLModel(spec, weights_dir=ml.weights_dir)
    ml.short_description = ("Ink segmenter v2: per-pixel multi-label sigmoid probabilities "
                            "[print ink, handwriting ink] for a 256x256 RGB tile (raw 0..255).")
    ml.input_description["input"] = "RGB tile, float32 [1,256,256,3], raw 0..255 (NHWC)"
    ml.output_description["probs"] = "float32 [1,256,256,2]: channel 0 = printed ink, channel 1 = handwriting ink"
    ml.author = "beacon_smart_scan ml/"
    ml.version = "2.0"
    if os.path.exists(COREML_PATH):
        import shutil
        shutil.rmtree(COREML_PATH)
    ml.save(COREML_PATH)
    print(f"saved {COREML_PATH} ({dir_size(COREML_PATH) / 1024:.0f} KB), outputs={[o.name for o in ml.get_spec().description.output]}")


def sample_tiles(n=6):
    p = os.path.join(ML_DIR, f"inkseg{os.environ.get('INKSEG_TAG', '')}_test.npz")
    if os.path.exists(p):
        X = np.load(p)["images"]
        idx = np.linspace(0, len(X) - 1, n).astype(int)
        return X[idx].astype(np.float32)
    return (np.random.rand(n, TILE, TILE, 3) * 255).astype(np.float32)


def verify(model):
    xs = sample_tiles()
    ref = model.predict(xs, verbose=0)
    # TFLite
    it = tf.lite.Interpreter(model_path=TFLITE_PATH)
    it.allocate_tensors()
    i_det, o_det = it.get_input_details()[0], it.get_output_details()[0]
    outs = []
    for x in xs:
        it.set_tensor(i_det["index"], x[None])
        it.invoke()
        outs.append(it.get_tensor(o_det["index"]))
    tfl = np.concatenate(outs)
    print(f"  tflite signatures: {it.get_signature_list()}")
    d_tfl = np.abs(tfl - ref)
    print(f"TFLite  vs Keras: max abs diff {d_tfl.max():.5f}  mean {d_tfl.mean():.2e}  "
          f"thresholded-mask agreement {((tfl > 0.5) == (ref > 0.5)).mean():.5%}")
    det = it.get_input_details()[0], it.get_output_details()[0]
    print(f"  tflite tensors: in {det[0]['name']} {det[0]['shape']} {det[0]['dtype'].__name__} | "
          f"out {det[1]['name']} {det[1]['shape']} {det[1]['dtype'].__name__}")
    # Core ML
    ml = ct.models.MLModel(COREML_PATH)
    cm = np.concatenate([ml.predict({"input": x[None]})["probs"] for x in xs])
    d_cm = np.abs(cm - ref)
    print(f"CoreML  vs Keras: max abs diff {d_cm.max():.5f}  mean {d_cm.mean():.2e}  "
          f"thresholded-mask agreement {((cm > 0.5) == (ref > 0.5)).mean():.5%}")
    return d_tfl.max(), d_cm.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-coreml", action="store_true")
    ap.add_argument("--fp16-tflite", action="store_true")
    ap.add_argument("--keras", default=KERAS_PATH, help="Keras model to export (default ink_segmenter_v2.keras)")
    args = ap.parse_args()
    model = keras.models.load_model(args.keras)
    print(f"loaded {args.keras}: params {model.count_params()}, in {model.input_shape}, out {model.output_shape}")
    export_tflite(model, fp16=args.fp16_tflite)
    export_coreml(model, fp32=args.fp32_coreml)
    verify(model)


if __name__ == "__main__":
    main()
