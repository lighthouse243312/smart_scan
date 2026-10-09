"""Recovers a trainable PyTorch checkpoint from the DEPLOYED ink segmenter (.tflite).

Used when the training checkpoint of the shipped model is not on this machine (v6 was trained on
another Mac). TFLite stores each conv with its BatchNorm already folded in, so every
conv/depthwise layer gets the folded weight and its BN becomes an identity + bias
(gamma=1, beta=b, mean=0, var=1-eps). Fine-tune such a checkpoint with BatchNorm in eval mode
(frozen statistics) — see finetune_v7.py.

Usage: venv/bin/python recover_from_tflite.py [in.tflite] [out.pt]
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import tensorflow as tf
import torch

import ink_seg_model as M

ML_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE = os.path.join(ML_DIR, "..", "android", "app", "src", "main", "assets", "ink_segmenter.tflite")


def recover(tflite_path, out_path):
    it = tf.lite.Interpreter(model_path=tflite_path)
    it.allocate_tensors()
    td = {t["index"]: t["name"] for t in it.get_tensor_details()}
    all_ops = it._get_ops_details()
    ops = M.arch()
    layers = {op[1]: op for op in ops if op[0] in ("conv", "conv_bias", "dw")}
    bn_of = {op[2]: op[1] for op in ops if op[0] == "bn"}
    # dilated depthwise convs are not BN-folded: their BN stays as MUL(scale) + ADD(shift)
    loose = {}
    for o in all_ops:
        if o["op_name"] in ("MUL", "ADD") and len(o["inputs"]) == 2:
            cname = td[o["inputs"][1]]
            m = re.match(r"functional_1/(\w+?)_1/batchnorm/(mul|sub)$", cname)
            if m:
                loose[(m.group(1), m.group(2))] = it.get_tensor(o["inputs"][1])
    model = M.make_torch_model()
    sd = model.state_dict()
    done = set()
    for t in all_ops:
        if t["op_name"] not in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            continue
        names = td[t["inputs"][1]] + ";" + td[t["inputs"][2]]
        hits = [n for n in layers if f"/{n}_1/" in names and n not in done]
        hits = [n for n in hits if (layers[n][0] == "dw") == (t["op_name"] == "DEPTHWISE_CONV_2D")]
        assert len(hits) == 1, (names[:120], hits)
        name = hits[0]
        kind = layers[name][0]
        done.add(name)
        w = it.get_tensor(t["inputs"][1])
        b = it.get_tensor(t["inputs"][2])
        if kind == "dw":
            w = w[0].transpose(2, 0, 1)[:, None]          # (1,kh,kw,C) -> (C,1,kh,kw)
        else:
            w = w.transpose(0, 3, 1, 2)                   # (O,kh,kw,I) -> (O,I,kh,kw)
        if layers[name][2] == "rescaled":                 # the /255 input rescale is folded in here
            w = w * 255.0
        assert tuple(sd[f"m.{name}.weight"].shape) == w.shape, (name, w.shape)
        sd[f"m.{name}.weight"] = torch.from_numpy(np.ascontiguousarray(w))
        if kind == "conv_bias":
            sd[f"m.{name}.bias"] = torch.from_numpy(b.copy())
            continue
        bn = bn_of[name]
        c = b.shape[0]
        gamma = torch.ones(c)
        if (bn, "mul") in loose:
            assert not np.any(b), name
            gamma = torch.from_numpy(loose[(bn, "mul")].copy())
            b = loose[(bn, "sub")]
        sd[f"m.{bn}.weight"] = gamma
        sd[f"m.{bn}.bias"] = torch.from_numpy(b.copy())
        sd[f"m.{bn}.running_mean"] = torch.zeros(c)
        sd[f"m.{bn}.running_var"] = torch.full((c,), 1.0 - M.BN_EPS)
    assert done == set(layers), set(layers) - done
    model.load_state_dict(sd)
    model.eval()

    # numeric check against the tflite itself
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (1, M.TILE, M.TILE, 3)).astype(np.float32)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(inp["index"], x)
    it.invoke()
    ref = it.get_tensor(out["index"])[0]
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x).permute(0, 3, 1, 2)))[0].permute(1, 2, 0).numpy()
    err = float(np.abs(p - ref).max())
    print(f"recovered {len(layers)} layers, max |torch - tflite| = {err:.2e}")
    assert err < 1e-3
    torch.save(model.state_dict(), out_path)
    print("saved", out_path)


if __name__ == "__main__":
    recover(sys.argv[1] if len(sys.argv) > 1 else TFLITE,
            sys.argv[2] if len(sys.argv) > 2 else os.path.join(ML_DIR, "ink_segmenter_v6_torch.pt"))
