"""Ports a torch checkpoint of the ink segmenter into the Keras model (export source) and checks
both give the same output. Usage: venv/bin/python torch_to_keras.py in_torch.pt out.keras"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import torch

import ink_seg_model as M

src, dst = sys.argv[1], sys.argv[2]
tm = M.make_torch_model()
tm.load_state_dict(torch.load(src, map_location="cpu"))
tm.eval()
km = M.build_model()
n = M.port_torch_to_keras(tm, km)
x = np.random.default_rng(0).uniform(0, 255, (2, M.TILE, M.TILE, 3)).astype(np.float32)
with torch.no_grad():
    pt = torch.sigmoid(tm(torch.from_numpy(x).permute(0, 3, 1, 2))).permute(0, 2, 3, 1).numpy()
pk = np.asarray(km(x, training=False))
err = float(np.abs(pt - pk).max())
print(f"ported {n} layers, max |torch - keras| = {err:.2e}")
assert err < 1e-4
km.save(dst)
print("saved", dst)
