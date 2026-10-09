"""Builds per-component, per-page-normalised stroke features for all labelled pages (cached) — the
input of the colour-independent stroke classifier (see stroke_features.py)."""
import os
import cv2
import numpy as np
from label_real import resize_long, paper_level, ruled_lines, ink_chroma, INK_MARGIN, BATCH
from stroke_features import score_components, _robust_z

LABELS = os.path.join(BATCH, "labels")
FEATS = ["dark", "width", "slant", "irregular", "chroma", "height", "absdark", "abswidth", "v20"]


def page_features(p):
    src = cv2.imread(os.path.join(BATCH, p + ".jpg"))[..., ::-1]
    rgb, s = resize_long(np.ascontiguousarray(src))
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    c = paper_level(g).astype(np.int16) - g
    ink = c > INK_MARGIN
    strokes = ink & ~ruled_lines(ink)
    lab, n, f, _ = score_components(g, c, strokes)
    area = f["area"]
    flat = lab.ravel()
    ch = ink_chroma(rgb, ink)
    chroma = np.bincount(flat, weights=ch.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    ref = area >= 12
    ref[0] = False
    z = dict(f["z"])
    z["chroma"] = _robust_z(chroma, area, ref)
    z["height"] = _robust_z(f["height"], area, ref)
    z["absdark"], z["abswidth"] = np.abs(z["dark"]), np.abs(z["width"])
    v20p = os.path.join(LABELS, p + "_v20.npy")
    pv = np.load(v20p).astype(np.float32)[..., 1] if os.path.exists(v20p) else np.zeros(g.shape, np.float32)
    z["v20"] = np.bincount(flat, weights=pv.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    d = np.load(os.path.join(LABELS, p + ".npz"))
    yh = (d["y"] & 2) > 0
    valid = ((d["valid"] & 2) > 0) | ((d["valid"] & 1) > 0)
    hwfrac = np.bincount(flat, weights=yh.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    vfrac = np.bincount(flat, weights=valid.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    X = np.stack([z[k] for k in FEATS], 1).astype(np.float32)
    return {"X": X, "hwfrac": hwfrac, "vfrac": vfrac, "area": area, "lab": lab, "n": n}


if __name__ == "__main__":
    out = os.path.join(LABELS, "strokes")
    os.makedirs(out, exist_ok=True)
    for f in sorted(os.listdir(LABELS)):
        if f.startswith("p") and f.endswith(".npz") and len(f) == 7:
            p = f[:-4]
            r = page_features(p)
            np.savez_compressed(os.path.join(out, p + ".npz"), **{k: v for k, v in r.items() if k != "n"})
            print(p, r["n"], flush=True)
