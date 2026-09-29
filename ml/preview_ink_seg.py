"""Preview images for ink_segmenter_v2 (Keras model), written to ml/preview/inkseg_*.png.

  inkseg_tiles.png          test tiles:  input | ground truth | prediction
  inkseg_page_<k>.png       full synthetic test pages (tiled inference like the app)
  inkseg_real_<name>.png    real photo(s) in ml/real/: input | print prob | hw prob | overlay
  inkseg_real_<name>_crop.png  zoomed crop (EDUCATION heading area for IMG_0101)

Colour code for masks: white = paper, blue = print, red = handwriting, magenta = both.
Tiled inference mirrors the app: resize so the longer side = 1536, 256x256 tiles with 32 px
overlap (stride 224), edge tiles padded with white 255; overlapping predictions are blended with
a centre-weighted window.
"""
from __future__ import annotations

import glob
import math
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import cv2
import keras
import numpy as np

ML_DIR = os.path.dirname(os.path.abspath(__file__))
PREVIEW = os.path.join(ML_DIR, "preview")
TILE, STRIDE, LONG = 256, 224, 1536


def colorize(p, h):
    """p, h boolean masks -> RGB uint8"""
    out = np.full(p.shape + (3,), 255, np.uint8)
    out[p] = (70, 110, 255)
    out[h] = (235, 40, 40)
    out[p & h] = (200, 0, 200)
    return out


def resize_long(img, long=LONG):
    s = long / max(img.shape[:2])
    return cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def predict_page(model, img, bs=8):
    """img uint8 RGB (already at inference scale) -> probs float32 (H, W, 2)"""
    h, w = img.shape[:2]
    nx = max(1, math.ceil((w - TILE) / STRIDE) + 1)
    ny = max(1, math.ceil((h - TILE) / STRIDE) + 1)
    H, W = (ny - 1) * STRIDE + TILE, (nx - 1) * STRIDE + TILE
    pad = np.full((H, W, 3), 255, np.uint8)
    pad[:h, :w] = img
    ramp = np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE)).astype(np.float32)
    win = np.clip(np.minimum.outer(ramp, ramp) / 32.0, 0.05, 1.0)[..., None]
    acc = np.zeros((H, W, 2), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)
    coords = [(j * STRIDE, i * STRIDE) for j in range(ny) for i in range(nx)]
    for k in range(0, len(coords), bs):
        cs = coords[k:k + bs]
        batch = np.stack([pad[y:y + TILE, x:x + TILE] for y, x in cs]).astype(np.float32)
        pr = model.predict(batch, verbose=0)
        for (y, x), p in zip(cs, pr):
            acc[y:y + TILE, x:x + TILE] += p * win
            wsum[y:y + TILE, x:x + TILE] += win
    return (acc / np.maximum(wsum, 1e-6))[:h, :w]


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    return img


def save(path, rgb):
    cv2.imwrite(path, rgb[..., ::-1])
    print("wrote", path)


def heat(p):
    g = (255 - np.clip(p, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], -1)


def overlay(img, probs):
    o = img.astype(np.float32).copy()
    hw = probs[..., 1] > 0.5
    pr = (probs[..., 0] > 0.5) & ~hw
    o[hw] = o[hw] * 0.3 + np.array([255, 0, 0]) * 0.7
    o[pr] = o[pr] * 0.6 + np.array([0, 90, 255]) * 0.4
    return o.astype(np.uint8)


def main():
    model = keras.models.load_model(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ML_DIR, "ink_segmenter_v2.keras"))
    os.makedirs(PREVIEW, exist_ok=True)

    # --- test tiles -------------------------------------------------------------------------------
    d = np.load(os.path.join(ML_DIR, f"inkseg{os.environ.get('INKSEG_TAG', '')}_test.npz"))
    X, Y = d["images"], d["masks"]
    idx = np.random.default_rng(1).choice(len(X), 8, replace=False)
    pr = model.predict(X[idx].astype(np.float32), verbose=0)
    rows = []
    for i, k in enumerate(idx):
        gt = colorize((Y[k] & 1) > 0, (Y[k] & 2) > 0)
        pd = colorize(pr[i, ..., 0] > 0.5, pr[i, ..., 1] > 0.5)
        sep = np.full((TILE, 6, 3), 128, np.uint8)
        rows.append(np.concatenate([X[k], sep, gt, sep, pd], 1))
    grid = np.concatenate([np.concatenate(rows[:4], 0), np.full((4 * TILE, 12, 3), 128, np.uint8), np.concatenate(rows[4:], 0)], 1)
    save(os.path.join(PREVIEW, "inkseg_tiles.png"), grid)

    # --- full synthetic test pages ----------------------------------------------------------------
    fp = os.path.join(ML_DIR, f"inkseg{os.environ.get('INKSEG_TAG', '')}_fullpages_test.npz")
    if os.path.exists(fp):
        d = np.load(fp, allow_pickle=True)
        for k in range(min(3, len(d["images"]))):
            img, m = d["images"][k], d["masks"][k]
            probs = predict_page(model, img)
            gt = colorize((m & 1) > 0, (m & 2) > 0)
            pd = colorize(probs[..., 0] > 0.5, probs[..., 1] > 0.5)
            panel = np.concatenate([label(img, "input"), label(gt, "ground truth"), label(pd, "prediction")], 1)
            save(os.path.join(PREVIEW, f"inkseg_page_{k}.png"), resize_long(panel, 2400))

    # --- real photos ------------------------------------------------------------------------------
    for path in sorted(glob.glob(os.path.join(ML_DIR, "real", "*.png")) + glob.glob(os.path.join(ML_DIR, "real", "*.jpg"))):
        name = os.path.splitext(os.path.basename(path))[0].replace("_reading", "")
        img = cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1]
        img = np.ascontiguousarray(resize_long(img))
        probs = predict_page(model, img)
        panel = np.concatenate([label(img, "input"), label(heat(probs[..., 0]), "print prob"),
                                label(heat(probs[..., 1]), "handwriting prob"), label(overlay(img, probs), "overlay")], 1)
        save(os.path.join(PREVIEW, f"inkseg_real_{name}.png"), resize_long(panel, 3000))
        np.save(os.path.join(PREVIEW, f"_probs_{name}.npy"), probs.astype(np.float16))
        if "IMG_0101" in name:
            h, w = img.shape[:2]
            y0, y1, x0, x1 = int(h * 0.64), int(h * 0.82), int(w * 0.03), int(w * 0.99)  # EDUCATION heading area
            crop = [img[y0:y1, x0:x1], heat(probs[y0:y1, x0:x1, 0]), heat(probs[y0:y1, x0:x1, 1]), overlay(img, probs)[y0:y1, x0:x1]]
            names = ["input", "print prob", "handwriting prob", "overlay"]
            save(os.path.join(PREVIEW, f"inkseg_real_{name}_crop.png"),
                 np.concatenate([label(c, n) for c, n in zip(crop, names)], 0))


if __name__ == "__main__":
    main()
