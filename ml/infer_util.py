"""Shared page-level inference for the ink segmenter (same tiling as the app: long side 1536,
256 px tiles, stride 224, triangular blending)."""
from __future__ import annotations

import math

import numpy as np
import torch

TILE, STRIDE = 256, 224


def predict_page(model, img, dev, bs=16):
    """img RGB uint8 (H,W,3) -> probs float32 (H,W,2) [print, hw]."""
    h, w = img.shape[:2]
    nx = max(1, math.ceil((w - TILE) / STRIDE) + 1)
    ny = max(1, math.ceil((h - TILE) / STRIDE) + 1)
    H, W = (ny - 1) * STRIDE + TILE, (nx - 1) * STRIDE + TILE
    pad = np.pad(img, ((0, H - h), (0, W - w), (0, 0)), mode="edge")
    acc = np.zeros((H, W, 2), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)
    ramp = np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE)).astype(np.float32)
    win = np.minimum(ramp[:, None], ramp[None, :])[..., None]
    coords = [(j * STRIDE, i * STRIDE) for j in range(ny) for i in range(nx)]
    was = model.training
    model.eval()
    with torch.no_grad():
        for k in range(0, len(coords), bs):
            cs = coords[k:k + bs]
            batch = np.stack([pad[y:y + TILE, x:x + TILE] for y, x in cs]).astype(np.float32)
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(dev)
            p = torch.sigmoid(model(x)).permute(0, 2, 3, 1).cpu().numpy()
            for (y, x0), pi in zip(cs, p):
                acc[y:y + TILE, x0:x0 + TILE] += pi * win
                wsum[y:y + TILE, x0:x0 + TILE] += win
    model.train(was)
    return (acc / wsum)[:h, :w]
