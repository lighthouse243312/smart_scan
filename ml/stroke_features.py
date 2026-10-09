"""Per-page stroke features that separate handwriting from print WITHOUT relying on ink colour.

Every page is its own reference: most ink on a worksheet is print, so robust statistics (median /
MAD over ink area) of the page's strokes describe ITS print — how dark, how thick, how upright —
and each stroke is scored by how far it departs from that. Ink colour is only the first, coarse
filter (label_real.py); these features decide where colour cannot (black pen on black print).

  darkness   median contrast to the local paper (print toner is uniform per page, pen pressure and
             pencil vary)
  width      median stroke width from the distance transform
  slant      shear of the stroke's near-vertical runs relative to the page's print, after removing
             the page tilt measured on the print rows (print is upright, handwriting leans)
  irregular  spread of width and darkness inside the stroke (toner is even, a pen is not)

score_components() returns, per connected stroke component, the robust z-scores and a combined
"departs from this page's print" score.
"""
from __future__ import annotations

import cv2
import numpy as np


def page_tilt(ink: np.ndarray) -> float:
    """Tilt (degrees) of the text rows: the angle whose horizontal projection is sharpest."""
    small = cv2.resize(ink.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best, best_a = -1.0, 0.0
    for a in np.arange(-4.0, 4.01, 0.25):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_NEAREST)
        prof = r.sum(1).astype(np.float64)
        s = float(np.var(prof))
        if s > best:
            best, best_a = s, a
    return best_a


def _robust_z(v: np.ndarray, weight: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """z of each value against the weighted median / MAD of the reference components."""
    vv, ww = v[ref], weight[ref]
    if len(vv) == 0:
        return np.zeros_like(v)
    o = np.argsort(vv)
    cw = np.cumsum(ww[o])
    med = vv[o][np.searchsorted(cw, cw[-1] / 2)]
    dev = np.abs(vv - med)
    do = np.argsort(dev)
    cw2 = np.cumsum(ww[do])
    mad = dev[do][np.searchsorted(cw2, cw2[-1] / 2)]
    return (v - med) / (1.4826 * mad + 1e-6)


def score_components(gray: np.ndarray, contrast: np.ndarray, strokes: np.ndarray, ref_mask: np.ndarray | None = None):
    """strokes: bool ink without ruled lines. ref_mask: pixels that may define the page's print
    (default: all strokes — print is the majority). Returns (labels, n, feats dict of per-component
    arrays, combined score per component)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(strokes.astype(np.uint8), connectivity=8)
    area = st[:, cv2.CC_STAT_AREA].astype(np.float64)
    hgt = st[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    flat = lab.ravel()

    def mean_of(x):
        return np.bincount(flat, weights=x.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)

    dist = cv2.distanceTransform(strokes.astype(np.uint8), cv2.DIST_L2, 3)
    # stroke width: twice the distance on the medial ridge (local maxima of the distance)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-3) & strokes
    rc = np.bincount(flat, weights=ridge.ravel().astype(np.float64), minlength=n)
    width = 2 * np.bincount(flat, weights=(dist * ridge).ravel(), minlength=n) / np.maximum(rc, 1)
    width_sq = 4 * np.bincount(flat, weights=(dist * dist * ridge).ravel(), minlength=n) / np.maximum(rc, 1)
    width_sd = np.sqrt(np.maximum(width_sq - width ** 2, 0))

    c = contrast.astype(np.float64) * strokes
    dark = mean_of(c)
    dark_sd = np.sqrt(np.maximum(mean_of(c * c) - dark ** 2, 0))

    # slant: shear of the near-vertical gradient structure, page tilt removed
    tilt = page_tilt(strokes)
    h, w = strokes.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.0)
    g = cv2.warpAffine(gray.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    labr = cv2.warpAffine(lab.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST).astype(np.int32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    # near-vertical strokes have mostly horizontal gradients; their lean is gy/gx there
    vert = (np.abs(gx) > 2.0 * np.abs(gy)) & (mag > 20) & (labr > 0)
    lean = np.where(vert, np.clip(gy / np.where(np.abs(gx) < 1e-3, 1e-3, gx), -2, 2), 0.0)
    lr = labr.ravel()
    vc = np.bincount(lr, weights=vert.ravel().astype(np.float64), minlength=n)
    slant = np.bincount(lr, weights=(lean * vert).ravel().astype(np.float64), minlength=n) / np.maximum(vc, 1)
    slant_ok = vc >= 6

    ref = np.ones(n, bool) if ref_mask is None else (np.bincount(flat, weights=ref_mask.ravel().astype(np.float64), minlength=n) > 0.5 * area)
    ref &= area >= 12
    ref[0] = False
    weight = area

    z = {
        "dark": _robust_z(dark, weight, ref),
        "width": _robust_z(width, weight, ref),
        "slant": np.where(slant_ok, _robust_z(np.abs(slant), weight, ref & slant_ok), 0.0),
        "irregular": _robust_z(width_sd / np.maximum(width, 1) + dark_sd / np.maximum(dark, 1), weight, ref),
    }
    # departures in either direction for darkness / width (a pen can be lighter OR darker than
    # toner), one-sided for slant and irregularity (print is the upright, even extreme)
    score = (np.maximum(np.abs(z["dark"]) - 1.5, 0) + np.maximum(np.abs(z["width"]) - 1.5, 0)
             + np.maximum(z["slant"] - 1.5, 0) + np.maximum(z["irregular"] - 1.5, 0))
    score[0] = 0
    feats = {"dark": dark, "width": width, "slant": slant, "area": area, "height": hgt, "tilt": tilt, "z": z}
    return lab, n, feats, score
