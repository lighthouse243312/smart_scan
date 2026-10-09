"""Per-page print profile: a second vote next to the segmentation model.

Each page has its own print: toner darkness, ink colour, stroke width (regular AND bold), slant
(the page's tilt removed) and text height. The profile is measured on the strokes the model is
sure are print, then every stroke is scored by how far it departs from it:

  colour     Lab a/b distance from the print's median colour, in units of the print's own spread
  darkness   robust z of the mean contrast to the local paper
  width      distance to the NEAREST print width mode (regular / bold), robust units
  slant      lean of near-vertical edges beyond the print's (print is upright)
  evenness   width / darkness spread inside the stroke (toner is even, a pen is not)
  height     stroke height / print x-height (handwriting rarely sits on the print grid)
  rules      share of ruled-line pixels around the stroke: features are measured on the stroke's
             own pixels only, the line is context (a pen mark ON a rule vs a rule fragment)

A small classifier (features + the model's own mean probability) decides per stroke; it only
overrides the model on strokes it is confident about and the model did not split itself.

Usage: venv/bin/python print_profile.py         (leave-one-page-out evaluation on all pages)
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from label_real import BATCH, INK_MARGIN, paper_level, resize_long, ruled_lines
from stroke_features import page_tilt

LABELS = os.path.join(BATCH, "labels")
CACHE = os.path.join(LABELS, "profile")
MODEL = "v21"
FEATS = ["colour", "dark", "width", "slant", "uneven", "height", "rules", "fill", "p_mean", "p_frac"]


def wmedian(v, w):
    if len(v) == 0:
        return 0.0
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, cw[-1] / 2)])


def wmad(v, w, med):
    return 1.4826 * wmedian(np.abs(v - med), w) + 1e-6


def width_modes(width, w):
    """Regular and bold print widths: two weighted medians split at the largest gap (bold only if
    it is a real second population, >= 8% of the print area and >= 1.35x the regular width)."""
    reg = wmedian(width, w)
    bold_side = width > 1.35 * reg
    if w[bold_side].sum() >= 0.08 * w.sum():
        return reg, wmedian(width[bold_side], w[bold_side])
    return reg, reg


def page_features(page):
    src = cv2.imread(os.path.join(BATCH, page + ".jpg"))[..., ::-1]
    rgb, _ = resize_long(np.ascontiguousarray(src))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contrast = paper_level(gray).astype(np.int16) - gray.astype(np.int16)
    ink = contrast > INK_MARGIN
    rules = ruled_lines(ink)
    strokes = ink & ~rules
    n, lab, st, _ = cv2.connectedComponentsWithStats(strokes.astype(np.uint8), connectivity=8)
    flat = lab.ravel()
    area = st[:, cv2.CC_STAT_AREA].astype(np.float64)
    bw, bh = st[:, cv2.CC_STAT_WIDTH].astype(np.float64), st[:, cv2.CC_STAT_HEIGHT].astype(np.float64)

    def mean(x):
        return np.bincount(flat, weights=x.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)

    labc = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    a_, b_ = mean(labc[..., 1]), mean(labc[..., 2])
    c = contrast.astype(np.float64) * strokes
    dark = mean(c)
    dark_sd = np.sqrt(np.maximum(mean(c * c) - dark ** 2, 0))
    dist = cv2.distanceTransform(strokes.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-3) & strokes
    rc = np.bincount(flat, weights=ridge.ravel().astype(np.float64), minlength=n)
    width = 2 * np.bincount(flat, weights=(dist * ridge).ravel(), minlength=n) / np.maximum(rc, 1)
    w2 = 4 * np.bincount(flat, weights=(dist * dist * ridge).ravel(), minlength=n) / np.maximum(rc, 1)
    width_sd = np.sqrt(np.maximum(w2 - width ** 2, 0))

    tilt = page_tilt(strokes)
    h, w = strokes.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.0)
    g = cv2.warpAffine(gray.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    labr = cv2.warpAffine(lab.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST).astype(np.int64).ravel()
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    vert = ((np.abs(gx) > 2 * np.abs(gy)) & (np.hypot(gx, gy) > 20)).ravel() & (labr > 0)
    lean = np.clip(gy.ravel() / np.where(np.abs(gx.ravel()) < 1e-3, 1e-3, gx.ravel()), -2, 2)
    vc = np.bincount(labr, weights=vert.astype(np.float64), minlength=n)
    slant = np.abs(np.bincount(labr, weights=lean * vert, minlength=n) / np.maximum(vc, 1))

    # ruled lines are context: share of rule pixels in a ring around the stroke
    near = cv2.dilate(strokes.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    ring_lab = cv2.dilate(lab.astype(np.float32), np.ones((7, 7), np.uint8)).astype(np.int64)
    ring = (near & ~strokes).ravel()
    rl = ring_lab.ravel()
    ring_n = np.bincount(rl[ring], minlength=n).astype(np.float64)
    ring_rule = np.bincount(rl[ring & rules.ravel()], minlength=n).astype(np.float64)
    rule_share = ring_rule / np.maximum(ring_n, 1)

    probs = np.load(os.path.join(LABELS, f"{page}_{MODEL}.npy")).astype(np.float32)[..., 1]
    p_mean = mean(probs)
    p_frac = mean((probs > 0.5).astype(np.float32))

    # the profile: strokes the model is sure are print, big enough to measure
    ref = (p_mean < 0.15) & (area >= 15) & (vc >= 4)
    ref[0] = False
    wr = area[ref]
    out = {}
    am, bm = wmedian(a_[ref], wr), wmedian(b_[ref], wr)
    cd = np.hypot(a_ - am, b_ - bm)
    out["colour"] = cd / wmad(cd[ref], wr, wmedian(cd[ref], wr))
    dm = wmedian(dark[ref], wr)
    out["dark"] = np.abs(dark - dm) / wmad(dark[ref], wr, dm)
    reg, bold = width_modes(width[ref], wr)
    wd = np.minimum(np.abs(width - reg), np.abs(width - bold))
    out["width"] = wd / wmad(wd[ref], wr, wmedian(wd[ref], wr))
    sm = wmedian(slant[ref & (vc >= 6)], area[ref & (vc >= 6)])
    out["slant"] = np.where(vc >= 6, (slant - sm) / wmad(slant[ref & (vc >= 6)], area[ref & (vc >= 6)], sm), 0.0)
    unev = width_sd / np.maximum(width, 1) + dark_sd / np.maximum(dark, 1)
    um = wmedian(unev[ref], wr)
    out["uneven"] = (unev - um) / wmad(unev[ref], wr, um)
    xh = wmedian(bh[ref], wr)
    out["height"] = bh / max(xh, 1)
    out["rules"] = rule_share
    out["fill"] = area / np.maximum(bw * bh, 1)
    out["p_mean"], out["p_frac"] = p_mean, p_frac
    X = np.stack([np.clip(out[k], -20, 20) for k in FEATS], 1).astype(np.float32)
    return {"X": X, "lab": lab.astype(np.int32), "area": area, "n": n, "strokes": strokes, "probs": probs}


def page_targets(page, lab, n, area):
    d = np.load(os.path.join(LABELS, page + ".npz"))
    yh = (d["y"] & 2) > 0
    valid = ((d["valid"] & 2) > 0) | ((d["valid"] & 1) > 0)
    flat = lab.ravel()
    hwfrac = np.bincount(flat, weights=yh.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    vfrac = np.bincount(flat, weights=valid.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    return hwfrac, vfrac


def fit(X, y, w, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    Xt, yt, wt = torch.tensor(X), torch.tensor(y), torch.tensor(np.sqrt(w).astype(np.float32))
    wt = torch.where(yt > 0, wt / (wt * yt).sum(), wt / (wt * (1 - yt)).sum())
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(800):
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(Xt)[:, 0], yt, reduction="none") * wt).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def combine(probs, lab, s, p_frac, hi=0.85, lo=0.15):
    """Model mask, with whole strokes overridden where the profile vote is confident and the
    model did not already split the stroke (mixed strokes: pen crossing print, keep the model)."""
    hw = probs > 0.5
    unsplit = (p_frac < 0.2) | (p_frac > 0.8)
    add = (s > hi) & unsplit
    rem = (s < lo) & unsplit
    add[0] = rem[0] = False
    return (hw | add[lab]) & ~rem[lab]


def main():
    from finetune_v7 import RealAcc, load_page, rfmt
    os.makedirs(CACHE, exist_ok=True)
    pages = sorted(f[:-4] for f in os.listdir(LABELS) if f.startswith("p") and f.endswith(".npz") and len(f) == 7)
    data = {}
    for p in pages:
        f = page_features(p)
        hwfrac, vfrac = page_targets(p, f["lab"], f["n"], f["area"])
        data[p] = (f, hwfrac, vfrac)
        print(p, end=" ", flush=True)
    print()

    def train_set(ps):
        Xs, ys, ws = [], [], []
        for q in ps:
            f, hwfrac, vfrac = data[q]
            m = (vfrac > 0.5) & (f["area"] >= 12)
            m[0] = False
            keep = m & ((hwfrac > 0.6) | (hwfrac < 0.05))
            Xs.append(f["X"][keep]); ys.append((hwfrac[keep] > 0.6).astype(np.float32)); ws.append(f["area"][keep])
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)

    tot_m, tot_c = RealAcc(), RealAcc()
    for p in pages:
        X, y, w = train_set([q for q in pages if q != p])
        net = fit(X, y, w)
        f, _, _ = data[p]
        with torch.no_grad():
            s = torch.sigmoid(net(torch.tensor(f["X"]))[:, 0]).numpy()
        page = load_page(p)
        model = np.stack([np.zeros_like(f["probs"]), f["probs"]], -1)
        comb = combine(f["probs"], f["lab"], s, f["X"][:, FEATS.index("p_frac")]).astype(np.float32)
        # the print channel is not changed by the vote: keep the model's print channel out of it
        a_m, a_c = RealAcc(), RealAcc()
        a_m.add(np.stack([np.ones_like(f["probs"]), f["probs"]], -1), page)
        a_c.add(np.stack([np.ones_like(f["probs"]), comb], -1), page)
        tot_m.add(np.stack([np.ones_like(f["probs"]), f["probs"]], -1), page)
        tot_c.add(np.stack([np.ones_like(f["probs"]), comb], -1), page)
        rm, rc = a_m.result(), a_c.result()
        print(f"{p}: model  rec={rm['hw_recall']:.4f} prec={rm['hw_precision']:.4f} kept={rm['print_kept']:.4f}   "
              f"+profile rec={rc['hw_recall']:.4f} prec={rc['hw_precision']:.4f} kept={rc['print_kept']:.4f}", flush=True)
        np.save(os.path.join(CACHE, p + "_vote.npy"), s.astype(np.float32))
    rm, rc = tot_m.result(), tot_c.result()
    print(f"TOTAL model  rec={rm['hw_recall']:.4f} prec={rm['hw_precision']:.4f} kept={rm['print_kept']:.4f}")
    print(f"TOTAL +prof  rec={rc['hw_recall']:.4f} prec={rc['hw_precision']:.4f} kept={rc['print_kept']:.4f}")


if __name__ == "__main__":
    main()
