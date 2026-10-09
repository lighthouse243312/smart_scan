"""Fine-tunes the shipped ink segmenter v6 -> v7 on REAL worksheet photos (ml/real/batch2), mixed
with synthetic tiles so overlap / black-ink behaviour is not forgotten.

  init   : ink_segmenter_v6_torch.pt (recovered from the deployed .tflite, BN folded -> BatchNorm
           layers are kept in eval mode: frozen statistics, affine params still train)
  labels : ml/real/batch2/labels/<page>.npz from label_real.py (y bit0 print / bit1 hw, valid)
  split  : TRAIN pages / VAL page (checkpoint choice) / TEST pages (report only, never trained)

Criteria (threshold 0.5, on valid pixels of the TEST pages, all must be > 0.90):
  hw_recall      handwriting pixels that get erased
  hw_precision   erased pixels that really are handwriting
  print_kept     printed-only pixels NOT predicted as handwriting  (= 1 - print->hw FP rate)
  iou_hw, iou_print

Usage: venv/bin/python finetune_v7.py [--steps 3000 --bs 16 --lr 3e-4] [--all]
  --all : also train on VAL+TEST pages (final model, after the held-out numbers were checked)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch

import ink_seg_model as M
from finetune_real_pseudo import TILE, masked_loss, sample_real_tile
from infer_util import predict_page
from train_ink_seg import augment, evaluate, fmt, load, loss_fn, to_targets

ML_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(ML_DIR, "real", "batch2", "labels")
PREVIEW = os.path.join(ML_DIR, "preview")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
VAL = ["p04"]
TEST = ["p02", "p05", "p11"]
CRITERIA = ["hw_recall", "hw_precision", "print_kept", "iou_hw", "iou_print"]


def all_pages():
    return sorted(f[:-4] for f in os.listdir(LABELS) if f.endswith(".npz"))


def load_page(name):
    d = np.load(os.path.join(LABELS, name + ".npz"))
    y = np.stack([(d["y"] & 1) > 0, (d["y"] & 2) > 0], -1).astype(np.float32)
    v = np.stack([(d["valid"] & 1) > 0, (d["valid"] & 2) > 0], -1).astype(np.float32)
    return {"name": name, "images": [d["image"]], "y": y, "valid": v,
            "ink": (d["ink"] > 0) if "ink" in d else np.ones(y.shape[:2], bool),
            "hw_pts": np.argwhere((y[..., 1] > 0) & (v[..., 1] > 0))}


def scan_like(img):
    """CamScanner-style version of a photo: flat white paper, boosted ink contrast."""
    f = img.astype(np.float32)
    bg = np.stack([cv2.medianBlur(cv2.dilate(f[..., c].astype(np.uint8), np.ones((15, 15), np.uint8)), 31)
                   for c in range(3)], -1).astype(np.float32)
    r = np.clip(f / np.maximum(bg, 1), 0, 1)
    r = r ** 1.6
    return np.ascontiguousarray((r * 255).astype(np.uint8))


def train_page(name):
    p = load_page(name)
    p["images"].append(scan_like(p["images"][0]))
    p["print_pts"] = np.argwhere((p["y"][..., 0] > 0) & (p["valid"][..., 0] > 0))
    return p


PEN_RGB = [(20, 20, 25), (35, 35, 45), (25, 40, 110), (20, 30, 75), (60, 40, 120), (110, 40, 140),
           (180, 30, 40), (150, 40, 90), (40, 70, 160)]


def tile_paper(x):
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(cv2.dilate(g, np.ones((11, 11), np.uint8)), 21)
    return np.maximum(bg.astype(np.float32), 1)


def paste_hw(dst, src, rng):
    """multiply-blend the handwriting strokes of tile `src` onto tile `dst` (x, y, v tuples).
    Ink darkness comes from the source pixel vs its local paper level; the pen colour is kept
    or replaced by a random pen colour (black / blue-black / blue / purple / red)."""
    xd, yd, vd = dst
    xs, ys, vs = src
    hw = (ys[..., 1] > 0.5) & (vs[..., 1] > 0.5)
    if hw.sum() < 30:
        return dst
    paper = tile_paper(xs)
    gray = cv2.cvtColor(xs, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dark = np.clip(1 - gray / paper, 0, 1)                              # 0 paper .. 1 black
    m = cv2.GaussianBlur(cv2.dilate(hw.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 0.7)
    m = np.clip(m * 1.3, 0, 1) * ((ys[..., 0] < 0.5) | hw)                 # never carry source print
    if rng.random() < 0.6:
        col = np.array(PEN_RGB[rng.integers(len(PEN_RGB))], np.float32) / 255.0
        col = np.clip(col * rng.uniform(0.8, 1.2), 0, 1)
        dark = np.clip(dark * rng.uniform(0.9, 1.4), 0, 1)
        ratio = 1 - dark[..., None] * (1 - col[None, None])
    else:
        ratio = np.clip(xs.astype(np.float32) / paper[..., None], 0, 1)
    out = xd.astype(np.float32) * (1 - m[..., None] * (1 - ratio))
    new_hw = hw & (dark > 0.12)
    y = yd.copy()
    y[..., 1] = np.maximum(y[..., 1], new_hw.astype(np.float32))
    v = vd.copy()
    band = cv2.dilate(new_hw.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~new_hw
    v[..., 1] = v[..., 1] * ~band
    v[..., 0] = v[..., 0] * ~(band | new_hw)        # print under fresh ink: not scored
    return np.clip(out, 0, 255).astype(np.uint8), y, v


def _shape_points(rng, cx, cy, rx, ry):
    kind = rng.choice(["cloud", "cloud", "roundrect", "star", "bubble"])
    t = np.linspace(0, 2 * np.pi, 400)
    if kind == "cloud":
        k = int(rng.integers(6, 13))
        amp = rng.uniform(0.1, 0.25)
        r = 1 + amp * np.abs(np.sin(k * t / 2))
        pts = np.stack([cx + rx * r * np.cos(t), cy + ry * r * np.sin(t)], 1)
        if rng.random() < 0.8:   # workbook clouds: a small inward curl at every scallop joint
            hooks = []
            for j in range(k):
                a = 2 * np.pi * j / k
                L = rng.uniform(0.08, 0.16)
                s_ = np.linspace(0, 1, 12)
                rr = 1 - L * s_
                aa = a + rng.choice([-1, 1]) * 0.25 * L * np.sin(np.pi * s_ / 2) * 3
                hooks.append(np.stack([cx + rx * rr * np.cos(aa), cy + ry * rr * np.sin(aa)], 1))
            return [pts] + hooks, True
    elif kind == "ellipse":
        pts = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)
    elif kind == "star":
        k = int(rng.integers(5, 8))
        r = np.where((np.floor(t / (np.pi / k)) % 2) == 0, 1.0, rng.uniform(0.4, 0.6))
        pts = np.stack([cx + rx * r * np.cos(t), cy + ry * r * np.sin(t)], 1)
    elif kind == "wave":
        x = np.linspace(cx - rx, cx + rx, 300)
        pts = np.stack([x, cy + ry * 0.25 * np.sin((x - cx) / rx * np.pi * rng.uniform(3, 8))], 1)
        return pts, False
    elif kind == "bubble":
        pts = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)
        tail = np.array([[cx + rx * 0.3, cy + ry * 0.95], [cx + rx * 0.6, cy + ry * 1.5], [cx + rx * 0.55, cy + ry * 0.85]])
        pts = np.concatenate([pts, tail])
    else:
        rr = min(rx, ry) * rng.uniform(0.2, 0.5)
        sq = np.stack([np.clip(np.cos(t) * 1.4, -1, 1), np.clip(np.sin(t) * 1.4, -1, 1)], 1)
        pts = np.stack([cx + (rx - rr) * sq[:, 0] + rr * np.cos(t), cy + (ry - rr) * sq[:, 1] + rr * np.sin(t)], 1)
    return pts, True


def add_print_shapes(tile, rng):
    """draw workbook-style printed outlines (clouds, bubbles, stars, rounded boxes...) on a real
    tile, labelled print, so smooth decorative curves are not taken for handwriting."""
    x, y, v = tile
    x = x.copy(); y = y.copy(); v = v.copy()
    hw = y[..., 1] > 0.5
    for _ in range(int(rng.integers(1, 3))):
        cx, cy = rng.uniform(20, 236, 2)
        rx, ry = rng.uniform(25, 110), rng.uniform(15, 70)
        pts, closed = _shape_points(rng, cx, cy, rx, ry)
        m = np.zeros((TILE, TILE), np.uint8)
        th = int(rng.integers(1, 3))
        dotted = rng.random() < 0.2
        parts = pts if isinstance(pts, list) else [pts]
        for q, part in enumerate(parts):
            P = np.round(part * 4).astype(np.int32)
            if dotted and q == 0:
                for i in range(0, len(P) - 1, 6):
                    cv2.line(m, tuple(P[i]), tuple(P[min(i + 2, len(P) - 1)]), 255, th * 4, cv2.LINE_AA, shift=2)
            else:
                cv2.polylines(m, [P], closed and q == 0, 255, th * 4, cv2.LINE_AA, shift=2)
        a = (m.astype(np.float32) / 255.0) * ~hw
        col = np.array([rng.uniform(10, 70)] * 3, np.float32) if rng.random() < 0.7 else \
            np.array([rng.uniform(20, 60), rng.uniform(40, 90), rng.uniform(120, 200)], np.float32)
        x = (x.astype(np.float32) * (1 - a[..., None]) + col * a[..., None] * (x.astype(np.float32) / 255.0)).clip(0, 255).astype(np.uint8)
        on = a > 0.5
        y[..., 0] = np.maximum(y[..., 0], on.astype(np.float32))
        v[..., 0] = np.maximum(v[..., 0], on.astype(np.float32))
        v[..., 1] = np.maximum(v[..., 1], on.astype(np.float32))
        ring = (a > 0.05) & ~on
        v[..., 0] = v[..., 0] * ~ring
    return x, y, v


def _pen_render(x, m, rng):
    """multiply-blend pen mask m (0..1 float) onto image x: near-black ballpoint most of the time
    (the hard case — the same colour as print), else a coloured pen."""
    if rng.random() < 0.6:
        g = rng.uniform(15, 80)
        col = np.array([g, g, g * rng.uniform(1.0, 1.5)], np.float32)
    else:
        col = np.array(PEN_RGB[rng.integers(len(PEN_RGB))], np.float32) * rng.uniform(0.8, 1.1)
    ratio = np.clip(col / 255.0, 0, 1)
    out = x.astype(np.float32) * (1 - m[..., None] * (1 - ratio[None, None]))
    return np.clip(out, 0, 255).astype(np.uint8)


def add_pen_marks(tile, rng):
    """answer marks drawn over real print: pen circles round single printed letters (often touching
    or crossing them), strike-throughs / underlines / scribbles across printed words, and ticks. Labelled exactly: the pen is handwriting; a printed
    letter stays print EVERYWHERE, under the pen too, and is scored there — the model learns which
    pixels are print hidden under a pen (to be redrawn after the erase) and which are only pen."""
    x, y, v = tile
    x = x.copy(); y = y.copy(); v = v.copy()
    printed = (y[..., 0] > 0.5) & (y[..., 1] < 0.5)
    n, lab, st, _ = cv2.connectedComponentsWithStats(printed.astype(np.uint8), connectivity=8)
    letters = [i for i in range(1, n) if 20 <= st[i, 4] and 7 <= st[i, 3] <= 45 and st[i, 2] <= 50]
    # printed text rows: letters on a shared baseline, side by side (for strike-throughs / underlines)
    rows = []
    for i in sorted(letters, key=lambda i: st[i, 0]):
        for r in rows:
            j = r[-1]
            if abs((st[i, 1] + st[i, 3]) - (st[j, 1] + st[j, 3])) <= 0.25 * max(st[i, 3], st[j, 3]) and 0 <= st[i, 0] - (st[j, 0] + st[j, 2]) <= 1.2 * st[j, 3]:
                r.append(i); break
        else:
            rows.append([i])
    rows = [r for r in rows if len(r) >= 2]
    pen = np.zeros((TILE, TILE), np.float32)
    for _ in range(int(rng.integers(1, 4))):
        m = np.zeros((TILE, TILE), np.uint8)
        th = max(1, int(round(rng.uniform(1.3, 3.2))))   # (thickness is in pixels, not shifted)
        kind0 = rng.random()
        if letters and kind0 < 0.35:                # circle round a letter
            i = letters[rng.integers(len(letters))]
            bx, by, bw, bh = st[i, :4]
            cx, cy = bx + bw / 2, by + bh / 2
            rx = bw / 2 + rng.uniform(2, 12); ry = bh / 2 + rng.uniform(2, 10)
            r = max(rx, ry) if rng.random() < 0.5 else None
            if r is not None:
                rx = ry = r
            cx += rng.uniform(-0.35, 0.35) * rx; cy += rng.uniform(-0.3, 0.3) * ry   # often touches the letter
            a0 = rng.uniform(0, 2 * np.pi); sweep = rng.uniform(1.6, 2.25) * np.pi
            t = np.linspace(a0, a0 + sweep, 200)
            wob = 1 + rng.uniform(0.02, 0.12) * np.sin(t * rng.uniform(1, 3) + rng.uniform(0, 6))
            rr = 1 + (t - a0) / sweep * rng.uniform(-0.15, 0.15)            # spiral: the ends miss
            P = np.stack([cx + rx * wob * rr * np.cos(t), cy + ry * wob * rr * np.sin(t)], 1)
        elif rows and kind0 < 0.8:                   # strike-through / underline / scribble over printed words
            r = rows[rng.integers(len(rows))]
            ids = [j for j in r]
            a = int(rng.integers(len(ids))); b = min(len(ids), a + int(rng.integers(2, 8)))
            sel = ids[a:b]
            x0 = min(st[j, 0] for j in sel) - rng.uniform(0, 6); x1 = max(st[j, 0] + st[j, 2] for j in sel) + rng.uniform(0, 6)
            top = np.median([st[j, 1] for j in sel]); bot = np.median([st[j, 1] + st[j, 3] for j in sel])
            kind = rng.random()
            t = np.linspace(0, 1, 120)
            if kind < 0.55:                          # strike-through: through the middle of the letters
                ym = top + (bot - top) * rng.uniform(0.35, 0.65)
                slope = rng.uniform(-0.06, 0.06) * (x1 - x0)
                P = np.stack([x0 + (x1 - x0) * t, ym + slope * (t - 0.5) + rng.uniform(0.3, 1.2) * np.sin(t * rng.uniform(3, 9))], 1)
                if rng.random() < 0.25:              # struck twice
                    P2 = P + np.array([0, rng.uniform(2, 4)])
                    cv2.polylines(m, [np.round(P2 * 4).astype(np.int32)], False, 255, th, cv2.LINE_AA, shift=2)
            elif kind < 0.8:                         # underline just under the baseline (touching it at times)
                yb = bot + rng.uniform(-1, 4)
                P = np.stack([x0 + (x1 - x0) * t, yb + rng.uniform(-1.5, 1.5) * (t - 0.5) + rng.uniform(0.2, 1.0) * np.sin(t * rng.uniform(2, 6))], 1)
            else:                                    # scribbled out: zigzag over the word
                nz = int(rng.integers(3, 8))
                xs = np.linspace(x0, x1, nz * 2 + 1)
                ys = np.where(np.arange(len(xs)) % 2 == 0, top + rng.uniform(-2, 3), bot + rng.uniform(-3, 2))
                P = np.stack([xs, ys], 1)
        else:                                        # tick through a word
            cx, cy = rng.uniform(30, 226, 2)
            s = rng.uniform(10, 35)
            P = np.array([[cx - 0.4 * s, cy - 0.3 * s], [cx, cy + 0.4 * s], [cx + rng.uniform(1.0, 2.2) * s, cy - rng.uniform(0.6, 1.4) * s]])
            t = np.linspace(0, 1, 60)
            P = np.concatenate([P[0] + (P[1] - P[0]) * t[:, None], P[1] + (P[2] - P[1]) * t[:, None]])
        cv2.polylines(m, [np.round(P * 4).astype(np.int32)], False, 255, th, cv2.LINE_AA, shift=2)
        pen = np.maximum(pen, m.astype(np.float32) / 255.0)
    x = _pen_render(x, pen, rng)
    on = pen > 0.5
    rim = (pen > 0.05) & ~on
    y[..., 1] = np.maximum(y[..., 1], on.astype(np.float32))
    # the pen is scored as handwriting; print stays print under it and is scored there; pen pixels
    # off the print are scored "not print"; the anti-aliased rim is not scored
    v[..., 1] = np.where(on, 1.0, v[..., 1] * ~rim)
    v[..., 0] = np.where(on, 1.0, v[..., 0] * ~(rim & ~printed))
    return x, y, v


def pen_val_set(pages, n=96, seed=7):
    """fixed tiles of the validation pages with drawn pen circles / ticks (exact labels)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = pages[rng.integers(len(pages))]
        t = sample_real_tile(p, rng)
        out.append(add_pen_marks(t, rng))
    return out


@torch.no_grad()
def pen_eval(model, tiles, thr=0.5):
    """on drawn pen marks: pen erased (recall), print under the pen kept as print (recall of the
    print channel where print and pen overlap), print beside it not taken for pen."""
    model.eval()
    prs = []
    for i in range(0, len(tiles), 16):
        x = torch.from_numpy(np.stack([t[0] for t in tiles[i:i + 16]])).to(DEV).permute(0, 3, 1, 2).float()
        prs.append(torch.sigmoid(model(x)).permute(0, 2, 3, 1).cpu().numpy())
    pr = np.concatenate(prs)
    y = np.stack([t[1] for t in tiles]); v = np.stack([t[2] for t in tiles])
    yp, yh, vp, vh = y[..., 0] > 0.5, y[..., 1] > 0.5, v[..., 0] > 0.5, v[..., 1] > 0.5
    ph, pp = pr[..., 1] > thr, pr[..., 0] > 0.5
    pen_rec = (ph & yh & vh).sum() / max(1, (yh & vh).sum())
    under = yp & yh & vp
    under_rec = (pp & under).sum() / max(1, under.sum())
    only_pen = yh & ~yp & vp
    pen_not_print = (~pp & only_pen).sum() / max(1, only_pen.sum())
    beside = yp & ~yh & vh
    print_kept = 1 - (ph & beside).sum() / max(1, beside.sum())
    return {"pen_recall": float(pen_rec), "under_pen_print": float(under_rec), "pen_not_print": float(pen_not_print), "print_kept": float(print_kept)}


def refresh_hard(model, pages, thr=0.5):
    """store per-page coordinates of pixels the current model gets wrong (hw FN / FP on print)."""
    for p in pages:
        pr = predict_page(model, p["images"][0], DEV)
        ph = pr[..., 1] > thr
        yh, yp = p["y"][..., 1] > 0.5, p["y"][..., 0] > 0.5
        vh, vp = p["valid"][..., 1] > 0.5, p["valid"][..., 0] > 0.5
        err = (~ph & yh & vh) | (ph & yp & ~yh & vp)
        p["hard_pts"] = np.argwhere(err)


def sample_aug_tile(pages, rng, p_paste=0.5, p_shape=0.45, p_hard=0.0, p_pen=0.0):
    if p_hard > 0 and rng.random() < p_hard:
        a = pages[rng.integers(len(pages))]
        if len(a.get("hard_pts", ())):
            hold = a["hw_pts"]
            a["hw_pts"] = a["hard_pts"]
            t = sample_real_tile(a, rng)
            a["hw_pts"] = hold
            return t
    t = _sample_aug_tile(pages, rng, p_paste)
    if rng.random() < p_shape:
        t = add_print_shapes(t, rng)
    if rng.random() < p_pen:
        t = add_pen_marks(t, rng)
    return t


def _sample_aug_tile(pages, rng, p_paste=0.5):
    a = pages[rng.integers(len(pages))]
    if rng.random() < p_paste and len(a["print_pts"]):
        # destination centred on printed text, source centred on handwriting of another page
        b = pages[rng.integers(len(pages))]
        hold = a["hw_pts"]
        a["hw_pts"] = a["print_pts"]
        dst = sample_real_tile(a, rng)
        a["hw_pts"] = hold
        src = sample_real_tile(b, rng)
        dx, dy = rng.integers(-40, 41, 2)
        src = tuple(np.roll(t, (dy, dx), (0, 1)) for t in src)
        return paste_hw(dst, src, rng)
    return sample_real_tile(a, rng)


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


class RealAcc:
    def __init__(self):
        self.c = dict.fromkeys(["h_tp", "h_fp", "h_fn", "p_tp", "p_fp", "p_fn", "po", "po_hw"], 0)

    def add(self, probs, page, thr=0.5):
        ph, pp = probs[..., 1] > thr, probs[..., 0] > 0.5
        yp, yh = page["y"][..., 0] > 0.5, page["y"][..., 1] > 0.5
        vp, vh = page["valid"][..., 0] > 0.5, page["valid"][..., 1] > 0.5
        # stroke-edge tolerance: the inner 1 px edge of each labelled stroke is not scored
        # (the app's erase step dilates the mask; edge pixels do not decide if a stroke is erased)
        vh = vh & ~(yh & ~(cv2.erode(yh.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0))
        c = self.c
        # hw false positives only count on ink: "erasing" blank paper leaves the page unchanged
        c["h_tp"] += int((ph & yh & vh).sum()); c["h_fp"] += int((ph & ~yh & vh & page["ink"]).sum()); c["h_fn"] += int((~ph & yh & vh).sum())
        # print false positives only count on ink too (calling blank paper "print" changes nothing)
        c["p_tp"] += int((pp & yp & vp).sum()); c["p_fp"] += int((pp & ~yp & vp & page["ink"]).sum()); c["p_fn"] += int((~pp & yp & vp).sum())
        po = yp & ~yh & vp
        c["po"] += int(po.sum()); c["po_hw"] += int((po & ph).sum())

    def result(self):
        c, e = self.c, 1e-9
        return {
            "hw_recall": c["h_tp"] / (c["h_tp"] + c["h_fn"] + e),
            "hw_precision": c["h_tp"] / (c["h_tp"] + c["h_fp"] + e),
            "print_kept": 1 - c["po_hw"] / (c["po"] + e),
            "iou_hw": c["h_tp"] / (c["h_tp"] + c["h_fp"] + c["h_fn"] + e),
            "iou_print": c["p_tp"] / (c["p_tp"] + c["p_fp"] + c["p_fn"] + e),
        }


def real_eval(model, pages, thr=0.5, keep=False):
    acc, per, probs = RealAcc(), {}, {}
    for p in pages:
        pr = predict_page(model, p["images"][0], DEV)
        a = RealAcc()
        a.add(pr, p, thr)
        acc.add(pr, p, thr)
        per[p["name"]] = a.result()
        if keep:
            probs[p["name"]] = pr
    return acc.result(), per, probs


def rfmt(r):
    return "  ".join(f"{k}={r[k]:.4f}" for k in CRITERIA)


def overlay(img, pr, thr=0.5):
    vis = (img.astype(np.float32) * 0.35 + 255 * 0.65).astype(np.uint8)
    vis[pr[..., 0] > 0.5] = (30, 60, 230)
    vis[pr[..., 1] > thr] = (230, 30, 30)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(ML_DIR, "ink_segmenter_v6_torch.pt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--real-w", type=float, default=2.0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--paste", type=float, default=0.5, help="share of real tiles with pasted handwriting")
    ap.add_argument("--hw-pos-w", type=float, default=4.0)
    ap.add_argument("--print-pos-w", type=float, default=0.0)
    ap.add_argument("--shape", type=float, default=0.45, help="share of real tiles with synthetic printed outlines")
    ap.add_argument("--hard", type=float, default=0.0, help="share of real tiles centred on current errors")
    ap.add_argument("--pen", type=float, default=0.0, help="share of real tiles with drawn pen circles / ticks over print")
    ap.add_argument("--pen-print-neg", type=float, default=0.0, help="extra weight on pen-only pixels in the print channel")
    ap.add_argument("--ov-print-w", type=float, default=3.0, help="weight of print hidden under pen in the print channel")
    ap.add_argument("--pen-score-w", type=float, default=0.5, help="weight of the pen-mark checks in checkpoint choice")
    ap.add_argument("--recall-w", type=float, default=1.0, help="weight of hw recall in checkpoint choice")
    ap.add_argument("--gate-kept", type=float, default=0.98)
    ap.add_argument("--gate-prec", type=float, default=0.90)
    ap.add_argument("--val", default=",".join(VAL))
    ap.add_argument("--test", default=",".join(TEST))
    ap.add_argument("--out", default="ink_segmenter_v7")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    names = all_pages()
    VAL[:] = [n for n in args.val.split(",") if n]
    TEST[:] = [n for n in args.test.split(",") if n]
    held = set(VAL + TEST)
    train_names = names if args.all else [n for n in names if n not in held]
    train = [train_page(n) for n in train_names]
    val = [load_page(n) for n in VAL]
    test = [load_page(n) for n in TEST]
    print("train pages:", train_names, " val:", VAL, " test:", TEST, flush=True)
    Xtr, Ytr = load("train")
    Xte, Yte = load("test")

    model = M.make_torch_model().to(DEV)
    model.load_state_dict(torch.load(args.init, map_location=DEV))

    r_val, _, _ = real_eval(model, val)
    r_test, per, _ = real_eval(model, test)
    print(f"BEFORE (v6) val : {rfmt(r_val)}", flush=True)
    print(f"BEFORE (v6) test: {rfmt(r_test)}", flush=True)
    for k, v in per.items():
        print(f"    {k}: {rfmt(v)}", flush=True)
    print(f"BEFORE (v6) synthetic test: {fmt(evaluate(model, Xte, Yte))}", flush=True)
    pen_tiles = pen_val_set([train_page(n) for n in VAL]) if args.pen > 0 else None
    if pen_tiles is not None:
        print(f"BEFORE pen marks: {pen_eval(model, pen_tiles)}", flush=True)

    n_real = int(round(args.bs * args.real_frac))
    n_syn = args.bs - n_real
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    model.train()
    freeze_bn(model)
    t0, run = time.time(), 0.0
    best_sd, best_score, best_step = None, -1e9, 0
    for step in range(1, args.steps + 1):
        idx = np.sort(rng.choice(len(Xtr), n_syn, replace=False))
        xs = torch.from_numpy(Xtr[idx]).to(DEV).permute(0, 3, 1, 2).float()
        ys = to_targets(torch.from_numpy(Ytr[idx]).to(DEV))
        if args.hard > 0 and (step == 1 or step % 300 == 0):
            refresh_hard(model, train)
            model.train()
            freeze_bn(model)
        tiles = [sample_aug_tile(train, rng, args.paste, p_shape=args.shape, p_hard=args.hard, p_pen=args.pen) for _ in range(n_real)]
        xr = torch.from_numpy(np.stack([t[0] for t in tiles])).to(DEV).permute(0, 3, 1, 2).float()
        yr = torch.from_numpy(np.stack([t[1] for t in tiles])).to(DEV).permute(0, 3, 1, 2).float()
        vr = torch.from_numpy(np.stack([t[2] for t in tiles])).to(DEV).permute(0, 3, 1, 2).float()
        x = augment(torch.cat([xs, xr]), g)
        logits = model(x)
        l_syn, _ = loss_fn(logits[:n_syn], ys, 2.5, 4.0, ov_print_w=3.0)
        l_real = masked_loss(logits[n_syn:], yr, vr, hw_pos_w=args.hw_pos_w, ov_print_w=args.ov_print_w)
        if args.pen_print_neg > 0:  # pen with no print under it must not light the print channel
            wn = yr[:, 1] * (1 - yr[:, 0]) * vr[:, 0]
            bn = torch.nn.functional.binary_cross_entropy_with_logits(logits[n_syn:, 0], yr[:, 0], reduction="none")
            l_real = l_real + args.pen_print_neg * (bn * wn).sum() / wn.sum().clamp(min=1)
        if args.print_pos_w > 0:  # extra weight on real printed pixels (thin rules, box sides)
            wp = yr[:, 0] * vr[:, 0]
            bp = torch.nn.functional.binary_cross_entropy_with_logits(logits[n_syn:, 0], yr[:, 0], reduction="none")
            l_real = l_real + args.print_pos_w * (bp * wp).sum() / wp.sum().clamp(min=1)
        loss = l_syn + args.real_w * l_real
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += loss.item()
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{args.steps} loss {run / 100:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"elapsed {el / 60:.1f}m eta {el / step * (args.steps - step) / 60:.1f}m", flush=True)
            run = 0.0
        if step % 150 == 0 or step == args.steps:
            r, _, _ = real_eval(model, val)
            gate = r["print_kept"] >= args.gate_kept and r["hw_precision"] >= args.gate_prec
            score = (args.recall_w * r["hw_recall"] + 0.5 * r["iou_hw"] + 0.25 * r["iou_print"]) if gate else -1 + r["print_kept"]
            pe = None
            if pen_tiles is not None:   # pen circles / ticks over print: not in the real labels
                pe = pen_eval(model, pen_tiles)
                score += args.pen_score_w * (pe["pen_recall"] + pe["under_pen_print"] + pe["pen_not_print"])
            tag = f"  pen={pe}" if pe else ""
            if score > best_score:
                best_score, best_step = score, step
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save(best_sd, os.path.join(ML_DIR, args.out + "_best_torch.pt"))
                tag += "  *best"
            print(f"  val@{step}: {rfmt(r)}{tag}", flush=True)
            model.train()
            freeze_bn(model)

    model.load_state_dict(best_sd)
    r_val, _, _ = real_eval(model, val)
    r_test, per, probs = real_eval(model, test, keep=True)
    syn = evaluate(model, Xte, Yte)
    print(f"\nAFTER (best @ step {best_step}) val : {rfmt(r_val)}", flush=True)
    print(f"AFTER test: {rfmt(r_test)}", flush=True)
    for k, v in per.items():
        print(f"    {k}: {rfmt(v)}", flush=True)
    print(f"AFTER synthetic test: {fmt(syn)}", flush=True)
    passes = lambda r: all(r[k] > 0.90 for k in CRITERIA) and r["print_kept"] >= 0.98
    ok = passes(r_test) and all(passes(v) for v in per.values())
    print("ALL CRITERIA > 90% (print_kept >= 98%) ON TEST, total and every page:", ok, flush=True)

    out_pt = os.path.join(ML_DIR, args.out + "_torch.pt")
    torch.save(model.state_dict(), out_pt)
    json.dump({"best_step": best_step, "val": r_val, "test": r_test, "test_pages": per, "synthetic_test": syn,
               "train_pages": train_names, "args": vars(args)},
              open(os.path.join(ML_DIR, args.out + "_metrics.json"), "w"), indent=1)
    v6 = M.make_torch_model().to(DEV)
    v6.load_state_dict(torch.load(args.init, map_location=DEV))
    for p in test:
        p6 = predict_page(v6, p["images"][0], DEV)
        sheet = np.concatenate([overlay(p["images"][0], p6), np.full((p6.shape[0], 8, 3), 128, np.uint8),
                                overlay(p["images"][0], probs[p["name"]])], 1)
        cv2.imwrite(os.path.join(PREVIEW, f"{args.out}_{p['name']}_v6_vs_v7.png"), sheet[..., ::-1])
    print("saved", out_pt, flush=True)


if __name__ == "__main__":
    main()
