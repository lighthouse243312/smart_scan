"""Fine-tunes ink segmenter v2 -> v3 on REAL photos with pseudo-labels, mixed 50/50 with synthetic
tiles so black-ink / overlap cases are not forgotten.

Pseudo-labels come from the app's colour+layout method (ios/Runner/InkAnalysis.hpp), stored in
ml/real/pseudo/<name>_{hw,print,overlap}.png (255 = set):
  hw target     = <name>_hw
  print target  = (<name>_print AND actual ink) OR <name>_overlap
                  (the print layer is dilated / covers pictures, so it is gated by "ink" =
                  grayscale darker than the local paper level by a margin)
  ignored       = outside the paper (largest bright region of a heavily blurred grayscale,
                  eroded), a thin frame at the image border, and a 1-2 px uncertainty band
                  around handwriting strokes (hw channel only).
Each real page is used in two versions: the raw capture and the app's `_processed` version
(sharpened / shadow-removed), so the model works on either.

Split: train on scan_1 + IMG_0101_reading, evaluate on scan_2. NOTE: scan_2 is the SAME printed CV
with the SAME handwriting as scan_1 (a second capture), so the real-photo score measures capture /
lighting generalisation, not new writers or documents.

Outputs: ml/ink_segmenter_v3.keras, ml/ink_segmenter_v3_torch.pt,
         ml/preview/inkseg_v3_scan_2.png (input | v2 hw prob | v3 hw prob | pseudo-label)

Usage: venv/bin/python finetune_real_pseudo.py [--steps 2400 --bs 12 --lr 3e-4]
"""
from __future__ import annotations

import argparse
import math
import os
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import cv2
import numpy as np
import torch

import ink_seg_model as M
from train_ink_seg import Acc, augment, evaluate, fmt, load, loss_fn, to_targets  # noqa: F401

ML_DIR = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(ML_DIR, "real")
PSEUDO = os.path.join(REAL, "pseudo")
PREVIEW = os.path.join(ML_DIR, "preview")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
TILE, STRIDE, LONG = 256, 224, 1536
TRAIN_PAGES = ["scan_1", "IMG_0101_reading"]
EVAL_PAGE = "scan_2"


# ------------------------------------------------------------------------------------------------
# real page loading + pseudo-label targets
# ------------------------------------------------------------------------------------------------
def _resize_long(a, interp):
    s = LONG / max(a.shape[:2])
    return cv2.resize(a, (round(a.shape[1] * s), round(a.shape[0] * s)), interpolation=interp)


def paper_mask(gray, rgb):
    """paper region: bright + low-saturation pixels, closed over print, largest component,
    convex hull (a page is roughly a quad), eroded so the unreliable page edge is ignored."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    g = cv2.GaussianBlur(gray, (0, 0), 3)
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((g > max(thr, 140)) & (hsv[..., 1] < 70)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.ones_like(gray, bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = np.argwhere(lab == k)[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(pts)
    filled = np.zeros_like(m)
    cv2.fillConvexPoly(filled, hull, 1)
    filled = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    return filled > 0


def ink_mask(gray, margin=28):
    """pixels noticeably darker than the local paper level."""
    bg = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    bg = cv2.medianBlur(bg, 21)
    return (bg.astype(np.int16) - gray.astype(np.int16)) > margin


def load_real(name):
    """-> dict(images=[raw, processed] RGB uint8, y=(H,W,2) float, valid=(H,W,2) float) at 1536."""
    raw = cv2.imread(os.path.join(REAL, name + ".png"), cv2.IMREAD_COLOR)[..., ::-1]
    proc_p = os.path.join(PSEUDO, name + "_processed.png")
    imgs = [raw] + ([cv2.imread(proc_p, cv2.IMREAD_COLOR)[..., ::-1]] if os.path.exists(proc_p) else [])
    imgs = [np.ascontiguousarray(_resize_long(im, cv2.INTER_AREA)) for im in imgs]
    rd = lambda s: _resize_long(cv2.imread(os.path.join(PSEUDO, f"{name}_{s}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0,
                                cv2.INTER_AREA)
    hw, pr, ov = rd("hw") > 0.35, rd("print") > 0.5, rd("overlap") > 0.35
    gray = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2GRAY)
    ink = ink_mask(gray)
    y_print = (pr & ink & ~hw) | ov
    y = np.stack([y_print, hw], -1).astype(np.float32)
    valid = paper_mask(gray, imgs[0])
    h, w = gray.shape
    b = 24
    frame = np.zeros_like(valid)
    frame[b:h - b, b:w - b] = True
    valid &= frame
    band = cv2.dilate(hw.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~hw
    v = np.stack([valid, valid & ~band], -1).astype(np.float32)
    return {"name": name, "images": imgs, "y": y, "valid": v, "hw_pts": np.argwhere(hw & valid)}


def sample_real_tile(page, rng):
    """random affine crop: scale 0.8-1.25, rotation +-3 deg, 70% centred near handwriting."""
    img = page["images"][rng.integers(len(page["images"]))]
    h, w = img.shape[:2]
    if rng.random() < 0.7 and len(page["hw_pts"]):
        cy, cx = page["hw_pts"][rng.integers(len(page["hw_pts"]))]
        cy += rng.uniform(-90, 90)
        cx += rng.uniform(-90, 90)
    else:
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    s = rng.uniform(0.8, 1.25)          # output px per source px
    a = math.radians(rng.uniform(-3, 3))
    # map output (u,v) -> source: src = C + R(a) * ((u,v) - 128) / s
    ca, sa = math.cos(a) / s, math.sin(a) / s
    Minv = np.float32([[ca, -sa, cx - ca * 128 + sa * 128], [sa, ca, cy - sa * 128 - ca * 128]])
    flags = cv2.WARP_INVERSE_MAP
    x = cv2.warpAffine(img, Minv, (TILE, TILE), flags=cv2.INTER_LINEAR | flags, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    yv = np.concatenate([page["y"], page["valid"]], -1)
    yv = cv2.warpAffine(yv, Minv, (TILE, TILE), flags=cv2.INTER_NEAREST | flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # photometric: blur / JPEG (contrast, WB, gamma are applied later in train_ink_seg.augment)
    if rng.random() < 0.35:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 95))])
        x = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    if rng.random() < 0.3:
        x = np.clip(x.astype(np.float32) + rng.normal(0, rng.uniform(1, 4), x.shape), 0, 255).astype(np.uint8)
    return x, yv[..., :2], yv[..., 2:]


def masked_loss(logits, y, valid, hw_pos_w=2.5, hw_neg_on_print_w=4.0, ov_print_w=3.0, dice_w=0.5):
    """train_ink_seg.loss_fn with a per-channel validity mask (N,2,H,W)."""
    import torch.nn.functional as F
    lp, lh = logits[:, 0], logits[:, 1]
    yp, yh = y[:, 0], y[:, 1]
    vp, vh = valid[:, 0], valid[:, 1]
    w_p = (1.0 + (ov_print_w - 1.0) * yp * yh) * vp
    bce_p = (F.binary_cross_entropy_with_logits(lp, yp, reduction="none") * w_p).sum() / vp.sum().clamp(min=1)
    w_h = (1.0 + (hw_pos_w - 1.0) * yh + (hw_neg_on_print_w - 1.0) * yp * (1 - yh)) * vh
    bce_h = (F.binary_cross_entropy_with_logits(lh, yh, reduction="none") * w_h).sum() / vh.sum().clamp(min=1)
    pr = torch.sigmoid(logits) * valid
    yy = y * valid
    inter = (pr * yy).sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (pr.sum((0, 2, 3)) + yy.sum((0, 2, 3)) + 1)
    return bce_p + bce_h + dice_w * dice.sum()


# ------------------------------------------------------------------------------------------------
# tiled page inference + real-page metrics
# ------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_page(model, img, bs=8):
    model.eval()
    h, w = img.shape[:2]
    nx = max(1, math.ceil((w - TILE) / STRIDE) + 1)
    ny = max(1, math.ceil((h - TILE) / STRIDE) + 1)
    H, W = (ny - 1) * STRIDE + TILE, (nx - 1) * STRIDE + TILE
    pad = np.full((H, W, 3), 255, np.uint8)
    pad[:h, :w] = img
    ramp = np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE)).astype(np.float32)
    win = np.clip(np.minimum.outer(ramp, ramp) / 32.0, 0.05, 1.0)[..., None]
    acc = np.zeros((H, W, 2), np.float32)
    ws = np.zeros((H, W, 1), np.float32)
    coords = [(j * STRIDE, i * STRIDE) for j in range(ny) for i in range(nx)]
    for k in range(0, len(coords), bs):
        cs = coords[k:k + bs]
        x = torch.from_numpy(np.stack([pad[y:y + TILE, x:x + TILE] for y, x in cs])).to(DEV).permute(0, 3, 1, 2).float()
        p = torch.sigmoid(model(x)).permute(0, 2, 3, 1).cpu().numpy()
        for (y, x_), pp in zip(cs, p):
            acc[y:y + TILE, x_:x_ + TILE] += pp * win
            ws[y:y + TILE, x_:x_ + TILE] += win
    model.train()
    return (acc / np.maximum(ws, 1e-6))[:h, :w]


def real_metrics(model, page):
    out = {}
    for k, img in enumerate(page["images"]):
        pr = predict_page(model, img)
        v = page["valid"][..., 0] > 0.5
        vh = page["valid"][..., 1] > 0.5
        yp, yh = page["y"][..., 0] > 0.5, page["y"][..., 1] > 0.5
        ph = pr[..., 1] > 0.5
        tp = (ph & yh & vh).sum()
        fp = (ph & ~yh & vh).sum()
        fn = (~ph & yh & vh).sum()
        po = yp & ~yh & v
        tag = "raw" if k == 0 else "processed"
        out[tag] = {"hw_iou": tp / max(1, tp + fp + fn), "hw_recall": tp / max(1, tp + fn), "hw_precision": tp / max(1, tp + fp),
                    "print_to_hw_fp": (po & ph).sum() / max(1, po.sum())}
        out[tag + "_probs"] = pr
    return out


def fmt_real(r):
    return " | ".join(f"{t}: " + "  ".join(f"{k}={v:.4f}" for k, v in r[t].items()) for t in ("raw", "processed") if t in r)


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(ML_DIR, "ink_segmenter_v2_torch.pt"))
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--real-w", type=float, default=2.0, help="loss weight of the real half of each batch")
    ap.add_argument("--out", default="ink_segmenter_v3")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    train_pages = [load_real(n) for n in TRAIN_PAGES]
    eval_page = load_real(EVAL_PAGE)
    for p in train_pages + [eval_page]:
        print(f"{p['name']}: {p['y'].shape[:2]} versions={len(p['images'])}  hw px {p['y'][..., 1].mean():.3%}  "
              f"print px {p['y'][..., 0].mean():.3%}  valid {p['valid'][..., 0].mean():.1%}", flush=True)
    Xtr, Ytr = load("train")
    Xte, Yte = load("test")

    model = M.make_torch_model().to(DEV)
    model.load_state_dict(torch.load(args.init, map_location=DEV))
    v2 = M.make_torch_model().to(DEV)
    v2.load_state_dict(torch.load(args.init, map_location=DEV))
    v2.eval()

    before_real = real_metrics(model, eval_page)
    before_syn = evaluate(model, Xte, Yte)
    print(f"BEFORE  real {EVAL_PAGE}: {fmt_real(before_real)}", flush=True)
    print(f"BEFORE  synthetic test: {fmt(before_syn)}", flush=True)

    n_real = int(round(args.bs * args.real_frac))
    n_syn = args.bs - n_real
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    t0 = time.time()
    run = 0.0
    best, best_score = None, -1e9
    for step in range(1, args.steps + 1):
        # synthetic half
        idx = np.sort(rng.choice(len(Xtr), n_syn, replace=False))
        xs = torch.from_numpy(Xtr[idx]).to(DEV).permute(0, 3, 1, 2).float()
        ys = to_targets(torch.from_numpy(Ytr[idx]).to(DEV))
        # real half
        tiles = [sample_real_tile(train_pages[rng.integers(len(train_pages))], rng) for _ in range(n_real)]
        xr = torch.from_numpy(np.stack([t[0] for t in tiles])).to(DEV).permute(0, 3, 1, 2).float()
        yr = torch.from_numpy(np.stack([t[1] for t in tiles])).to(DEV).permute(0, 3, 1, 2).float()
        vr = torch.from_numpy(np.stack([t[2] for t in tiles])).to(DEV).permute(0, 3, 1, 2).float()
        x = augment(torch.cat([xs, xr]), g)
        logits = model(x)
        l_syn, _ = loss_fn(logits[:n_syn], ys, 2.5, 4.0, ov_print_w=3.0)
        l_real = masked_loss(logits[n_syn:], yr, vr)
        loss = l_syn + args.real_w * l_real
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += loss.item()
        if step % 200 == 0:
            el = time.time() - t0
            print(f"step {step}/{args.steps} loss {run / 200:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"elapsed {el / 60:.1f}m eta {el / step * (args.steps - step) / 60:.1f}m", flush=True)
            run = 0.0
        if step % 600 == 0 or step == args.steps:
            # model selection on SYNTHETIC val only (the real eval page stays untouched)
            Xva, Yva = load("val")
            r = evaluate(model, Xva, Yva)
            score = r["iou_hw"] + 0.5 * r["overlap_print_recall"] - 5 * r["print_to_hw_fp"]
            print(f"  step {step} synthetic val: {fmt(r)}  score {score:.4f}", flush=True)
            if step >= args.steps // 2 and score > best_score:
                best_score = score
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    torch.save(best, os.path.join(ML_DIR, f"{args.out}_torch.pt"))

    after_real = real_metrics(model, eval_page)
    after_syn = evaluate(model, Xte, Yte)
    print(f"\nBEFORE  real {EVAL_PAGE}: {fmt_real(before_real)}")
    print(f"AFTER   real {EVAL_PAGE}: {fmt_real(after_real)}")
    print(f"BEFORE  synthetic test: {fmt(before_syn)}")
    print(f"AFTER   synthetic test: {fmt(after_syn)}", flush=True)

    # preview: input | v2 hw prob | v3 hw prob | pseudo-label (processed version if present)
    os.makedirs(PREVIEW, exist_ok=True)
    tag = "processed" if "processed_probs" in after_real else "raw"
    img = eval_page["images"][-1]
    heat = lambda p: np.repeat((255 - np.clip(p, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
    lab = np.full(img.shape, 255, np.uint8)
    lab[eval_page["y"][..., 0] > 0.5] = (70, 110, 255)
    lab[eval_page["y"][..., 1] > 0.5] = (235, 40, 40)
    lab[eval_page["valid"][..., 0] < 0.5] = (200, 200, 200)
    panels = [img, heat(before_real[tag + "_probs"][..., 1]), heat(after_real[tag + "_probs"][..., 1]), lab]
    names = ["input", "v2 handwriting prob", "v3 handwriting prob", "pseudo-label (grey = ignored)"]
    out = []
    for p, n in zip(panels, names):
        p = p.copy()
        cv2.putText(p, n, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(p, n, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2, cv2.LINE_AA)
        out.append(p)
    panel = np.concatenate(out, 1)
    s = 3200 / panel.shape[1]
    panel = cv2.resize(panel, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(PREVIEW, f"inkseg_v3_{EVAL_PAGE}.png"), panel[..., ::-1])
    # zoomed crop of the handwriting-dense lower half
    h = img.shape[0]
    crop = np.concatenate([p[int(h * 0.62):int(h * 0.82)] for p in out], 0)
    cv2.imwrite(os.path.join(PREVIEW, f"inkseg_v3_{EVAL_PAGE}_crop.png"), crop[..., ::-1])

    # port to Keras + save
    model = model.to("cpu").eval()
    km = M.build_model()
    M.port_torch_to_keras(model, km)
    xs = Xte[:4].astype(np.float32)
    with torch.no_grad():
        pt = torch.sigmoid(model(torch.from_numpy(xs).permute(0, 3, 1, 2))).permute(0, 2, 3, 1).numpy()
    print(f"torch-vs-keras max abs diff {np.abs(pt - km.predict(xs, verbose=0)).max():.2e}")
    km.save(os.path.join(ML_DIR, f"{args.out}.keras"))
    print(f"saved {os.path.join(ML_DIR, args.out + '.keras')}")


if __name__ == "__main__":
    main()
