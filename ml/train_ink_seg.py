"""Trains ink_segmenter_v2: 2-channel multi-label (print, handwriting) compact U-Net.

Training runs on the M1 GPU through the PyTorch twin of the model (see ink_seg_model.py — TF has
no GPU here), then the weights are ported 1:1 into the Keras model, verified numerically, and
saved as ml/ink_segmenter_v2.keras (the export source).

Loss (per channel): BCE + soft dice. Handwriting positives weighted up (minority class) and
handwriting-channel negatives on printed-only pixels weighted up (print->handwriting false
positives erase real content — the costly error).

Metrics (threshold 0.5) on the held-out synthetic test set:
  IoU print, IoU handwriting,
  print->hw FP rate  = P(pred_hw | print=1, hw=0)
  overlap print recall = P(pred_print | print=1, hw=1)   (print hidden under handwriting)

Usage: venv/bin/python train_ink_seg.py [--epochs 7 --bs 12 --lr 2e-3]
"""
from __future__ import annotations

import argparse
import math
import os
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import torch
import torch.nn.functional as F

import ink_seg_model as M

ML_DIR = os.path.dirname(os.path.abspath(__file__))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load(name):
    d = np.load(os.path.join(ML_DIR, f"inkseg{os.environ.get('INKSEG_TAG', '')}_{name}.npz"))
    return d["images"], d["masks"]


def to_targets(m):  # uint8 bitmask (B,H,W) -> float (B,2,H,W)
    m = m.long()
    return torch.stack([(m & 1) > 0, (m & 2) > 0], 1).float()


def augment(x, g):
    """x float (B,3,H,W) 0..255 on device. Mild photometric aug (geometry is baked into data)."""
    B = x.shape[0]
    dev = x.device
    # contrast / brightness around paper level
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.15)
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(-18, 12)
    x = (x - 200.0) * c + 200.0 + b
    # per-channel gain (white balance)
    x = x * torch.empty(B, 3, 1, 1, device=dev).uniform_(0.94, 1.06)
    # gamma
    gam = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.25)
    x = 255.0 * (x.clamp(0, 255) / 255.0) ** gam
    # occasional channel permutation (colour must not be the cue) / grayscale
    if torch.rand(1, generator=g).item() < 0.25:
        x = x[:, torch.randperm(3, generator=g)]
    gray = (torch.rand(B, 1, 1, 1, device=dev) < 0.12).float()
    x = gray * x.mean(1, keepdim=True).expand_as(x) + (1 - gray) * x
    return x.clamp(0, 255)


def loss_fn(logits, y, hw_pos_w=2.0, hw_neg_on_print_w=4.0, dice_w=0.5, ov_print_w=1.0):
    lp, lh = logits[:, 0], logits[:, 1]
    yp, yh = y[:, 0], y[:, 1]
    w_p = 1.0 + (ov_print_w - 1.0) * yp * yh  # print hidden under handwriting: must still be found
    bce_p = (F.binary_cross_entropy_with_logits(lp, yp, reduction="none") * w_p).mean()
    w_h = 1.0 + (hw_pos_w - 1.0) * yh + (hw_neg_on_print_w - 1.0) * yp * (1 - yh)
    bce_h = (F.binary_cross_entropy_with_logits(lh, yh, reduction="none") * w_h).mean()
    pr = torch.sigmoid(logits)
    inter = (pr * y).sum((0, 2, 3))
    den = pr.sum((0, 2, 3)) + y.sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    return bce_p + bce_h + dice_w * dice.sum(), (bce_p.item(), bce_h.item(), dice[0].item(), dice[1].item())


class Acc:
    """confusion accumulators for the report metrics."""

    def __init__(self):
        self.c = {k: 0 for k in ["p_tp", "p_fp", "p_fn", "h_tp", "h_fp", "h_fn", "printonly", "printonly_hw", "ov", "ov_p", "bg", "bg_hw"]}

    def add(self, prob, y, thr=0.5):
        pp, ph = prob[:, 0] > thr, prob[:, 1] > thr
        yp, yh = y[:, 0] > 0.5, y[:, 1] > 0.5
        c = self.c
        c["p_tp"] += (pp & yp).sum().item(); c["p_fp"] += (pp & ~yp).sum().item(); c["p_fn"] += (~pp & yp).sum().item()
        c["h_tp"] += (ph & yh).sum().item(); c["h_fp"] += (ph & ~yh).sum().item(); c["h_fn"] += (~ph & yh).sum().item()
        po = yp & ~yh
        c["printonly"] += po.sum().item(); c["printonly_hw"] += (po & ph).sum().item()
        ov = yp & yh
        c["ov"] += ov.sum().item(); c["ov_p"] += (ov & pp).sum().item()
        bg = ~yp & ~yh
        c["bg"] += bg.sum().item(); c["bg_hw"] += (bg & ph).sum().item()

    def result(self):
        c = self.c
        e = 1e-9
        return {
            "iou_print": c["p_tp"] / (c["p_tp"] + c["p_fp"] + c["p_fn"] + e),
            "iou_hw": c["h_tp"] / (c["h_tp"] + c["h_fp"] + c["h_fn"] + e),
            "hw_precision": c["h_tp"] / (c["h_tp"] + c["h_fp"] + e),
            "hw_recall": c["h_tp"] / (c["h_tp"] + c["h_fn"] + e),
            "print_to_hw_fp": c["printonly_hw"] / (c["printonly"] + e),
            "overlap_print_recall": c["ov_p"] / (c["ov"] + e),
            "paper_to_hw_fp": c["bg_hw"] / (c["bg"] + e),
        }


@torch.no_grad()
def evaluate(model, X, Y, bs=24):
    model.eval()
    acc = Acc()
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(DEV).permute(0, 3, 1, 2).float()
        y = to_targets(torch.from_numpy(Y[i:i + bs]).to(DEV))
        acc.add(torch.sigmoid(model(x)), y)
    model.train()
    return acc.result()


def fmt(r):
    return "  ".join(f"{k}={v:.4f}" for k, v in r.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hw-pos-w", type=float, default=2.0)
    ap.add_argument("--hw-neg-print-w", type=float, default=4.0)
    ap.add_argument("--ov-print-w", type=float, default=3.0)
    ap.add_argument("--init", default=None, help="optional torch state_dict to fine-tune from")
    ap.add_argument("--out", default="ink_segmenter_v2")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N train tiles")
    args = ap.parse_args()

    Xtr, Ytr = load("train")
    Xva, Yva = load("val")
    if args.limit:
        Xtr, Ytr, Xva, Yva = Xtr[:args.limit], Ytr[:args.limit], Xva[:args.limit], Yva[:args.limit]
    print(f"train {Xtr.shape}  val {Xva.shape}  device {DEV}", flush=True)

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    model = M.make_torch_model().to(DEV)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=DEV))
    nparams = sum(p.numel() for p in model.parameters())
    print(f"trainable params {nparams}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(Xtr) // args.bs
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.08)

    best, best_score = None, -1
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(len(Xtr))
        run = np.zeros(4)
        for k in range(steps_per_epoch):
            idx = np.sort(perm[k * args.bs:(k + 1) * args.bs])
            x = torch.from_numpy(Xtr[idx]).to(DEV).permute(0, 3, 1, 2).float()
            y = to_targets(torch.from_numpy(Ytr[idx]).to(DEV))
            x = augment(x, g)
            logits = model(x)
            loss, parts = loss_fn(logits, y, args.hw_pos_w, args.hw_neg_print_w, ov_print_w=args.ov_print_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run += np.array(parts)
            if (k + 1) % 200 == 0:
                el = time.time() - t0
                eta = el / step * (total - step)
                print(f"ep {ep + 1} step {k + 1}/{steps_per_epoch}  bce_p {run[0] / 200:.4f} bce_h {run[1] / 200:.4f} "
                      f"dice_p {run[2] / 200:.3f} dice_h {run[3] / 200:.3f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"elapsed {el / 60:.1f}m eta {eta / 60:.1f}m", flush=True)
                run[:] = 0
        r = evaluate(model, Xva, Yva)
        score = r["iou_hw"] + 0.5 * r["overlap_print_recall"] - 5 * r["print_to_hw_fp"]
        print(f"== epoch {ep + 1} val: {fmt(r)}  score {score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    torch.save(best, os.path.join(ML_DIR, f"{args.out}_torch.pt"))

    del Xtr, Ytr
    Xte, Yte = load("test")
    if args.limit:
        Xte, Yte = Xte[:args.limit], Yte[:args.limit]
    r = evaluate(model, Xte, Yte)
    print(f"\nTEST ({len(Xte)} tiles): {fmt(r)}", flush=True)

    # ---- port to Keras, verify, save -----------------------------------------------------------
    model = model.to("cpu").eval()
    kmodel = M.build_model()
    n = M.port_torch_to_keras(model, kmodel)
    xs = Xte[:8].astype(np.float32)
    with torch.no_grad():
        pt = torch.sigmoid(model(torch.from_numpy(xs).permute(0, 3, 1, 2))).permute(0, 2, 3, 1).numpy()
    pk = kmodel.predict(xs, verbose=0)
    print(f"ported {n} layers; keras params {kmodel.count_params()}; torch-vs-keras max abs diff {np.abs(pt - pk).max():.2e}")
    out = os.path.join(ML_DIR, f"{args.out}.keras")
    kmodel.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
