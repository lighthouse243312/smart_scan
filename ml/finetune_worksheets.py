"""Fine-tunes the ink segmenter on REAL worksheet photos (ml/make_pseudo_labels.py labels), mixed
50/50 with synthetic tiles so black-ink and overlap cases are not forgotten.

Each page is used in two versions — the flattened page the labels were made on and the
straightened capture (shadows, colour cast) — so the model learns the label from either. Pages
whose pen isn't coloured have their handwriting channel ignored (hwvalid) rather than taught as
"no handwriting".

Split: --eval pages are held out (pick different documents from the training ones).

usage: venv/bin/python finetune_worksheets.py <labels dir> <pages dir> [--eval p09,p14,p03]
       [--init ink_segmenter_v2_torch.pt --steps 3000 --out ink_segmenter_v4] [--resume]

A checkpoint (<out>_ckpt.pt: model, optimiser, scheduler, RNG, best-so-far) is written at every
evaluation, so an interrupted run continues from there with --resume.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import cv2
import numpy as np
import torch

import ink_seg_model as M
from finetune_real_pseudo import DEV, LONG, _resize_long, masked_loss, predict_page, sample_real_tile
from train_ink_seg import augment, evaluate, fmt, load, loss_fn, to_targets

ML_DIR = os.path.dirname(os.path.abspath(__file__))


def load_page(labels, pages, name):
    rd = lambda s: cv2.imread(os.path.join(labels, f"{name}_{s}.png"), cv2.IMREAD_GRAYSCALE)
    flat = cv2.imread(os.path.join(labels, f"{name}_img.png"))[..., ::-1]
    raw = cv2.imread(os.path.join(pages, f"{name}_original.png"))[..., ::-1]
    imgs = [np.ascontiguousarray(_resize_long(im, cv2.INTER_AREA)) for im in (flat, raw)]
    size = imgs[0].shape[1::-1]
    m = lambda s: cv2.resize(rd(s), size, interpolation=cv2.INTER_NEAREST) > 127
    hw, pr, valid, hwvalid = m("hw"), m("print"), m("valid"), m("hwvalid")
    h, w = hw.shape
    b = max(12, int(0.02 * max(h, w)))                      # off-paper desk at the frame
    frame = np.zeros_like(valid); frame[b:h - b, b:w - b] = True
    valid &= frame; hwvalid &= frame
    band = cv2.dilate(hw.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~hw
    y = np.stack([pr, hw], -1).astype(np.float32)
    # print channel: unknown under and around the pen — a pen stroke may cross print, and the
    # colour labels can't see it (teaching "no print" there makes the model erase it)
    pen_zone = cv2.dilate(hw.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    v = np.stack([valid & ~pen_zone, hwvalid & ~band], -1).astype(np.float32)
    return {"name": name, "images": imgs, "y": y, "valid": v, "hw_pts": np.argwhere(hw & valid)}


def page_metrics(model, page):
    """on the RAW capture (what the app sees): hw IoU / recall / precision, print→hw false positives"""
    pr = predict_page(model, page["images"][1])
    vh = page["valid"][..., 1] > 0.5
    v = page["valid"][..., 0] > 0.5
    yp, yh = page["y"][..., 0] > 0.5, page["y"][..., 1] > 0.5
    ph = pr[..., 1] > 0.5
    tp, fp, fn = (ph & yh & vh).sum(), (ph & ~yh & vh).sum(), (~ph & yh & vh).sum()
    po = yp & ~yh & v
    return {"iou": tp / max(1, tp + fp + fn), "rec": tp / max(1, tp + fn), "prec": tp / max(1, tp + fp),
            "print_fp": (po & ph).sum() / max(1, po.sum())}, pr


def mean_metrics(ms):
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def show(tag, ms):
    return f"{tag}: " + "  ".join(f"{k}={v:.4f}" for k, v in ms.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("pages")
    ap.add_argument("--eval", default="p09,p14,p03")
    ap.add_argument("--init", default=os.path.join(ML_DIR, "ink_segmenter_v2_torch.pt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--real-frac", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--real-w", type=float, default=2.0)
    ap.add_argument("--out", default="ink_segmenter_v4")
    ap.add_argument("--resume", action="store_true", help="continue from <out>_ckpt.pt")
    args = ap.parse_args()

    names = sorted(os.path.basename(p)[:-len("_hw.png")] for p in glob.glob(os.path.join(args.labels, "*_hw.png")))
    eval_names = [n for n in args.eval.split(",") if n in names]
    train = [load_page(args.labels, args.pages, n) for n in names if n not in eval_names]
    held = [load_page(args.labels, args.pages, n) for n in eval_names]
    print(f"train {[p['name'] for p in train]}\neval  {[p['name'] for p in held]}", flush=True)

    rng = np.random.default_rng(0); torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    Xtr, Ytr = load("train"); Xte, Yte = load("test"); Xva, Yva = load("val")
    model = M.make_torch_model().to(DEV)
    model.load_state_dict(torch.load(args.init, map_location=DEV))

    before = mean_metrics([page_metrics(model, p)[0] for p in held])
    before_syn = evaluate(model, Xte, Yte)
    print(show("BEFORE held-out real", before), flush=True)
    print(f"BEFORE synthetic test: {fmt(before_syn)}", flush=True)

    n_real = int(round(args.bs * args.real_frac)); n_syn = args.bs - n_real
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    t0, run, best, best_score, first = time.time(), 0.0, None, -1e9, 1
    ckpt_path = os.path.join(ML_DIR, f"{args.out}_ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["rng"]; g.set_state(ck["g"]); torch.set_rng_state(ck["torch"])
        best, best_score, first = ck["best"], ck["best_score"], ck["step"] + 1
        print(f"resumed at step {first}", flush=True)
    for step in range(first, args.steps + 1):
        idx = np.sort(rng.choice(len(Xtr), n_syn, replace=False))
        xs = torch.from_numpy(Xtr[idx]).to(DEV).permute(0, 3, 1, 2).float()
        ys = to_targets(torch.from_numpy(Ytr[idx]).to(DEV))
        tiles = [sample_real_tile(train[rng.integers(len(train))], rng) for _ in range(n_real)]
        xr = torch.from_numpy(np.stack([t[0] for t in tiles])).to(DEV).permute(0, 3, 1, 2).float()
        yr = torch.from_numpy(np.stack([t[1] for t in tiles])).to(DEV).permute(0, 3, 1, 2).float()
        vr = torch.from_numpy(np.stack([t[2] for t in tiles])).to(DEV).permute(0, 3, 1, 2).float()
        logits = model(augment(torch.cat([xs, xr]), g))
        l_syn, _ = loss_fn(logits[:n_syn], ys, 2.5, 4.0, ov_print_w=3.0)
        loss = l_syn + args.real_w * masked_loss(logits[n_syn:], yr, vr)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run += loss.item()
        if step % 250 == 0:
            el = time.time() - t0
            print(f"step {step}/{args.steps} loss {run / 250:.4f}  eta {el / step * (args.steps - step) / 60:.1f}m", flush=True)
            run = 0.0
        if step % 500 == 0 or step == args.steps:
            # select on synthetic val + training-page fit (held-out pages stay untouched)
            r = evaluate(model, Xva, Yva)
            tr = mean_metrics([page_metrics(model, p)[0] for p in train[::3]])
            score = r["iou_hw"] + r["overlap_print_recall"] - 5 * r["print_to_hw_fp"] + tr["iou"] - 2 * tr["print_fp"]
            print(f"  val {fmt(r)} | {show('train-real', tr)}  score {score:.4f}", flush=True)
            if step >= args.steps // 3 and score > best_score:
                best_score = score
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "g": g.get_state(), "torch": torch.get_rng_state(),
                        "best": best, "best_score": best_score, "step": step}, ckpt_path)
    model.load_state_dict(best)
    torch.save(best, os.path.join(ML_DIR, f"{args.out}_torch.pt"))

    after = []
    os.makedirs(os.path.join(ML_DIR, "preview"), exist_ok=True)
    v2 = M.make_torch_model().to(DEV); v2.load_state_dict(torch.load(args.init, map_location=DEV))
    for p in held:
        m, pr = page_metrics(model, p); after.append(m)
        _, p0 = page_metrics(v2, p)
        img = p["images"][1]
        heat = lambda q: np.repeat((255 - np.clip(q[..., 1], 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
        panel = np.concatenate([img, heat(p0), heat(pr)], 1)
        cv2.imwrite(os.path.join(ML_DIR, "preview", f"{args.out}_{p['name']}.jpg"), panel[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    after = mean_metrics(after)
    after_syn = evaluate(model, Xte, Yte)
    print("\n" + show("BEFORE held-out real", before))
    print(show("AFTER  held-out real", after))
    print(f"BEFORE synthetic test: {fmt(before_syn)}")
    print(f"AFTER  synthetic test: {fmt(after_syn)}", flush=True)

    model = model.to("cpu").eval()
    km = M.build_model()
    M.port_torch_to_keras(model, km)
    km.save(os.path.join(ML_DIR, f"{args.out}.keras"))
    print(f"saved {args.out}.keras / {args.out}_torch.pt")


if __name__ == "__main__":
    main()
