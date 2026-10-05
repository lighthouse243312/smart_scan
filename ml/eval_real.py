"""Scores ink-segmenter checkpoints on labelled real pages (same criteria as finetune_v7).
Usage: venv/bin/python eval_real.py ckpt.pt [ckpt2.pt ...] --pages p02,p05,p11,p19,p22"""
import argparse

import torch

import ink_seg_model as M
from finetune_v7 import CRITERIA, DEV, load_page, real_eval, rfmt

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--pages", default="p02,p05,p11,p19,p22")
ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args()
pages = [load_page(n) for n in a.pages.split(",")]
for c in a.ckpts:
    m = M.make_torch_model().to(DEV)
    m.load_state_dict(torch.load(c, map_location=DEV))
    r, per, _ = real_eval(m, pages, thr=a.thr)
    passes = lambda x: all(x[k] > 0.90 for k in CRITERIA) and x["print_kept"] >= 0.98
    print(f"{c} thr={a.thr}\n  total: {rfmt(r)}  {'PASS' if passes(r) else ''}")
    for k, v in per.items():
        print(f"    {k}: {rfmt(v)}  {'PASS' if passes(v) else 'fail'}")
