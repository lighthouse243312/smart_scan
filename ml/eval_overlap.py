"""Pen over print, measured where the truth is known: real pen strokes (from the labelled pages'
handwriting) pasted over the print of the TEST pages, and drawn pen marks (circles, strike-throughs,
ticks) over the same print. Print under the pen is labelled exactly.

  pen_recall       pen pixels taken as handwriting
  under_pen_print  print hidden under the pen still seen as print (to be redrawn after the erase)
  pen_not_print    pen pixels off the print not taken for print
  print_kept       print right beside the pen not taken for handwriting

Usage: venv/bin/python eval_overlap.py model1.pt [model2.pt ...] [--pages p02,p11,...]
"""
import sys, numpy as np, torch
import ink_seg_model as M
import finetune_v7 as F

args = [a for a in sys.argv[1:] if not a.startswith("--")]
pages = "p02,p11,p19,p26,p29"
if "--pages" in sys.argv: pages = sys.argv[sys.argv.index("--pages") + 1]; args = [a for a in args if a != pages]
F.PASTE_UNDER = True
dst_pages = [F.train_page(n) for n in pages.split(",")]
src_pages = [F.train_page(n) for n in F.all_pages()]
rng = np.random.default_rng(11)
pasted, drawn = [], []
while len(pasted) < 160:
    a = dst_pages[rng.integers(len(dst_pages))]
    if not len(a["print_pts"]): continue
    hold = a["hw_pts"]; a["hw_pts"] = a["print_pts"]; dst = F.sample_real_tile(a, rng); a["hw_pts"] = hold
    b = src_pages[rng.integers(len(src_pages))]
    if b["name"] == a["name"]: continue
    src = F.sample_real_tile(b, rng)
    dx, dy = rng.integers(-40, 41, 2)
    src = tuple(np.roll(t, (dy, dx), (0, 1)) for t in src)
    t = F.paste_hw(dst, src, rng)
    if ((t[1][..., 0] > .5) & (t[1][..., 1] > .5)).sum() < 40: continue   # needs real overlap
    pasted.append(t)
drawn = F.pen_val_set(dst_pages, n=160, seed=12)
for m in args:
    model = M.make_torch_model().to(F.DEV); model.load_state_dict(torch.load(m, map_location=F.DEV))
    a = F.pen_eval(model, pasted); b = F.pen_eval(model, drawn)
    f = lambda d: " ".join(f"{k}={v:.3f}" for k, v in d.items())
    print(f"{m}\n   real pen pasted: {f(a)}\n   drawn pen marks: {f(b)}", flush=True)
