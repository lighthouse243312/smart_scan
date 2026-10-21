"""Pen over / against print on held-out labelled pages, where the labels know it:
  pen_on_print   pen pixels touching printed text, taken as handwriting (thr 0.44)
  print_by_pen   printed pixels within 3 px of the pen, NOT taken as handwriting
  page hw / print_kept as finetune_v7 reports them.
Usage: eval_touch.py pages model.pt [model.pt ...]"""
import sys, numpy as np, cv2, torch, ink_seg_model as M
from infer_util import predict_page
import finetune_v7 as F
pages = sys.argv[1].split(",")
for m in sys.argv[2:]:
    net = M.make_torch_model().to('mps'); net.load_state_dict(torch.load(m, map_location='mps')); net.eval()
    a = np.zeros(6)
    for n in pages:
        p = F.load_page(n); pr = predict_page(net, p["images"][0], 'mps')
        yh, yp = p["y"][..., 1] > .5, p["y"][..., 0] > .5; vh, vp = p["valid"][..., 1] > .5, p["valid"][..., 0] > .5
        txt = (yp & ~yh).astype(np.uint8)
        touch = yh & vh & (cv2.dilate(txt, np.ones((3, 3), np.uint8)) > 0)
        near = yp & ~yh & vp & (cv2.dilate(yh.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        h = pr[..., 1] > 0.44
        a += [(h & touch).sum(), touch.sum(), (~h & near).sum(), near.sum(), (h & yh & vh).sum(), (yh & vh).sum()]
    print(f"{m}: pen_on_print {a[0]/max(a[1],1):.3f} ({int(a[1])} px)  print_by_pen {a[2]/max(a[3],1):.3f} ({int(a[3])} px)  hw_recall {a[4]/max(a[5],1):.3f}", flush=True)
