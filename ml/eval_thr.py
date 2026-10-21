import sys, numpy as np, cv2, torch, ink_seg_model as M
from infer_util import predict_page
import finetune_v7 as F
pages = sys.argv[1].split(","); data = [F.load_page(n) for n in pages]
for m in sys.argv[2:]:
    net = M.make_torch_model().to('mps'); net.load_state_dict(torch.load(m, map_location='mps')); net.eval()
    prs = [predict_page(net, p["images"][0], 'mps') for p in data]
    for thr in [float(t) for t in __import__("os").environ.get("THRS","0.44,0.5,0.55,0.6,0.65,0.7").split(",")]:
        a = np.zeros(6)
        for p, pr in zip(data, prs):
            yh, yp = p["y"][..., 1] > .5, p["y"][..., 0] > .5; vh, vp = p["valid"][..., 1] > .5, p["valid"][..., 0] > .5
            touch = yh & vh & (cv2.dilate((yp & ~yh).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            pon = yp & ~yh & vp; h = pr[..., 1] > thr
            a += [(h & yh & vh).sum(), (yh & vh).sum(), (h & pon).sum(), pon.sum(), (h & touch).sum(), touch.sum()]
        print(f"{m.split('_')[2]} thr {thr:.2f}: hw_recall {a[0]/a[1]:.4f}  print_lost_px {int(a[2])} ({a[2]/a[3]*100:.3f}%)  pen_on_print {a[4]/a[5]:.4f}", flush=True)
