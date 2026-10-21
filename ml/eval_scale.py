"""v46 run at a larger working size: recall at matched print loss vs the app's 1536 px."""
import sys, numpy as np, cv2, torch, ink_seg_model as M
from infer_util import predict_page
import finetune_v7 as F
pages = sys.argv[1].split(","); m = sys.argv[2]
net = M.make_torch_model().to('mps'); net.load_state_dict(torch.load(m, map_location='mps')); net.eval()
data = [F.load_page(n) for n in pages]
for sc in [1.0, 1.25, 1.5]:
    H = []; P = []; T = []
    for p in data:
        img = p["images"][0]; h, w = img.shape[:2]
        im = img if sc == 1 else cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_CUBIC)
        pr = predict_page(net, im, 'mps')[..., 1]
        if sc != 1: pr = cv2.resize(pr, (w, h), interpolation=cv2.INTER_AREA)
        yh, yp = p["y"][..., 1] > .5, p["y"][..., 0] > .5; vh, vp = p["valid"][..., 1] > .5, p["valid"][..., 0] > .5
        touch = yh & vh & (cv2.dilate((yp & ~yh).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        H.append(pr[yh & vh]); P.append(pr[yp & ~yh & vp]); T.append(pr[touch])
    H, P, T = map(np.concatenate, (H, P, T))
    for budget in [20, 60, 200]:
        thr = np.sort(P)[::-1][budget]
        print(f"scale {sc}: print budget {budget} px -> thr {thr:.3f} hw_recall {(H > thr).mean():.4f} pen_on_print {(T > thr).mean():.4f}", flush=True)
