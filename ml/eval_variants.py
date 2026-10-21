"""Inference variants on held-out labelled pages: plain, flip-TTA, two-scale, weight soup.
Reports page hw recall / print kept (thr 0.44, like the app) and pen-on-print recall.
Usage: eval_variants.py pages a.pt [b.pt for soup]"""
import sys, numpy as np, cv2, torch, ink_seg_model as M
from infer_util import predict_page
import finetune_v7 as F
pages = sys.argv[1].split(","); A = sys.argv[2]; B = sys.argv[3] if len(sys.argv) > 3 else None
def net(sd):
    n = M.make_torch_model().to('mps'); n.load_state_dict(sd); n.eval(); return n
sa = torch.load(A, map_location='mps')
nets = {"plain": net(sa)}
if B:
    sb = torch.load(B, map_location='mps')
    nets["soup"] = net({k: ((sa[k].float() + sb[k].float()) / 2).to(sa[k].dtype) for k in sa})
def pred(n, img, mode):
    if mode == "tta":
        p = predict_page(n, img, 'mps')
        p += predict_page(n, img[:, ::-1].copy(), 'mps')[:, ::-1]
        p += predict_page(n, img[::-1].copy(), 'mps')[::-1]
        return p / 3
    if mode == "2scale":
        p = predict_page(n, img, 'mps')
        h, w = img.shape[:2]
        up = cv2.resize(img, (int(w * 1.3), int(h * 1.3)), interpolation=cv2.INTER_CUBIC)
        q = cv2.resize(predict_page(n, up, 'mps'), (w, h), interpolation=cv2.INTER_AREA)
        return np.maximum(p, q) * [0, 1] + (p + q) / 2 * [1, 0]
    return predict_page(n, img, 'mps')
data = [F.load_page(n) for n in pages]
for name, n in nets.items():
    for mode in (["plain", "tta", "2scale"] if name == "plain" else ["plain"]):
        a = np.zeros(6)
        for p in data:
            pr = pred(n, p["images"][0], mode)
            yh, yp = p["y"][..., 1] > .5, p["y"][..., 0] > .5; vh, vp = p["valid"][..., 1] > .5, p["valid"][..., 0] > .5
            touch = yh & vh & (cv2.dilate((yp & ~yh).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            pon = yp & ~yh & vp
            h = pr[..., 1] > 0.44
            a += [(h & yh & vh).sum(), (yh & vh).sum(), (~h & pon).sum(), pon.sum(), (h & touch).sum(), touch.sum()]
        print(f"{name:5s} {mode:6s}: hw_recall {a[0]/a[1]:.4f}  print_kept {a[2]/a[3]:.4f}  pen_on_print {a[4]/a[5]:.4f}", flush=True)
