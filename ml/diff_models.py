"""Where two models disagree at equal print loss: per page, pixels model B gets wrong and model A
right (missed handwriting, erased print), as review tiles. Usage: diff_models.py pages A.pt thrA B.pt thrB out.png"""
import sys, numpy as np, cv2, torch, ink_seg_model as M
from infer_util import predict_page
import finetune_v7 as F
pages = sys.argv[1].split(","); A, ta, B, tb, out = sys.argv[2], float(sys.argv[3]), sys.argv[4], float(sys.argv[5]), sys.argv[6]
def net(m):
    n = M.make_torch_model().to('mps'); n.load_state_dict(torch.load(m, map_location='mps')); n.eval(); return n
na, nb = net(A), net(B); tiles = []; tot = {}
for name in pages:
    p = F.load_page(name); img = p["images"][0]
    pa = predict_page(na, img, 'mps')[..., 1] > ta; pb = predict_page(nb, img, 'mps')[..., 1] > tb
    yh, yp = p["y"][..., 1] > .5, p["y"][..., 0] > .5; vh, vp = p["valid"][..., 1] > .5, p["valid"][..., 0] > .5
    miss = yh & vh & pa & ~pb            # B misses handwriting A catches
    lost = yp & ~yh & vp & pb & ~pa      # B erases print A keeps
    gain = yh & vh & pb & ~pa
    tot[name] = (int(miss.sum()), int(lost.sum()), int(gain.sum()))
    for kind, m, col in (("miss", miss, (255, 0, 255)), ("lost", lost, (0, 140, 255))):
        n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)))
        items = sorted([(int(m[lab == i].sum()), i) for i in range(1, n)], reverse=True)[:4]
        for c, i in items:
            x, y, w, h = st[i][:4]; cx, cy = x + w // 2, y + h // 2; R = 40
            y0, y1, x0, x1 = max(0, cy - R), cy + R, max(0, cx - R), cx + R
            a = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2BGR); b = a.copy(); b[m[y0:y1, x0:x1]] = col
            t = np.hstack([a, b]); t = cv2.resize(t, (320, 160), interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, f"{name} {kind} {c}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 0), 1); tiles.append(t)
print("page: (B misses hw, B loses print, B gains hw)", tot)
while len(tiles) % 4: tiles.append(np.full_like(tiles[0], 255))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
