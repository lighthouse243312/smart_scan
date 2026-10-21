"""Real strike-throughs over print (p24, the worksheet the user sent): share of the strike pixels the
model takes as handwriting (thr 0.44, the app's default) and its mean print probability there."""
import sys, numpy as np, cv2, torch, ink_seg_model as M
from infer_util import predict_page
z = np.load('real/batch2/labels/p24.npz'); y = z['y']; img = z['image']
st = np.load(sys.argv[1]).astype(np.uint8)
st = (cv2.resize(st, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_NEAREST) > 0) & ((y & 2) > 0)
for m in sys.argv[2:]:
    net = M.make_torch_model().to('mps'); net.load_state_dict(torch.load(m, map_location='mps')); net.eval()
    pr = predict_page(net, img, 'mps')
    print(f"{m}: strike hw>0.44 {(pr[..., 1][st] > 0.44).mean():.3f}  hw mean {pr[..., 1][st].mean():.2f}  print mean {pr[..., 0][st].mean():.2f}", flush=True)
