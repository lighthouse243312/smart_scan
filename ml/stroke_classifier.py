"""Leave-one-page-out evaluation / training of the colour-independent stroke classifier."""
import os, sys, json
import numpy as np
import torch
from stroke_dataset import FEATS, LABELS

D = os.path.join(LABELS, "strokes")
PAGES = sorted(f[:-4] for f in os.listdir(D) if f.endswith(".npz"))


def load(p, cols):
    d = np.load(os.path.join(D, p + ".npz"))
    m = (d["vfrac"] > 0.5) & (d["area"] >= 12)
    m[0] = False
    hw, pr = m & (d["hwfrac"] > 0.6), m & (d["hwfrac"] < 0.05)
    keep = hw | pr
    return d["X"][keep][:, cols], hw[keep].astype(np.float32), d["area"][keep].astype(np.float32)


def fit(X, y, w, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    Xt, yt, wt = torch.tensor(X), torch.tensor(y), torch.tensor(np.sqrt(w))
    pos = (wt * yt).sum(); neg = (wt * (1 - yt)).sum()
    wt = torch.where(yt > 0, wt / pos, wt / neg)          # balance hw vs print by area
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(600):
        l = (torch.nn.functional.binary_cross_entropy_with_logits(net(Xt)[:, 0], yt, reduction="none") * wt).sum()
        opt.zero_grad(); l.backward(); opt.step()
    return net


def auc(s, y, w):
    o = np.argsort(s, kind="mergesort"); y, w = y[o], w[o]
    cn = np.cumsum(w * (1 - y))
    return float((w * y * cn).sum() / ((w * y).sum() * (w * (1 - y)).sum() + 1e-9))


if __name__ == "__main__":
    sets = {"A features": [FEATS.index(k) for k in FEATS if k != "v20"], "B model v20": [FEATS.index("v20")], "C both": list(range(len(FEATS)))}
    res = {k: {} for k in sets}
    for p in PAGES:
        for name, cols in sets.items():
            tr = [load(q, cols) for q in PAGES if q != p]
            X = np.concatenate([t[0] for t in tr]); y = np.concatenate([t[1] for t in tr]); w = np.concatenate([t[2] for t in tr])
            Xs, ys, ws = load(p, cols)
            if ys.sum() == 0 or (1 - ys).sum() == 0:
                continue
            net = fit(X, y, w)
            with torch.no_grad():
                s = torch.sigmoid(net(torch.tensor(Xs))[:, 0]).numpy()
            res[name][p] = auc(s, ys, ws)
    print("page   " + "  ".join(f"{k:>12s}" for k in sets))
    for p in PAGES:
        print(p, "  ".join(f"{res[k].get(p, float('nan')):12.3f}" for k in sets))
    print("mean ", "  ".join(f"{np.mean(list(res[k].values())):12.3f}" for k in sets))
