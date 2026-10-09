"""Scores ERASED pages (the app's final output) against the labelled real pages, per page type.

Paper is the local MEDIAN of the original page (not its brightest value): phone photos have grey
paper, and a brightest-value estimate counted cleanly erased pixels as "pen left" (the old
measurement under-reported erasing on grey photos by up to ~15 points).

  handwriting erased : labelled pen pixels less than 20 grey levels darker than paper after erase
  print kept         : labelled print pixels keeping more than half their darkness

Page types (from measurable properties, see the session notes): A = phone photo with colour
(grey paper, pen colour distinct), B = scan with colour lost (white paper), C = scan with many
drawings. Usage:
  venv/bin/python eval_erase.py <dir> <tag>[,<tag>...]
<dir> holds <page>.png (the label image, BGR) and out/<page>_<tag>_erased.png (InkHarness -e tag).
"""
import glob
import os
import sys

import cv2
import numpy as np

LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real", "batch2", "labels")
TYPE_A = {"p01", "p02", "p03", "p04", "p06", "p07", "p09", "p10", "p11", "p19", "p20", "p21"}
TYPE_C = {"p22"}


def paper(gray):
    s = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    s = cv2.medianBlur(s.astype(np.uint8), 21)
    return cv2.resize(s, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def main():
    d_in, tags = sys.argv[1], sys.argv[2].split(",")
    tot = {t: {k: np.zeros(4) for k in "ABC"} for t in tags}
    print("page type | " + " | ".join(f"{t}: hw    print" for t in tags))
    for f in sorted(glob.glob(os.path.join(LABELS, "*.npz"))):
        n = os.path.basename(f)[:-4]
        src = os.path.join(d_in, n + ".png")
        if not os.path.exists(src):
            continue
        typ = "A" if n in TYPE_A else "C" if n in TYPE_C else "B"
        lab = np.load(f); y, v = lab["y"], lab["valid"]
        g0 = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2GRAY).astype(np.float32)
        p0 = paper(g0); dk0 = np.clip(p0 - g0, 0, 255)
        hw = ((y & 2) > 0) & ((v & 2) > 0) & ~((y & 1) > 0) & (dk0 > 40)
        pr = ((y & 1) > 0) & ((v & 1) > 0) & ~((y & 2) > 0) & (dk0 > 40)
        row = []
        for t in tags:
            ef = os.path.join(d_in, "out", f"{n}_{t}_erased.png")
            if not os.path.exists(ef):
                row.append("  -      -  "); continue
            ge = cv2.cvtColor(cv2.imread(ef), cv2.COLOR_BGR2GRAY).astype(np.float32)
            dk = np.clip(p0 - ge, 0, 255)
            er = int((dk[hw] < 20).sum()); kp = int((dk[pr] > 0.5 * dk0[pr]).sum())
            tot[t][typ] += [er, hw.sum(), kp, pr.sum()]
            row.append(f"{er / max(1, hw.sum()):.3f} {kp / max(1, pr.sum()):.3f}")
        print(f"{n}  {typ}   | " + " | ".join(row))
    for t in tags:
        print(t, "  ".join(f"{k}: hw {x[0] / max(1, x[1]):.4f} print {x[2] / max(1, x[3]):.4f}" for k, x in tot[t].items()))


if __name__ == "__main__":
    main()
