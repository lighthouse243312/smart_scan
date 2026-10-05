"""Draws a labelled 50 px coordinate grid (ORIGINAL pixels) over a batch2 page label preview,
for writing overrides.json rectangles. Usage: venv/bin/python grid_view.py p05 [x0 y0 x1 y1]"""
import os, sys
import cv2
import numpy as np
ML = os.path.dirname(os.path.abspath(__file__))
p = sys.argv[1]
src = cv2.imread(os.path.join(ML, "real", "batch2", p + ".jpg"))
lab = cv2.imread(os.path.join(ML, "preview", f"label_{p}.png"))
lab = cv2.resize(lab, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
x0, y0, x1, y1 = [int(v) for v in sys.argv[2:6]] if len(sys.argv) > 5 else (0, 0, src.shape[1], src.shape[0])
img = lab[y0:y1, x0:x1].copy()
z = max(1, int(1100 / max(img.shape[:2])))
img = cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
step = 50 if z == 1 else 25
for gx in range((x0 // step + 1) * step, x1, step):
    X = (gx - x0) * z
    cv2.line(img, (X, 0), (X, img.shape[0]), (0, 170, 0), 1)
    cv2.putText(img, str(gx), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 120, 0), 1)
for gy in range((y0 // step + 1) * step, y1, step):
    Y = (gy - y0) * z
    cv2.line(img, (0, Y), (img.shape[1], Y), (0, 170, 0), 1)
    cv2.putText(img, str(gy), (2, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 120, 0), 1)
out = os.path.join(ML, "preview", f"grid_{p}.png")
cv2.imwrite(out, img)
print(out, img.shape)
