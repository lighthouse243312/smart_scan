"""Local brush tool for correcting the real-page labels by hand (ml/real/batch2).

Colour alone cannot tell a black pen from black print, so the automatic labels (label_real.py)
leave such ink unscored or wrong. Here a person paints the truth over the page:
  red   = handwriting, blue = print, grey = ignore, eraser = back to the automatic label.
Brushes are saved to ml/real/batch2/brush/<page>.png (label resolution, RGBA: R = hw, B = print,
G = ignore) and label_real.py applies them last, over every automatic rule.

Shown under the brush: the current label (red hw / blue print, dimmed where unscored) and, in
yellow, where the model disagrees with the label — the likeliest places for missed handwriting.

Usage: venv/bin/python label_tool.py [--port 8765] [--model ink_segmenter_v20_torch.pt]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

ML_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH = os.path.join(ML_DIR, "real", "batch2")
LABELS = os.path.join(BATCH, "labels")
BRUSH = os.path.join(BATCH, "brush")
HTML = os.path.join(ML_DIR, "label_tool.html")
SUSPECT = os.path.join(LABELS, "suspect")


def pages():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(LABELS, "p*.npz")))


def png(a):
    ok, b = cv2.imencode(".png", a)
    return b.tobytes()


def label_overlay(page):
    d = np.load(os.path.join(LABELS, page + ".npz"))
    y, v = d["y"], d["valid"]
    h, w = y.shape
    out = np.zeros((h, w, 4), np.uint8)                          # BGRA
    hw, pr = (y & 2) > 0, ((y & 1) > 0) & ((y & 2) == 0)
    scored_hw, scored_pr = (v & 2) > 0, (v & 1) > 0
    out[pr] = (230, 60, 30, 110); out[pr & ~scored_pr] = (230, 60, 30, 40)
    out[hw] = (30, 30, 230, 170); out[hw & ~scored_hw] = (30, 30, 230, 60)
    return out


def suspects(page, model_path):
    """Yellow where the model and the label disagree (computed once per model, cached)."""
    os.makedirs(SUSPECT, exist_ok=True)
    tag = os.path.splitext(os.path.basename(model_path))[0]
    cache = os.path.join(SUSPECT, f"{page}_{tag}.png")
    if os.path.exists(cache):
        return open(cache, "rb").read()
    import torch
    import ink_seg_model as M
    from infer_util import predict_page
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = M.make_torch_model().to(dev)
    m.load_state_dict(torch.load(model_path, map_location=dev))
    m.eval()
    d = np.load(os.path.join(LABELS, page + ".npz"))
    pr = predict_page(m, d["image"], dev)
    yh = ((d["y"] & 2) > 0).astype(np.uint8)
    ink = d["ink"] > 0
    ph = pr[..., 1] > 0.5
    k3 = np.ones((3, 3), np.uint8)
    # a stroke's 1 px edge is not a disagreement: only whole pieces the two call differently
    disagree = ((ph & ~(cv2.dilate(yh, k3) > 0)) | ((cv2.erode(yh, k3) > 0) & ~ph)) & ink
    disagree = cv2.morphologyEx(disagree.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(disagree, k3), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= 25
    keep[0] = False
    disagree = (keep[lab] & (disagree > 0)).astype(np.uint8)
    ring = cv2.dilate(disagree, np.ones((9, 9), np.uint8)) - disagree
    out = np.zeros(yh.shape + (4,), np.uint8)
    out[ring > 0] = (0, 220, 255, 200)
    b = png(out)
    open(cache, "wb").write(b)
    return b


class H(BaseHTTPRequestHandler):
    model = None

    def _send(self, body, ctype="application/octet-stream", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        parts = self.path.split("?")[0].strip("/").split("/")
        try:
            if parts == [""]:
                return self._send(open(HTML, "rb").read(), "text/html; charset=utf-8")
            if parts == ["pages"]:
                done = {p: os.path.exists(os.path.join(BRUSH, p + ".png")) for p in pages()}
                return self._send(json.dumps(done).encode(), "application/json")
            kind, page = parts[0], parts[1]
            if page not in pages():
                return self._send(b"no such page", "text/plain", 404)
            if kind == "image":
                d = np.load(os.path.join(LABELS, page + ".npz"))
                ok, b = cv2.imencode(".jpg", d["image"][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                return self._send(b.tobytes(), "image/jpeg")
            if kind == "label":
                return self._send(png(label_overlay(page)), "image/png")
            if kind == "suspect":
                return self._send(suspects(page, self.model), "image/png")
            if kind == "brush":
                p = os.path.join(BRUSH, page + ".png")
                if not os.path.exists(p):
                    return self._send(b"", "image/png", 204)
                return self._send(open(p, "rb").read(), "image/png")
        except Exception as e:  # surface errors in the page, not a hung request
            return self._send(str(e).encode(), "text/plain", 500)
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "brush" or parts[1] not in pages():
            return self._send(b"bad request", "text/plain", 400)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        a = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
        if a is None or a.ndim != 3 or a.shape[2] != 4:
            return self._send(b"expected an RGBA png", "text/plain", 400)
        os.makedirs(BRUSH, exist_ok=True)
        cv2.imwrite(os.path.join(BRUSH, parts[1] + ".png"), a)
        self._send(b"{}", "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=os.path.join(ML_DIR, "ink_segmenter_v20a_torch.pt"))
    a = ap.parse_args()
    H.model = a.model
    print(f"label tool on http://localhost:{a.port}  (model for suspects: {os.path.basename(a.model)})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
