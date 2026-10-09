"""Teaches the segmenter that ink colour doesn't decide pen vs print — SHAPE does.

Draws onto the real labelled worksheet pages (make_pseudo_labels.py output):

  pen marks (label = handwriting), black / dark grey / blue-black:
    - circles / ovals around printed words, as a hand draws them: radius wobbling a few percent
      along the loop, centre drifting, start and end not meeting (gap or overlap), stroke width
      and darkness varying with pressure
    - ticks, strike-throughs across a word, wavy underlines, crosses
  printed shapes (label = print), the same colours:
    - mathematically perfect circles, ellipses, rounded / square boxes and straight rules with a
      constant stroke width — so "round" alone is never a reason to erase

usage: augment_black_pen.py <labels dir> <pages dir> <out labels dir> <out pages dir> [--variants 3]
"""
import argparse
import glob
import os

import cv2
import numpy as np


def smooth_noise(n, rng, k=6, amp=1.0):
    """Low-frequency 1D noise of length n (a few random sine harmonics)."""
    t = np.linspace(0, 2 * np.pi, n)
    out = np.zeros(n)
    for f in range(1, k + 1):
        out += rng.normal(0, 1 / f) * np.sin(f * t + rng.uniform(0, 2 * np.pi))
    return amp * out / max(np.abs(out).max(), 1e-6)


def draw_stroke(canvas, mask, pts, width, rng, color):
    """Pressure-varying polyline: per-segment width / darkness noise, antialiased."""
    n = len(pts)
    pressure = 1 + 0.35 * smooth_noise(n, rng, k=4)
    pressure *= np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / max(3, n * 0.06), 0.45, 1)  # taper ends
    for i in range(n - 1):
        w = max(1, int(round(width * pressure[i])))
        dark = np.clip(0.75 + 0.25 * pressure[i], 0.5, 1.0)
        c = tuple(int(255 - (255 - ch) * dark) for ch in color)
        p, q = tuple(np.round(pts[i]).astype(int)), tuple(np.round(pts[i + 1]).astype(int))
        cv2.line(canvas, p, q, c, w, cv2.LINE_AA)
        cv2.line(mask, p, q, 255, w + 1, cv2.LINE_8)


def hand_loop(cx, cy, rx, ry, rng):
    """A hand-drawn closed loop: wobbling radius, drifting centre, gap/overlap at the join."""
    n = int(max(60, (rx + ry) * 1.2))
    start = rng.uniform(0, 2 * np.pi)
    sweep = 2 * np.pi * rng.uniform(0.88, 1.15)          # gap (<1) or overlap (>1)
    t = start + np.linspace(0, sweep, n) * rng.choice([-1, 1])
    wob = 1 + rng.uniform(0.03, 0.09) * smooth_noise(n, rng, k=5)
    drift = np.linspace(0, 1, n)[:, None] * rng.normal(0, 0.06, 2) * [rx, ry]
    rot = rng.uniform(-0.25, 0.25)
    x = rx * wob * np.cos(t); y = ry * wob * np.sin(t)
    xr, yr = x * np.cos(rot) - y * np.sin(rot), x * np.sin(rot) + y * np.cos(rot)
    return np.stack([cx + xr, cy + yr], 1) + drift


def hand_line(p0, p1, rng, wave=0.0):
    n = int(max(20, np.hypot(*(np.subtract(p1, p0))) / 2))
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) * np.array(p0, float) + t * np.array(p1, float)
    d = np.subtract(p1, p0); nrm = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6)
    bow = rng.normal(0, 0.03) * np.hypot(*d) * np.sin(np.pi * t)              # slight bow
    wig = (wave * np.sin(t * rng.uniform(6, 14) * np.pi) + smooth_noise(n, rng, k=3, amp=0.8)[:, None])
    return pts + nrm * (bow + wig)


def word_boxes(print_mask, gh):
    """Printed words: print components merged along the row."""
    m = cv2.morphologyEx(print_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((max(3, gh // 3), max(5, gh)), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    boxes = [st[i, :4] for i in range(1, n) if gh * 0.6 < st[i, 3] < gh * 2.5 and gh * 1.2 < st[i, 2] < gh * 12]
    return boxes


def augment(img, hw, pr, valid, hwvalid, rng):
    img = img.copy(); hw = hw.copy(); pr = pr.copy()
    H, W = hw.shape
    n_pr, _, st, _ = cv2.connectedComponentsWithStats(pr.astype(np.uint8))
    hts = st[1:, 3][(st[1:, 3] > 5) & (st[1:, 3] < H * 0.04)]
    gh = int(np.median(hts)) if len(hts) else max(10, H // 80)
    stroke_w = max(1, int(round(gh * rng.uniform(0.11, 0.2))))
    pen_layer = np.full_like(img, 255); pen_mask = np.zeros((H, W), np.uint8)
    print_layer = np.full_like(img, 255); print_mask = np.zeros((H, W), np.uint8)
    inks = [(30, 30, 30), (60, 60, 60), (90, 85, 80), (70, 40, 30), (25, 25, 45)]   # BGR: black, greys, blue-black
    boxes = word_boxes(pr, gh)
    rng.shuffle(boxes)
    k = 0
    for x, y, w, h in boxes[:int(rng.integers(6, 14))]:
        color = inks[rng.integers(len(inks))]
        kind = rng.choice(["circle", "circle", "circle", "strike", "tick", "underline", "cross"])
        cx, cy = x + w / 2, y + h / 2
        if kind == "circle":
            pts = hand_loop(cx, cy, w / 2 + rng.uniform(0.3, 0.8) * gh, h / 2 + rng.uniform(0.3, 0.7) * gh, rng)
        elif kind == "strike":
            pts = hand_line((x - 0.2 * gh, cy + rng.normal(0, 0.15) * h), (x + w + 0.2 * gh, cy + rng.normal(0, 0.15) * h), rng)
        elif kind == "underline":
            pts = hand_line((x, y + h + 0.25 * gh), (x + w, y + h + 0.25 * gh + rng.normal(0, 0.1) * gh), rng, wave=rng.uniform(0, 0.25) * gh)
        elif kind == "tick":
            s = gh * rng.uniform(0.8, 1.6); bx, by = x + w + gh * 0.5, cy
            a = hand_line((bx, by), (bx + s * 0.35, by + s * 0.45), rng); b = hand_line((bx + s * 0.35, by + s * 0.45), (bx + s, by - s * 0.7), rng)
            pts = np.concatenate([a, b])
        else:
            s = gh * rng.uniform(0.7, 1.3); bx, by = x + w + gh * 0.6, cy
            draw_stroke(pen_layer, pen_mask, hand_line((bx, by - s / 2), (bx + s, by + s / 2), rng), stroke_w, rng, color)
            pts = hand_line((bx + s, by - s / 2), (bx, by + s / 2), rng)
        draw_stroke(pen_layer, pen_mask, pts, stroke_w, rng, color)
        k += 1
    # perfect printed shapes, constant width, in free paper areas (and some around words too)
    ink_any = (hw | pr | (pen_mask > 0))
    for _ in range(int(rng.integers(3, 8))):
        color = inks[rng.integers(len(inks))]
        for _try in range(20):
            r = int(gh * rng.uniform(0.7, 3.0)); cx, cy = int(rng.uniform(r + 5, W - r - 5)), int(rng.uniform(r + 5, H - r - 5))
            if not ink_any[max(0, cy - r - 4):cy + r + 4, max(0, cx - r - 4):cx + r + 4].any() and valid[cy, cx]:
                break
        else:
            continue
        t = max(1, int(round(gh * rng.uniform(0.08, 0.16))))
        shape = rng.choice(["circle", "ellipse", "rect", "round"])
        tmp = np.zeros((H, W), np.uint8)
        if shape == "circle":
            cv2.circle(tmp, (cx, cy), r, 255, t, cv2.LINE_AA)
        elif shape == "ellipse":
            cv2.ellipse(tmp, (cx, cy), (r, int(r * rng.uniform(0.4, 0.8))), rng.uniform(0, 180), 0, 360, 255, t, cv2.LINE_AA)
        elif shape == "rect":
            cv2.rectangle(tmp, (cx - r, cy - r // 2), (cx + r, cy + r // 2), 255, t, cv2.LINE_AA)
        else:
            q = max(2, r // 3)
            cv2.rectangle(tmp, (cx - r + q, cy - r // 2), (cx + r - q, cy + r // 2), 255, t, cv2.LINE_AA)
            cv2.rectangle(tmp, (cx - r, cy - r // 2 + q), (cx + r, cy + r // 2 - q), 255, t, cv2.LINE_AA)
            for sx, sy in ((cx - r + q, cy - r // 2 + q), (cx + r - q, cy - r // 2 + q), (cx - r + q, cy + r // 2 - q), (cx + r - q, cy + r // 2 - q)):
                cv2.circle(tmp, (sx, sy), q, 255, t, cv2.LINE_AA)
            inner = np.zeros_like(tmp); cv2.rectangle(inner, (cx - r + q + t, cy - r // 2 + t), (cx + r - q - t, cy + r // 2 - t), 255, -1)
            cv2.rectangle(inner, (cx - r + t, cy - r // 2 + q + t), (cx + r - t, cy + r // 2 - q - t), 255, -1)
            tmp[inner > 0] = 0
        a = tmp.astype(np.float32)[..., None] / 255
        print_layer = (print_layer * (1 - a) + np.array(color) * a).astype(np.uint8)
        print_mask |= (tmp > 127).astype(np.uint8) * 255
    # multiply-blend both layers onto the page (ink on paper)
    out = (img.astype(np.float32) * pen_layer / 255 * print_layer / 255).astype(np.uint8)
    pm, qm = pen_mask > 0, print_mask > 0
    hw2 = hw | pm
    pr2 = (pr & ~pm) | qm
    # handwriting is now known around the drawn marks even on black-pen pages
    near = cv2.dilate(pen_mask, np.ones((15, 15), np.uint8)) > 0
    hwvalid2 = hwvalid | (near & valid) | (qm & valid)
    return out, hw2, pr2, valid, hwvalid2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("pages"); ap.add_argument("out_labels"); ap.add_argument("out_pages")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--skip", default="p03,p09,p14", help="held-out pages: never augmented")
    a = ap.parse_args()
    os.makedirs(a.out_labels, exist_ok=True); os.makedirs(a.out_pages, exist_ok=True)
    rng = np.random.default_rng(7)
    skip = set(a.skip.split(","))
    for p in sorted(glob.glob(os.path.join(a.labels, "*_hw.png"))):
        name = os.path.basename(p)[:-len("_hw.png")]
        if name in skip or "_bp" in name:
            continue
        rd = lambda s: cv2.imread(os.path.join(a.labels, f"{name}_{s}.png"), cv2.IMREAD_GRAYSCALE) > 127
        flat = cv2.imread(os.path.join(a.labels, f"{name}_img.png"))
        raw = cv2.imread(os.path.join(a.pages, f"{name}_original.png"))
        raw = cv2.resize(raw, flat.shape[1::-1], interpolation=cv2.INTER_AREA)
        hw, pr, valid, hwvalid = rd("hw"), rd("print"), rd("valid"), rd("hwvalid")
        for v in range(a.variants):
            seed = rng.integers(1 << 30)
            outs = []
            for base in (flat, raw):          # same marks on the flattened and the raw capture
                r2 = np.random.default_rng(seed)
                outs.append(augment(base, hw, pr, valid, hwvalid, r2))
            tag = f"{name}_bp{v}"
            cv2.imwrite(os.path.join(a.out_labels, f"{tag}_img.png"), outs[0][0])
            cv2.imwrite(os.path.join(a.out_pages, f"{tag}_original.png"), outs[1][0])
            for s, m in zip(("hw", "print", "valid", "hwvalid"), outs[0][1:]):
                cv2.imwrite(os.path.join(a.out_labels, f"{tag}_{s}.png"), m.astype(np.uint8) * 255)
            viz = np.full_like(flat, 255); viz[~outs[0][3]] = 200; viz[outs[0][2]] = 0; viz[outs[0][1]] = (0, 0, 230)
            cv2.imwrite(os.path.join(a.out_labels, f"{tag}_viz.jpg"), np.hstack([outs[0][0], viz]))
        print(name, "→", a.variants, "variants", flush=True)


if __name__ == "__main__":
    main()
