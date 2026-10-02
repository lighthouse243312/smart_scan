"""Straighten a photographed page before handwriting analysis (reference for the native
`straighten` step).

1. rules: the page's long printed lines (table borders, writing grids) — fit the outermost
   horizontal and vertical ones, intersect them into a quad and warp it to a rectangle; that
   fixes rotation AND keystone even when the sheet's own edges are out of frame.
2. no usable rules: deskew by the angle that levels the print (neutral ink only, so a
   coloured pen can't drag it), applied only when clearly better than leaving it.
3. a page that is already level (scanner output) comes back unchanged.

usage: straighten.py input outdir   → <outdir>/<name>_straight.png + _debug.png
"""
import os
import sys

import cv2
import numpy as np

WORK = 1200          # analysis size (long side)
MAX_DEG = 8.0


def ink_map(small, neutral_only=False):
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(cv2.dilate(g, np.ones((7, 7), np.uint8)), 31)
    ink = (bg.astype(np.int32) - g) > 25
    if neutral_only:
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        ink &= np.hypot(lab[..., 1] - 128, lab[..., 2] - 128) < 12
    return ink.astype(np.uint8) * 255


def fit_line(mask):
    """Least-squares line through a thin component: returns (point, unit direction)."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def long_lines(ink, horizontal):
    """Long, thin, nearly-axis-aligned printed lines as (point, dir, length, mask)."""
    H, W = ink.shape
    span = W if horizontal else H
    k = max(15, span // 25)
    kern = np.ones((1, k), np.uint8) if horizontal else np.ones((k, 1), np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kern)
    # re-join a rule the photo's slope or a pen crossing broke into pieces
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lines = cv2.dilate(lines, np.ones((3, 1) if horizontal else (1, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        length = w if horizontal else h
        thick = h if horizontal else w
        if length < 0.35 * span or thick > 0.08 * span:
            continue
        m = labels == i
        p, d = fit_line(m)
        if not horizontal:
            d = np.array([d[1], d[0]]) if abs(d[1]) < abs(d[0]) else d
        angle = np.degrees(np.arctan2(d[1], d[0])) if horizontal else np.degrees(np.arctan2(-d[0], d[1]))
        angle = (angle + 90) % 180 - 90
        if abs(angle) > MAX_DEG:
            continue
        out.append((p, d / np.linalg.norm(d), length, angle))
    return out


def intersect(p1, d1, p2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], np.float64)
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, p2 - p1)[0]
    return p1 + t * d1


def short_verticals(ink):
    """Short vertical rule pieces (grid / table columns): (x centre, angle°, length)."""
    H, W = ink.shape
    k = max(12, H // 60)
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(v, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.04 * H or w > 0.02 * W:
            continue
        p, d = fit_line(labels == i)
        if abs(d[1]) < abs(d[0]):
            continue
        if d[1] < 0:
            d = -d
        angle = np.degrees(np.arctan2(-d[0], d[1]))           # 0 = vertical
        if abs(angle) <= MAX_DEG:
            out.append((cents[i][0], angle, h))
    return out


def rules_quad(small):
    """Quad (in `small` coords) squaring the page's rule field, or None.

    Horizontal rules' angle is fitted as a function of height and vertical rule pieces' angle
    as a function of x (perspective makes both drift across the page); the quad is the four
    lines at the outer rules, intersected.
    """
    ink = ink_map(small)
    H, W = ink.shape
    hs = long_lines(ink, True)
    vs = short_verticals(ink)
    if len(hs) < 2:
        return None, hs, vs
    hy = np.array([p[1] + (W / 2 - p[0]) * d[1] / d[0] for p, d, *_ in hs])
    ha = np.array([a for *_, a in hs]); hl = np.array([l for _, _, l, _ in hs])
    if hy.max() - hy.min() < 0.25 * H:
        return None, hs, vs
    fh = np.polyfit(hy, ha, 1, w=hl) if len(hs) >= 3 else np.array([0.0, np.average(ha, weights=hl)])
    y_top, y_bot = hy.min(), hy.max()
    if len(vs) >= 6:
        vx = np.array([x for x, _, _ in vs]); va = np.array([a for _, a, _ in vs]); vl = np.array([l for *_, l in vs])
        fv = np.polyfit(vx, va, 1, w=vl) if vx.max() - vx.min() > 0.3 * W else np.array([0.0, np.average(va, weights=vl)])
        x_l, x_r = np.percentile(vx, 5), np.percentile(vx, 95)
    else:
        # no verticals to read: keep them square to the mean horizontal (rotation only)
        fv = np.array([0.0, float(np.average(ha, weights=hl))])
        x_l, x_r = 0.1 * W, 0.9 * W
    if x_r - x_l < 0.3 * W:
        x_l, x_r = 0.1 * W, 0.9 * W

    def hline(y):
        a = np.radians(np.polyval(fh, y))
        return np.array([W / 2, y]), np.array([np.cos(a), np.sin(a)])

    def vline(x):
        a = np.radians(np.polyval(fv, x))
        return np.array([x, H / 2]), np.array([-np.sin(a), np.cos(a)])

    pts = [intersect(*hline(y), *vline(x)) for y, x in ((y_top, x_l), (y_top, x_r), (y_bot, x_r), (y_bot, x_l))]
    if any(p is None for p in pts):
        return None, hs, vs
    return np.array(pts, np.float32), hs, vs


def warp_keep_page(img, quad):
    """Homography that squares `quad`, applied to the WHOLE image (nothing outside the quad is
    cropped away), output sized to the warped image's bounding box."""
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32) + tl
    M = cv2.getPerspectiveTransform(quad, dst)
    H, W = img.shape[:2]
    corners = cv2.perspectiveTransform(np.array([[[0, 0], [W, 0], [W, H], [0, H]]], np.float32), M)[0]
    x0, y0 = corners.min(0)
    x1, y1 = corners.max(0)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    size = (int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0)))
    if size[0] > 1.6 * W or size[1] > 1.6 * H:          # degenerate fit
        return None
    return cv2.warpPerspective(img, T @ M, size, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def projection_skew(small):
    """(angle, confidence) levelling the print rows; confidence = best score / score at 0."""
    ink = ink_map(small, neutral_only=True)

    def score(a):
        M = cv2.getRotationMatrix2D((ink.shape[1] / 2, ink.shape[0] / 2), a, 1)
        prof = cv2.warpAffine(ink, M, (ink.shape[1], ink.shape[0])).sum(1).astype(np.float64)
        return np.square(np.diff(prof)).sum()

    angles = np.arange(-MAX_DEG, MAX_DEG + 1e-6, 0.25)
    s = [score(a) for a in angles]
    a0 = angles[int(np.argmax(s))]
    fine = np.arange(a0 - 0.25, a0 + 0.25 + 1e-6, 0.05)
    sf = [score(a) for a in fine]
    best = fine[int(np.argmax(sf))]
    return best, max(sf) / max(score(0.0), 1e-6)


def rotate(img, deg):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H * sin + W * cos), int(H * cos + W * sin)
    M[0, 2] += nW / 2 - W / 2
    M[1, 2] += nH / 2 - H / 2
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def straighten(img):
    H, W = img.shape[:2]
    s = WORK / max(H, W)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    quad, hs, vs = rules_quad(small)
    info = {'h_lines': len(hs), 'v_lines': len(vs)}
    if quad is not None:
        # already square (scan): all four sides of the fitted quad within 0.3° of the axes
        tl, tr, br, bl = quad
        sides = [np.degrees(np.arctan2(*(tr - tl)[::-1])), np.degrees(np.arctan2(*(br - bl)[::-1])),
                 np.degrees(np.arctan2(-(bl - tl)[0], (bl - tl)[1])), np.degrees(np.arctan2(-(br - tr)[0], (br - tr)[1]))]
        info['sides'] = [round(float(x), 2) for x in sides]
        if max(abs(x) for x in sides) < 0.6:
            return img, 'level', quad / s, info
        out = warp_keep_page(img, quad / s)
        if out is not None:
            return out, 'rules', quad / s, info
    a, conf = projection_skew(small)
    info['skew'], info['conf'] = round(float(a), 2), round(float(conf), 2)
    if abs(a) >= 0.3 and conf >= 1.15:
        return rotate(img, a), f'rot{a:.2f}', None, info
    return img, 'level', None, info


if __name__ == '__main__':
    src, outdir = sys.argv[1], sys.argv[2]
    img = cv2.imread(src)
    out, how, quad, info = straighten(img)
    name = os.path.splitext(os.path.basename(src))[0]
    cv2.imwrite(f'{outdir}/{name}_straight.png', out)
    dbg = img.copy()
    if quad is not None:
        cv2.polylines(dbg, [quad.astype(np.int32)], True, (0, 0, 255), 3)
    cv2.imwrite(f'{outdir}/{name}_debug.png', dbg)
    print(name, how, info, img.shape[1::-1], '->', out.shape[1::-1])
