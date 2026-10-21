"""Builds training/eval labels for real worksheet photos in ml/real/batch2/ (v6 -> v7 fine-tune).

Per page (working size: long side 1536, same as the app):
  ink        = noticeably darker than the local paper level
  colour pen = ink whose colour differs from the local paper colour by a large chroma margin
               (purple / red / blue ballpoint and marker; print is black/grey)
  hw target  = colour-pen ink, plus manual "hw" rectangles (black-pen handwriting)
  print      = all other ink
  ignored    = outside the paper, a 2 px band around hw strokes (anti-aliased colour fringe),
               the print channel on hw pixels (print hidden under pen is unknown on real photos),
               manual "ignore" rectangles (pictures / coloured print / anything ambiguous)
Manual corrections live in ml/real/batch2/overrides.json:
  {"p05": [{"r": [x0, y0, x1, y1], "l": "hw" | "print" | "ignore"}, ...], ...}
  rectangles in ORIGINAL image pixels; later entries win.

Outputs ml/real/batch2/labels/<page>.npz (y uint8 bit0=print bit1=hw, valid uint8 bit0/bit1,
image RGB at 1536) and ml/preview/label_<page>.png (blue = print, red = hw, grey = ignored).

Usage: venv/bin/python label_real.py [page ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

ML_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH = os.path.join(ML_DIR, "real", os.environ.get("LABEL_BATCH", "batch2"))
OUT = os.path.join(BATCH, "labels")
PREVIEW = os.path.join(ML_DIR, "preview")
LONG = 1536
INK_MARGIN = 22        # gray levels darker than local paper
CHROMA_PEN = 18.0      # Lab ab-distance from local paper colour -> coloured pen
CHROMA_PRINT = 10.0    # below this the ink is neutral (print or black pen)


def resize_long(a, interp=cv2.INTER_AREA):
    s = LONG / max(a.shape[:2])
    return cv2.resize(a, (round(a.shape[1] * s), round(a.shape[0] * s)), interpolation=interp), s


def paper_level(gray):
    bg = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    return cv2.medianBlur(bg, 21)


def paper_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    g = cv2.GaussianBlur(gray, (0, 0), 3)
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((g > max(thr, 120)) & (hsv[..., 1] < 80)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.ones_like(gray, bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = np.argwhere(lab == k)[:, ::-1].astype(np.int32)
    filled = np.zeros_like(m)
    cv2.fillConvexPoly(filled, cv2.convexHull(pts), 1)
    filled = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return filled > 0


def _open_rotated(u, length, angle, horizontal):
    h, w = u.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    r = cv2.warpAffine(u, M, (w, h), flags=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if horizontal else (1, length))
    o = cv2.morphologyEx(r, cv2.MORPH_OPEN, k)
    return cv2.warpAffine(o, cv2.invertAffineTransform(M), (w, h), flags=cv2.INTER_NEAREST)


def ruled_lines(ink):
    """straight horizontal / vertical ink runs (table borders, grid, answer lines), found by a
    thin line opening at small rotations (tilted photos). Runs >= 60 px are lines; shorter runs
    (34-60 px) only when they touch a long line (box sides)."""
    u = ink.astype(np.uint8)
    lng, sht = {}, {}
    for horizontal in (True, False):
        lng[horizontal] = np.zeros_like(u)
        sht[horizontal] = np.zeros_like(u)
        for ang in (-3.0, -1.5, 0.0, 1.5, 3.0):
            lng[horizontal] |= _open_rotated(u, 60, ang, horizontal)
            sht[horizontal] |= _open_rotated(u, 30, ang, horizontal)
    long_all = lng[True] | lng[False]
    out = long_all.copy()
    for horizontal in (True, False):
        # a short side must join the END of a long line of the other orientation (box corner),
        # not its middle (handwriting touching a box border)
        other = lng[not horizontal]
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(other, connectivity=8)
        ends = np.zeros_like(u)
        for k in range(1, n2):
            x, y, w, h = st2[k, :4]
            comp = (lab2[y:y + h, x:x + w] == k)
            e = np.zeros_like(comp)
            if horizontal:      # other = vertical lines: ends are top / bottom 10 px
                e[:10], e[-10:] = comp[:10], comp[-10:]
            else:               # other = horizontal lines: ends are left / right 10 px
                e[:, :10], e[:, -10:] = comp[:, :10], comp[:, -10:]
            ends[y:y + h, x:x + w] |= e.astype(np.uint8)
        near = cv2.dilate(ends, np.ones((9, 9), np.uint8)) > 0
        sh = sht[horizontal] & ~long_all
        n, lab = cv2.connectedComponents(sh, connectivity=8)
        # both ends must reach a long line (a box side spans corner to corner)
        touch = np.bincount(lab[near], minlength=n)
        size = np.bincount(lab.ravel(), minlength=n)
        ok = touch >= 2
        ok[0] = False
        out |= ok[lab].astype(np.uint8)
    # short sides also touch the OTHER orientation's long lines (box corners)
    return (out > 0) & ink


def ink_chroma(rgb, ink):
    """ab-distance of each ink pixel from the local paper colour, median-smoothed over ink only."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    paper = np.where(ink[..., None], np.nan, lab)
    # local paper colour: blur of non-ink pixels
    w = (~ink).astype(np.float32)
    k = (0, 0)
    num = cv2.GaussianBlur(np.nan_to_num(paper) * w[..., None], k, 12)
    den = cv2.GaussianBlur(w, k, 12)[..., None] + 1e-6
    bg = num / den
    d = np.sqrt(((lab[..., 1:] - bg[..., 1:]) ** 2).sum(-1))
    # smooth along strokes: mean over 3x3 ink neighbours (anti-aliased edges have weak colour)
    di = (d * ink).astype(np.float32)
    s = cv2.blur(di, (3, 3)) / (cv2.blur(ink.astype(np.float32), (3, 3)) + 1e-6)
    return np.where(ink, np.maximum(d, s), 0)


stats = {}


def build(page, overrides):
    sp = os.path.join(BATCH, page + ".jpg")
    if not os.path.exists(sp): sp = os.path.join(BATCH, page + ".png")
    src = cv2.imread(sp, cv2.IMREAD_COLOR)[..., ::-1]
    rgb, s = resize_long(np.ascontiguousarray(src))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contrast = paper_level(gray).astype(np.int16) - gray.astype(np.int16)
    ink = contrast > INK_MARGIN
    ink = (cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0) | (contrast > 2 * INK_MARGIN)
    ch = ink_chroma(rgb, ink)
    # faint but clearly coloured pen (light pink / red ticks) counts as ink too
    faint_pen = (contrast > INK_MARGIN // 2) & ~ink
    if faint_pen.any():
        ch_f = ink_chroma(rgb, ink | faint_pen)
        faint_pen &= ch_f > 1.6 * next((o['chroma_pen'] for o in overrides.get(page, []) if 'chroma_pen' in o), CHROMA_PEN)
        faint_pen = cv2.morphologyEx(faint_pen.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        ink = ink | faint_pen
        ch = np.where(faint_pen, ch_f, ch)
    chroma_pen = next((o["chroma_pen"] for o in overrides.get(page, []) if "chroma_pen" in o), float(os.environ.get("LABEL_CHROMA_PEN", CHROMA_PEN)))
    pen = ink & (ch > chroma_pen)
    lines = ruled_lines(ink)
    v6p = os.path.join(OUT, page + "_v6.npy")
    phw = np.load(v6p).astype(np.float32)[..., 1] if os.path.exists(v6p) else np.zeros(ink.shape, np.float32)
    strokes = ink & ~lines
    # base: current model's handwriting (pixel level) + coloured-pen stroke components
    n, lab = cv2.connectedComponents(strokes.astype(np.uint8), connectivity=8)
    cnt = np.bincount(lab.ravel(), minlength=n).astype(np.float32) + 1e-6
    f_pen = np.bincount(lab.ravel(), weights=pen.ravel().astype(np.float32), minlength=n) / cnt
    is_pen = f_pen > next((o["pen_frac"] for o in overrides.get(page, []) if "pen_frac" in o), 0.5)
    if chroma_pen < CHROMA_PEN:
        # a low colour bar (blue-black ballpoint) also passes grey printed pictures and a grey page
        # edge: there a stroke counts as pen only when strongly coloured, or when the model sees
        # some handwriting in it (it only vetoes, it does not decide)
        hi = ink & (ch > CHROMA_PEN)
        f_hi = np.bincount(lab.ravel(), weights=hi.ravel().astype(np.float32), minlength=n) / cnt
        m_hw = np.bincount(lab.ravel(), weights=phw.ravel(), minlength=n) / cnt
        is_pen = (f_hi > 0.5) | (is_pen & (m_hw > 0.2))
    is_pen[0] = False
    hw = (strokes & (phw > 0.4)) | (is_pen[lab] & strokes)
    # shape evidence: whole strokes the fine-tuned model (v20, trained on these pages' corrected
    # labels) is sure are handwriting — colour cannot separate a dull purple / black pen from print.
    # Whole components only (a stroke fused with print dilutes its mean and is left to the rules);
    # every page's additions were reviewed by eye, wrong ones are removed with "print" rectangles.
    v20p = os.path.join(OUT, page + "_v20.npy")
    if os.path.exists(v20p) and not any(o.get("no_model") for o in overrides.get(page, [])):
        p20 = np.load(v20p).astype(np.float32)[..., 1]
        mean20 = np.bincount(lab.ravel(), weights=p20.ravel(), minlength=n) / cnt
        frac20 = np.bincount(lab.ravel(), weights=(p20 > 0.5).ravel().astype(np.float32), minlength=n) / cnt
        sure = (mean20 > 0.6) & (frac20 > 0.6)
        sure[0] = False
        hw |= sure[lab] & strokes
    # ruled-line pixels are handwriting only where coloured pen actually crosses them
    hw |= lines & pen & (cv2.dilate(hw.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    # solid strokes: stroke components with a real dark core (drops JPEG / paper speckle)
    strong = contrast > 2 * INK_MARGIN
    n2, lab2 = cv2.connectedComponents(strokes.astype(np.uint8), connectivity=8)
    core = np.bincount(lab2.ravel(), weights=strong.ravel().astype(np.float32), minlength=n2)
    size = np.bincount(lab2.ravel(), minlength=n2)
    ok = (core >= 6) & (size >= 15)
    ok[0] = False
    solid = ok[lab2]
    ign = np.zeros_like(ink)
    unk = np.zeros_like(ink)
    # punch holes / solid dark discs: not ink of either kind
    filled = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    w3, h3, a3 = st3[:, cv2.CC_STAT_WIDTH], st3[:, cv2.CC_STAT_HEIGHT], st3[:, cv2.CC_STAT_AREA]
    disc = (w3 >= 22) & (h3 >= 22) & (w3 <= 70) & (h3 <= 70) & (a3 > 0.62 * w3 * h3) & (np.abs(w3 - h3) < 0.3 * np.maximum(w3, h3))
    disc[0] = False
    for k in np.nonzero(disc)[0]:  # must be round: circularity of the outer contour
        cs, _ = cv2.findContours((lab3[st3[k, 1]:st3[k, 1] + h3[k], st3[k, 0]:st3[k, 0] + w3[k]] == k).astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        disc[k] = per > 0 and 4 * np.pi * cv2.contourArea(c) / per ** 2 > 0.7
    if disc.any():
        dm = cv2.dilate(disc[lab3].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        ign |= dm
        hw &= ~dm

    # cautious batches (labels from the model's hints only): ink the model is not clearly calling print
    # and not handwriting is neither — left unscored rather than taught as print (LABEL_CAUTIOUS=1)
    if os.environ.get("LABEL_CAUTIOUS") == "1":
        # but print the model clearly reads as print stays scored as print — it is exactly the print
        # beside the pen that must be taught not to erase (unscoring it let the next model spill there)
        pprint = np.load(v6p).astype(np.float32)[..., 0] if os.path.exists(v6p) else np.zeros(ink.shape, np.float32)
        unk |= ink & ~hw & (phw > 0.15) & ~((pprint > 0.6) & (phw < 0.5))
    # coloured pictures and colour blocks (printed illustrations, a torn-page edge): dense colour,
    # not pen — pen strokes are thin and sparse in their box. Not scored (opt-in: AUTO_PICTURES=1)
    if os.environ.get("AUTO_PICTURES") == "1":
        blob = cv2.morphologyEx((pen | hw).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        nb, lb, sb, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
        for i in range(1, nb):
            bx, by, bw, bh, ba = sb[i]
            sel = lb[by:by + bh, bx:bx + bw] == i
            dens = (hw[by:by + bh, bx:bx + bw] & sel).sum() / max(1, bw * bh)
            solid = (pen[by:by + bh, bx:bx + bw] & sel).sum() / max(1, sel.sum())
            if bw * bh >= 2500 and dens >= 0.22 and solid >= 0.5:
                ign[by:by + bh, bx:bx + bw] |= sel
                hw[by:by + bh, bx:bx + bw] &= ~sel
    for o in overrides.get(page, []):
        if "r" not in o:
            continue
        x0, y0, x1, y1 = [int(round(v * s)) for v in o["r"]]
        sl = (slice(max(y0, 0), y1), slice(max(x0, 0), x1))
        if o["l"] == "hw":           # all solid ink strokes except ruled lines
            hw[sl] = solid[sl] | hw[sl]
        elif o["l"] == "hwall":      # all ink, ruled lines included
            hw[sl] = ink[sl]
        elif o["l"] == "print":
            hw[sl] = False
        elif o["l"] == "ignore":
            ign[sl] = True
        elif o["l"] == "no_pen":         # (handled with the pen proposals below)
            pass
        elif o["l"] == "neutral_unknown":   # black-pen marks mixed with print: only colour pen is
            unk[sl] |= ink[sl] & ~hw[sl]    # scored inside the rect, neutral ink is not scored
        elif o["l"] == "pen":      # colour-pen only (keeps neutral print inside the rect)
            hw[sl] = pen[sl] | (hw[sl] & (ch[sl] > CHROMA_PRINT))
        else:
            raise ValueError(o)

    # black-pen marks (circles, ticks, dots, lines) inside the regions colour cannot settle: the
    # pen-trained model v23 proposes the pixels it sees as pen and NOT print; every page's proposals
    # were reviewed by eye, wrong ones are removed with "no_pen" rectangles. Scored as handwriting;
    # the rest of the region (print, and print under the pen) stays unscored.
    v23p = os.path.join(OUT, page + "_v23.npy")
    if unk.any() and os.path.exists(v23p):
        p23 = np.load(v23p).astype(np.float32)
        pen23 = (p23[..., 1] > 0.5) & (p23[..., 0] < 0.5) & ink & unk
        for o in overrides.get(page, []):
            if o.get("l") == "no_pen":
                x0, y0, x1, y1 = [int(round(v * s)) for v in o["r"]]
                pen23[max(y0, 0):y1, max(x0, 0):x1] = False
        hw |= pen23
        unk &= ~pen23

    # hand-painted truth (label_tool.py) wins over every automatic rule: colour cannot tell a black
    # pen from black print, a person can. Painted ink is scored even inside "neutral_unknown".
    brush_hw = np.zeros_like(ink)
    bp = os.path.join(BATCH, "brush", page + ".png")
    if os.path.exists(bp):
        b = cv2.imread(bp, cv2.IMREAD_UNCHANGED)
        if b.shape[:2] != ink.shape:
            b = cv2.resize(b, (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_NEAREST)
        a = b[..., 3] > 0
        red, green, blue = b[..., 2] > 127, b[..., 1] > 127, b[..., 0] > 127
        b_hw, b_print, b_ign = a & red & ~green & ~blue, a & blue & ~red & ~green, a & green & ~red & ~blue
        brush_hw = b_hw & (contrast > INK_MARGIN // 2)     # faint pencil counts when a person says so
        hw[b_hw] = brush_hw[b_hw]
        hw[b_print] = False
        ign |= b_ign
        unk &= ~(b_hw | b_print)

    # print: ink with a solid dark core (grown 1 px into the surrounding ink); faint ink (show-
    # through from the back side, halos, JPEG ringing) and a 1 px ring are ignored for print
    core = contrast > 2 * INK_MARGIN
    # handwriting must be real stroke: a dark core (grown 1 px) or clearly coloured pen; the rest
    # (tinted paper, JPEG speckle around digits) is not scored
    stroke_ok = core | pen
    loose_hw = hw & ~stroke_ok & ~brush_hw
    hw &= stroke_ok | brush_hw
    grid = any(o.get("grid") for o in overrides.get(page, []))
    if grid:   # grid paper: faint grid lines are print
        print_ = ink & ~hw & (core | lines)
    else:
        print_ = ink & ~hw & core
    faint = (contrast > INK_MARGIN // 2) & ~hw & ~print_   # weak ink: unscored for print
    ring = cv2.dilate(print_.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~print_
    valid = paper_mask(rgb) & ~ign
    for o in overrides.get(page, []):
        if "paper" in o:  # manual paper polygon (original px): everything outside is ignored
            poly = np.round(np.array(o["paper"], np.float32) * s).astype(np.int32)
            pm = np.zeros(valid.shape, np.uint8)
            cv2.fillPoly(pm, [poly], 1)
            valid = (pm > 0) & ~ign
    band = cv2.dilate(hw.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~hw
    # faint ink right around handwriting (anti-aliased stroke edge) is neither scored as hw nor
    # as paper: the erase may or may not take it, both are fine
    band |= cv2.dilate(hw.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & (contrast > INK_MARGIN // 2) & ~core & ~hw
    # a printed letter's dark core right beside the pen is print for sure: only the band's faint rim
    # is left unscored for print (unscoring the core taught nothing about print next to the pen,
    # and a model then spilled the erase onto it)
    v_print = valid & ~hw & ~(band & ~core) & ~faint & ~ring & ~loose_hw
    # pen crossing a printed rule: overlap pixel, not scored for hw (the erase keeps the rule)
    v_hw = valid & ~band & ~unk & ~loose_hw & ~(hw & cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    v_print = v_print & ~unk
    # printed rule pixels touching handwriting: overlap, not scored for print either
    rule_ov = lines & (cv2.dilate(hw.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0)
    # (for print: only where the pen actually crosses the rule; the rule beyond is print)
    v_print = v_print & ~(lines & (cv2.dilate(hw.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
    v_hw = v_hw & ~rule_ov
    # ── pen vs print, case by case (each ink pixel falls in exactly one) ──
    #  A. pure handwriting: pen with no print under or right beside it (2 px of a printed glyph's
    #     core) — 100% handwriting: hw = 1 AND print = 0, both scored
    #  B. overlap: pen crossing printed text — the only place print may lie under pen: hw = 1
    #     scored; print unknown on a real photo (which pixels of the glyph the pen hides), unscored
    #  C. pure print: hw = 0 and print = 1, both scored (set above)
    #  D. pen on a printed rule: neither scored (the erase keeps and redraws the rule; set above)
    text = print_ & ~ign & ~(cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)   # printed text (scored, not rules: D)
    # a printed glyph the pen crosses stands mostly clear of the pen; slivers of the pen's own rim
    # that the automatic rules left as "print" lie wholly along it — they are not print under pen
    pen_zone = cv2.dilate(hw.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    nt, lt = cv2.connectedComponents(text.astype(np.uint8), connectivity=8)
    clear = np.bincount(lt.ravel(), (text & ~pen_zone).ravel().astype(np.float64), nt)
    size = np.bincount(lt.ravel(), text.ravel().astype(np.float64), nt)
    glyph = (clear >= 0.3 * np.maximum(size, 1)) & (size >= 6); glyph[0] = False
    text = glyph[lt]
    print_near = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    overlap = hw & print_near
    pure_hw = hw & ~print_near
    base = valid & ~unk & ~loose_hw & ~rule_ov
    v_print = (v_print & ~overlap) | (pure_hw & base)          # A: print 0 scored; B: unscored
    v_hw = v_hw | (overlap & base & ~band)                      # B: the pen itself is scored
    stats["overlap"] = int(overlap.sum()); stats["pure_hw"] = int(pure_hw.sum())
    # hand-painted handwriting is the truth: always scored as handwriting, whatever the automatic
    # rules above say (a pen strike-through is a long thin run, i.e. a "rule" to them)
    v_hw = v_hw | (brush_hw & valid)
    y = print_.astype(np.uint8) | (hw.astype(np.uint8) << 1)
    v = v_print.astype(np.uint8) | (v_hw.astype(np.uint8) << 1)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, page + ".npz"), image=rgb, y=y, valid=v,
                        ink=(contrast > INK_MARGIN // 2).astype(np.uint8))

    vis = (rgb.astype(np.float32) * 0.35 + 255 * 0.65).astype(np.uint8)
    vis[print_] = (30, 60, 230)
    vis[hw] = (230, 30, 30)
    vis[~valid] = (vis[~valid] * 0.5).astype(np.uint8)
    os.makedirs(PREVIEW, exist_ok=True)
    cv2.imwrite(os.path.join(PREVIEW, f"label_{page}.png"), vis[..., ::-1])
    return int(hw.sum()), int(print_.sum())


def main():
    ov_p = os.path.join(BATCH, "overrides.json")
    overrides = json.load(open(ov_p)) if os.path.exists(ov_p) else {}
    pages = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in (glob.glob(os.path.join(BATCH, "*.jpg")) + glob.glob(os.path.join(BATCH, "*.png"))))
    for p in pages:
        h, pr = build(p, overrides)
        print(f"{p}: hw {h} px, print {pr} px, pure hw (A) {stats.get('pure_hw')} px, overlap (B) {stats.get('overlap')} px")


if __name__ == "__main__":
    main()
