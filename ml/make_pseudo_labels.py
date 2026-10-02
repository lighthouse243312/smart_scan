"""Pseudo-labels for real photos from INK COLOUR, for fine-tuning the ink segmenter.

Most real worksheets are answered in a coloured pen (purple / blue / red) over black print. Once
the page is straightened and its lighting flattened (shadows / colour cast divided out per
channel), that colour difference is a reliable label source:

  ink    = darker than the local paper
  hw     = ink whose colour (stroke-smoothed) is far from the print colour
  print  = ink whose colour is close to the print colour
  ignore = in between, a band around pen strokes (print under / beside the pen is ambiguous),
           and everything off the paper

Pages whose pen is not coloured (black pen: almost no chromatic ink) get print labels only —
their handwriting channel is ignored entirely rather than labelled wrong.

usage: make_pseudo_labels.py <pages dir with *_original.png (the app's straightened page)> <out dir>
       writes <name>_img.png (flattened), <name>_hw.png, <name>_print.png, <name>_valid.png (255 = set)
       and <name>_viz.jpg (white paper, black print, red hw, grey ignore)
"""
import glob
import os
import sys

import cv2
import numpy as np


def flatten(img, text=15, med=41):
    """Divide out the paper's illumination per channel (shadows, colour cast)."""
    out = []
    for c in cv2.split(img):
        bg = cv2.morphologyEx(c, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (text, text)))
        bg = cv2.medianBlur(bg, med)
        out.append(np.clip(c.astype(np.float32) / np.maximum(bg, 1) * 235, 0, 255))
    return np.dstack(out).astype(np.uint8)


def paper_mask(flat):
    """The sheet on the FLATTENED page: bright and unsaturated (a desk / hand / fabric keeps its
    colour after flattening), the largest such region, holes (the print on it) filled."""
    small = cv2.resize(flat, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 170) & (hsv[..., 1] < 50)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = ((lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) * 255).astype(np.uint8)
    # fill holes: everything not reachable from the border through non-paper
    inv = cv2.bitwise_not(m)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv)
    h, w = inv.shape
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    for i in range(1, n):
        if i not in border:
            m[lab == i] = 255
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    return cv2.resize(m, (flat.shape[1], flat.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def label(orig):
    F = flatten(orig)
    lab = cv2.cvtColor(F, cv2.COLOR_BGR2LAB).astype(np.float32)
    dark = 255 - lab[..., 0]
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    paper = paper_mask(F)                 # on the flattened page: shadows are paper too
    ink = (dark > 45) & paper

    # print colour = the most common colour of the darkest ink (print outnumbers the pen)
    core = ink & (dark > 90)
    hist, ae, be = np.histogram2d(a[core], b[core], bins=40, range=[[-40, 40], [-40, 40]])
    ia, ib = np.unravel_index(np.argmax(hist), hist.shape)
    a0, b0 = (ae[ia] + ae[ia + 1]) / 2, (be[ib] + be[ib + 1]) / 2
    cd = np.hypot(a - a0, b - b0)
    # smooth colour over the stroke (edge pixels carry colour fringes)
    w = ink.astype(np.float32)
    cd_s = cv2.blur(cd * w, (5, 5)) / np.maximum(cv2.blur(w, (5, 5)), 1e-3)

    PEN, PRINT = 9.0, 5.0
    pen_px = ink & (cd_s > PEN)
    print_px = ink & (cd_s < PRINT)
    coloured_pen = pen_px.sum() / max(ink.sum(), 1)

    # printed rules (long straight runs) are print whatever pen touches them — they would
    # otherwise join a pen stroke's component and take its vote
    W = ink.shape[1]
    u8 = ink.astype(np.uint8)
    rules = (cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((1, max(25, W // 30)), np.uint8)) |
             cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((max(25, W // 30), 1), np.uint8))) > 0
    rules &= ~pen_px
    strokes = ink & ~rules

    # per stroke: a component mostly one colour takes that label (its fringe pixels follow)
    n, cl, st, _ = cv2.connectedComponentsWithStats(strokes.astype(np.uint8), connectivity=8)
    cnt = np.maximum(np.bincount(cl[strokes], minlength=n), 1)
    pen_share = np.bincount(cl[strokes], weights=pen_px[strokes], minlength=n) / cnt
    pr_share = np.bincount(cl[strokes], weights=print_px[strokes], minlength=n) / cnt
    hw = (strokes & (pen_share[cl] > 0.7)) | pen_px
    pr = (strokes & ((pr_share[cl] > 0.7) & ~pen_px | print_px & (pen_share[cl] < 0.3))) | rules

    valid = paper.copy()
    # ambiguous: ink that is neither, and print right next to the pen (crossings, overlap)
    near_pen = cv2.dilate(hw.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    valid &= ~(ink & ~hw & ~pr)
    valid &= ~(pr & near_pen & ~rules)
    hw_valid = valid.copy()
    if coloured_pen < 0.03:          # black pen page: don't claim anything about handwriting
        hw_valid[:] = False
        hw[:] = False
    return F, hw, pr & ~hw, valid, hw_valid, coloured_pen


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(src, '*_original.png'))):
        name = os.path.basename(path)[:-len('_original.png')]
        F, hw, pr, valid, hw_valid, cp = label(cv2.imread(path))
        cv2.imwrite(f'{out}/{name}_img.png', F)
        for tag, m in (('hw', hw), ('print', pr), ('valid', valid), ('hwvalid', hw_valid)):
            cv2.imwrite(f'{out}/{name}_{tag}.png', m.astype(np.uint8) * 255)
        viz = np.full_like(F, 255)
        viz[~valid] = (200, 200, 200)
        viz[pr] = (0, 0, 0)
        viz[hw] = (0, 0, 230)
        cv2.imwrite(f'{out}/{name}_viz.jpg', np.hstack([F, viz]))
        print(f'{name}: coloured pen {cp:.3f}  hw {hw.mean():.4f}  print {pr.mean():.4f}  valid {valid.mean():.2f}'
              + ('  (black pen: hw ignored)' if cp < 0.03 else ''))


if __name__ == '__main__':
    main()
