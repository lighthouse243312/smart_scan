"""Post-erase restore prototype — runs AFTER the eraser (InkRefine.erase / MaskEraser):

1. print the eraser took: erased strokes inside a printed-text line band, print-coloured
2. rules (grid) the eraser took: long thin straight neutral runs of the original that
   continue a surviving rule, bridged across the pen strokes crossing them
3. pen the eraser left: bits of mostly-erased writing, and chromatic ink, off the print lines

usage: print_restore.py original erased outdir [clean [mask]]   (writes restored.png, restore_viz.png)
       mask: the app's handwriting mask PNG (alpha = handwriting, blue = print)
"""
import sys
import cv2
import numpy as np

orig_path, erased_path, out = sys.argv[1:4]
O = cv2.imread(orig_path)
E = cv2.imread(erased_path)
H, W = E.shape[:2]
if O.shape[:2] != (H, W):
    O = cv2.resize(O, (W, H), interpolation=cv2.INTER_AREA)
k = max(1, round(max(H, W) / 1280))          # stroke unit at this resolution


def paper(img, ksz=7, blur=31):
    return np.stack([cv2.medianBlur(cv2.dilate(img[..., c], np.ones((ksz, ksz), np.uint8)), blur) for c in range(3)], -1)


def normalize(img):
    bg = paper(img).astype(np.float32)
    return np.clip(img.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)


def lab(img):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return 255 - L[..., 0], L[..., 1] - 128, L[..., 2] - 128   # darkness, a, b


def u8(m):
    return m.astype(np.uint8)


def components(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8(mask), connectivity=8)
    return n, labels, stats


# darkness: the page the eraser worked on (app: the shadow-removed "clean" page, which is
# near-greyscale); colour: the white-balanced original
C = normalize(O)
N = cv2.imread(sys.argv[4]) if len(sys.argv) > 4 else C
dN, _, _ = lab(N)
_, aN, bN = lab(C)
dE, _, _ = lab(E)

# the app's own layers, when given: handwriting = alpha, print = blue (HandwritingMask.kt)
if len(sys.argv) > 5:
    M = cv2.imread(sys.argv[5], cv2.IMREAD_UNCHANGED)
    if M.shape[:2] != (H, W):
        M = cv2.resize(M, (W, H), interpolation=cv2.INTER_NEAREST)
    hw_mask, print_mask = M[..., 3] > 0, M[..., 0] > 127
else:
    hw_mask = print_mask = np.zeros((H, W), bool)
purple = aN - bN                       # pen axis: purple/red pen is a+ b-, print/rules neutral
INK = 40
inkN, inkE = dN > INK, dE > INK

# what the eraser took: ink in the original that is paper in the erased page
erased = inkN & (dE < 0.35 * dN)
# …and what it only faded to a grey ghost (restore candidates, not used for the pen model)
faded = inkN & (dE < 0.6 * dN)

# ── rule lines (grid) ────────────────────────────────────────────────────────
L_OPEN = 15 * k
h_lines = cv2.morphologyEx(u8(inkE), cv2.MORPH_OPEN, np.ones((1, L_OPEN), np.uint8))
v_lines = cv2.morphologyEx(u8(inkE), cv2.MORPH_OPEN, np.ones((L_OPEN, 1), np.uint8))
lines_E = (h_lines | v_lines) > 0
# ── printed text lines ──────────────────────────────────────────────────────
text_E = inkE & ~(cv2.dilate(u8(lines_E), np.ones((3, 3), np.uint8)) > 0)
n, labels, stats = components(text_E)
area = stats[:, cv2.CC_STAT_AREA]
hgt = stats[:, cv2.CC_STAT_HEIGHT]
glyph = np.zeros(n, bool)
wid = stats[:, cv2.CC_STAT_WIDTH]
glyph[1:] = (area[1:] >= 12 * k * k) & (hgt[1:] >= 6 * k) & (hgt[1:] <= 0.03 * H) & (wid[1:] <= 0.05 * W)
text_core = glyph[labels]
# median glyph height → band: a bit taller than a text line, extending sideways along it
gh = int(np.median(hgt[1:][glyph[1:]])) if glyph[1:].any() else 12 * k
# share of the original ink around each point that the eraser took: a stray pen bit sits in
# a mostly-erased neighbourhood, print beside an erased letter does not
win = (gh // 2) | 1      # tight: a printed word right beside an erased one stays intact
erased_ratio = cv2.blur(erased.astype(np.float32), (win, win)) / np.maximum(cv2.blur(inkN.astype(np.float32), (win, win)), 1e-3)
n_t, labels_t, stats_t = components(text_E)
ratio_t = np.zeros(n_t); np.add.at(ratio_t, labels_t, erased_ratio)
ratio_t /= np.maximum(stats_t[:, cv2.CC_STAT_AREA], 1)
intact = ratio_t < 0.5
glyph &= intact
text_core = glyph[labels_t]
# a printed line = several glyphs side by side on one row (dashes, dots count as members)
member = (stats_t[:, cv2.CC_STAT_AREA] >= 4 * k * k) & intact
member[0] = False
rows = cv2.morphologyEx(u8(member[labels_t]), cv2.MORPH_CLOSE, np.ones((1, 2 * gh), np.uint8))
n, labels, stats = components(rows)
line_ok = np.zeros(n, bool)
has_glyph = np.zeros(n, bool); has_glyph[np.unique(labels[text_core])] = True
line_ok[1:] = (stats[1:, cv2.CC_STAT_WIDTH] >= 6 * gh) & has_glyph[1:]
text_core &= line_ok[labels]
# surviving print glyphs (box + centre) to check that a candidate sits on their line, and short
# surviving bars (a fraction's line) that stacked digits hang on
g_ids = np.unique(labels_t[text_core]); g_ids = g_ids[g_ids > 0]
g_box = stats_t[g_ids, :4].astype(np.float32)          # x, y, w, h
g_cy = g_box[:, 1] + g_box[:, 3] / 2
bar_ok = np.zeros(n_t, bool)
bar_ok[1:] = (stats_t[1:, cv2.CC_STAT_WIDTH] <= 1.5 * gh) & (stats_t[1:, cv2.CC_STAT_WIDTH] >= 0.4 * gh) & \
             (stats_t[1:, cv2.CC_STAT_HEIGHT] <= max(2, 0.3 * gh))
b_ids = np.nonzero(bar_ok)[0]
b_box = stats_t[b_ids, :4].astype(np.float32)


def on_print_line(x, y, w, h):
    """Same line as its print neighbours: centre level with theirs, about their height."""
    near = (g_box[:, 0] < x + w + 4 * gh) & (g_box[:, 0] + g_box[:, 2] > x - 4 * gh) & \
           (np.abs(g_cy - (y + h / 2)) < gh)
    if near.sum() < 2:
        return False
    cy, gh_n = np.median(g_cy[near]), np.median(g_box[near, 3])
    base = np.median(g_box[near, 1] + g_box[near, 3])
    # centre or baseline level with theirs (capitals with accents are taller than the
    # lowercase around them; photos are slightly tilted)
    level = min(abs(y + h / 2 - cy), abs(y + h - base)) <= 0.4 * gh_n
    return level and 0.6 * gh_n <= h <= 1.6 * gh_n


def on_glyph_mark(x, y, w, h):
    """A diacritic: a small mark just above or below a surviving print glyph it overlaps."""
    if w > 0.6 * gh or h > 0.6 * gh:
        return False
    ov = (g_box[:, 0] < x + w) & (g_box[:, 0] + g_box[:, 2] > x)
    above = (g_box[:, 1] - (y + h) >= -1) & (g_box[:, 1] - (y + h) <= 0.6 * gh)
    below = (y - (g_box[:, 1] + g_box[:, 3]) >= -1) & (y - (g_box[:, 1] + g_box[:, 3]) <= 0.4 * gh)
    return bool((ov & (above | below)).any())


def on_fraction_bar(x, y, w, h):
    """Directly above or below a short surviving bar it overlaps horizontally."""
    if not len(b_box):
        return False
    ov = (b_box[:, 0] < x + w) & (b_box[:, 0] + b_box[:, 2] > x)
    above = (b_box[:, 1] - (y + h) >= -1) & (b_box[:, 1] - (y + h) <= 0.5 * gh)
    below = (y - (b_box[:, 1] + b_box[:, 3]) >= -1) & (y - (b_box[:, 1] + b_box[:, 3]) <= 0.5 * gh)
    return bool((ov & (above | below)).any())


band_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4 * gh, gh | 1))
band = cv2.dilate(u8(text_core), band_kernel) > 0
# a glyph's height above/below the line: stacked fraction digits, sub/superscripts
tall_band = cv2.dilate(u8(text_core), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * gh, 3 * gh | 1))) > 0

# colour models — pen from what was erased, print from surviving glyphs
pen_c = np.median(purple[erased & (dN > 70)])
print_c = np.median(purple[text_core & (dN > 70)])
split = (pen_c + print_c) / 2
margin = 0.25 * abs(pen_c - print_c)
separable = abs(pen_c - print_c) >= 4      # else colour can't tell pen from print
print(f'pen {pen_c:.1f} print {print_c:.1f} split {split:.1f} glyph h {gh}')

# ── rebuild rules the eraser took: long, straight, thin, neutral runs in the original
# (closing first jumps the pen strokes crossing them), that continue a surviving rule ──
LINE_INK = 18          # printed rules are often fainter than the ink threshold
neutralN = (dN > LINE_INK) & (purple < split)
# only where the eraser worked (its strokes plus the halo it clears around them) — the
# processed page elsewhere differs from the original by contrast alone
footprint = cv2.dilate(u8(erased), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7 * k, 7 * k))) > 0
missing = (dN > LINE_INK) & (dE < 0.5 * dN) & footprint
touch = cv2.dilate(u8(lines_E), np.ones((3, 3), np.uint8)) > 0
rule_gap = np.zeros((H, W), bool)
for close_k, open_k in [((1, 7 * k), (1, 31 * k)), ((7 * k, 1), (31 * k, 1))]:
    run = cv2.morphologyEx(u8(neutralN), cv2.MORPH_CLOSE, np.ones(close_k, np.uint8))
    run = cv2.morphologyEx(run, cv2.MORPH_OPEN, np.ones(open_k, np.uint8))
    n, labels, stats = components(run)
    for i in range(1, n):
        m = labels == i
        thick = stats[i, cv2.CC_STAT_AREA] / max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        # a real rule mostly survived the eraser and was only cut; pen digits' straight
        # strokes rebuilt from the original would be almost entirely gone
        survived = (m & lines_E).sum() / max(m.sum(), 1)
        if thick <= 3 * k and survived >= 0.3:
            rule_gap |= m & missing
rule_gap &= ~lines_E

# per-pixel colour distance from the print colour (and how far real print strays)
a_p0, b_p0 = np.median(aN[text_core & (dN > 70)]), np.median(bN[text_core & (dN > 70)])
cd_px = np.hypot(aN - a_p0, bN - b_p0)
cd_print_t = np.percentile(cd_px[text_core & (dN > 70)], 90)

# ── restore print: erased strokes in a print band, print-coloured, by component ──
n, labels, stats = components(faded & ~rule_gap)
restore_print = np.zeros((H, W), bool)
for _ in range(3):           # grow along the line: a restored glyph extends the band
    for i in range(1, n):
        m = labels == i
        x, y, w, h = stats[i, :4]
        print_coloured = np.median(purple[m]) < print_c + margin and np.median(cd_px[m]) <= cd_print_t
        if separable and not print_coloured and np.median(purple[m]) >= split:
            continue                                   # pen-coloured
        in_band = band[m].mean() > 0.6
        glyph_sized = h <= 1.5 * gh and w <= 1.5 * gh
        stacked = glyph_sized and tall_band[m].mean() > 0.8 and \
            (on_fraction_bar(x, y, w, h) or on_glyph_mark(x, y, w, h))
        if not (in_band or stacked):
            continue
        # pixels the eraser took beyond the handwriting mask (its halo) are restorable on the
        # band alone. Inside the mask the model called them handwriting — and a dark pen can
        # be print-coloured — so it takes the model's print layer, or print geometry: level
        # with the print beside it at its height, or a digit stacked on a fraction bar
        inside = hw_mask[m].mean() > 0.5
        if not inside or print_mask[m].mean() > 0.5 or \
                (print_coloured and (stacked or on_print_line(x, y, w, h))):
            restore_print |= m
    band |= cv2.dilate(u8(restore_print), band_kernel) > 0

# ── leftover pen ─────────────────────────────────────────────────────────────
# (a) what survives of erased writing: ink off the print lines in a neighbourhood the
#     eraser mostly cleared (same signal that keeps such bits out of the print band)
# (b) any chromatic ink off the print lines — colour distance from the print colour, so a
#     red teacher's mark counts as well as the purple pen
a_p, b_p = np.median(aN[text_core & (dN > 70)]), np.median(bN[text_core & (dN > 70)])
cdist = np.hypot(aN - a_p, bN - b_p)
cd_s = cv2.blur(np.where(inkN, cdist, 0), (3, 3)) / np.maximum(cv2.blur(inkN.astype(np.float32), (3, 3)), 1e-3)
pen_cd = np.median(cd_s[erased & (dN > 70)])
print_cd = np.percentile(cd_s[text_core & (dN > 70)], 90)
chroma_t = (pen_cd + print_cd) / 2
print(f'chroma: pen {pen_cd:.1f} print p90 {print_cd:.1f} threshold {chroma_t:.1f}')
keep = band | restore_print | rule_gap | (print_mask & ~hw_mask)
n, labels, stats = components(text_E)
ratio = np.zeros(n); np.add.at(ratio, labels, erased_ratio)
ratio /= np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
ratio[0] = 0
# leftovers are always beside the erased writing; rules keep their colour fringes
beside = cv2.dilate(u8(erased), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gh | 1, 2 * gh | 1))) > 0
rules_all = cv2.dilate(u8(lines_E | rule_gap), np.ones((3, 3), np.uint8)) > 0
residue = (ratio[labels] > 0.5) & ~keep & beside
chroma_ink = inkE & (cd_s > chroma_t) & beside & ~rules_all
if separable:
    residue |= (cv2.dilate(u8(chroma_ink), np.ones((3, 3), np.uint8)) > 0) & inkE & ~keep & ~rules_all

# ── compose ─────────────────────────────────────────────────────────────────
R = E.copy()
R[restore_print] = N[restore_print]
rule_col = np.median(N[lines_E & (dN > 25)], axis=0)
R[rule_gap] = np.where(purple[rule_gap, None] < split, N[rule_gap], rule_col)
paperE = paper(E, 9, 21)
R[residue] = paperE[residue]

cv2.imwrite(f'{out}/restored.png', R)
viz = E.copy()
viz[restore_print] = (0, 200, 0)
viz[rule_gap] = (255, 140, 0)
viz[residue] = (0, 0, 255)
cv2.imwrite(f'{out}/restore_viz.png', viz)
print('erased', erased.sum(), 'print', restore_print.sum(), 'rule gaps', rule_gap.sum(), 'residue', residue.sum())
