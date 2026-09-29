# Handwriting detection + rebuild-erase prototype (v2) on a real photo.
#  1. optical density per channel vs local paper; ink colour = OD_B / OD_R (density-independent)
#  2. regular text lines found by GEOMETRY only (shared baseline, equal heights, even spacing)
#  3. local print-colour reference from those lines (lens colour shift varies across the photo)
#  4. pen candidates = ink bluer than the LOCAL print reference
#  5. line vote: a regular line whose glyphs are mostly print-coloured is print (incl. its dots)
#  6. stroke vote + grow along faint pen ink, stop at print
#  7. erase: restore print continuations under strokes, inpaint the paper
import cv2, numpy as np, sys, time
S = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else 'real/IMG_0101_reading.png'
img = cv2.imread(src); H, W = img.shape[:2]; longside = max(H, W)
f = img.astype(np.float32)
k = max(5, int(longside / 450)) | 1

# ---- 1. optical density
sm = cv2.resize(f, None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA)
paperMax = cv2.dilate(cv2.medianBlur(sm.astype(np.uint8), 15), np.ones((9, 9), np.uint8)).astype(np.float32)
P = cv2.GaussianBlur(cv2.resize(paperMax, (W, H)), (0, 0), 8)
paperMed = cv2.GaussianBlur(cv2.resize(cv2.medianBlur(sm.astype(np.uint8), 21), (W, H)), (0, 0), 8)
OD = -np.log(np.clip((f + 1) / (P + 1), 1e-3, 1.0)); odm = OD.mean(-1)
ink = odm > 0.30
w = (ink & (odm < 1.6)).astype(np.float32)
cw = max(k, int(longside / 270)) | 1   # colour window: wide enough to average out sensor noise
sb = cv2.boxFilter(OD[..., 0] * w, -1, (cw, cw)); sr = cv2.boxFilter(OD[..., 2] * w, -1, (cw, cw)); sw = cv2.boxFilter(w, -1, (cw, cw))
ratio = np.where(sw > 0.05, sb / (sr + 1e-3), np.nan)

# ---- 2. regular lines by geometry
n, lbl, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
xs, ys, ws, hs, areas = st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4]
cx = xs + ws / 2; bottom = ys + hs
glyph = (areas >= k * 2) & (hs >= k) & (hs <= H * 0.03) & (ws <= hs * 4); glyph[0] = False
gid = np.where(glyph)[0]; order = gid[np.argsort(cx[gid])]
parent = {i: i for i in gid}
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
ocx = cx[order]
for idx, i in enumerate(order):
    j = idx + 1
    while j < len(order) and ocx[j] - cx[i] < 3.0 * hs[i]:
        o = order[j]
        if abs(bottom[o] - bottom[i]) < 0.18 * max(hs[i], hs[o]) and 0.6 < hs[o] / hs[i] < 1.67:
            parent[find(o)] = find(i)
        j += 1
groups = {}
for i in gid: groups.setdefault(find(i), []).append(i)
lines = []
for g in groups.values():
    if len(g) < 5: continue
    g = np.array(g); x = cx[g]; b = bottom[g].astype(float); h = hs[g].astype(float)
    A = np.stack([x, np.ones_like(x)], 1); coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = np.abs(A @ coef - b); hm = np.median(h)
    if np.percentile(resid, 70) < 0.08 * hm and np.percentile(np.abs(h - hm), 50) < 0.2 * hm:
        lines.append((g, coef, hm))

# ---- 3. local print-colour reference from regular-line pixels (robust: median per coarse cell)
regular = np.zeros(n, bool)
for g, _, _ in lines: regular[g] = True
regPix = regular[lbl] & ~np.isnan(ratio)
cell = max(64, longside // 12)
gh, gw = (H + cell - 1) // cell, (W + cell - 1) // cell
refGrid = np.full((gh, gw), np.nan, np.float32)
yy, xx = np.nonzero(regPix); rv = ratio[yy, xx]; cidx = (yy // cell) * gw + (xx // cell)
for c in np.unique(cidx):
    v = rv[cidx == c]
    if v.size >= 200: refGrid.flat[c] = np.median(v)
globalRef = float(np.nanmedian(refGrid)) if np.isfinite(refGrid).any() else 1.0
# fill empty cells from neighbours, then smooth
filled = np.where(np.isnan(refGrid), globalRef, refGrid)
for _ in range(3):
    blur = cv2.blur(np.where(np.isnan(refGrid), 0, refGrid), (3, 3)); cnt = cv2.blur(np.isfinite(refGrid).astype(np.float32), (3, 3))
    filled = np.where(np.isnan(refGrid), np.where(cnt > 0, blur / np.maximum(cnt, 1e-6), filled), refGrid)
refMap = cv2.resize(cv2.GaussianBlur(filled, (0, 0), 1.0), (W, H), interpolation=cv2.INTER_LINEAR)

# ---- 4. pen candidates
DELTA = 0.05
cand = ink & (np.nan_to_num(ratio, nan=9) < refMap - DELTA)
fracC = np.bincount(lbl.ravel(), weights=cand.ravel(), minlength=n) / np.maximum(areas, 1)

# ---- 5. line vote (colour decides whether a regular line is print; its dots/accents follow)
printComp = np.zeros(n, bool); lineBoxes = []
for g, coef, hm in lines:
    # regularity alone cannot tell (neat handwriting lines are just as straight — measured), the
    # line's colour decides: pen lines measured ≈0.98 pen-coloured, print lines ≤0.45 even at
    # the lens-shifted page edge
    if np.median(fracC[g]) < 0.6:
        printComp[g] = True
        lineBoxes.append((xs[g].min(), ys[g].min() - 0.6 * hm, (xs[g] + ws[g]).max(), bottom[g].max() + 0.3 * hm))
small = areas < k * k * 4; small[0] = False
# line consensus per character: a glyph whose neighbours on its (loose) text line are all print is
# print too, unless it is itself strongly pen-coloured — merged letters can break a line's
# regularity, but not its colour context
allG = np.where(glyph)[0]; oa = allG[np.argsort(cx[allG])]
par3 = {i: i for i in allG}
def find3(a):
    while par3[a] != a:
        par3[a] = par3[par3[a]]; a = par3[a]
    return a
ocx3 = cx[oa]
for idx, i in enumerate(oa):
    j = idx + 1
    while j < len(oa) and ocx3[j] - cx[i] < 2.5 * max(hs[i], k * 3):
        o = oa[j]
        if abs((ys[o] + hs[o] / 2) - (ys[i] + hs[i] / 2)) < 0.5 * max(hs[i], hs[o]) and 0.5 < hs[o] / hs[i] < 2.0:
            par3[find3(o)] = find3(i)
        j += 1
looseLines = {}
for i in oa: looseLines.setdefault(find3(i), []).append(i)
for g in looseLines.values():
    if len(g) < 4: continue
    g = sorted(g, key=lambda i: cx[i])
    for p_, i in enumerate(g):
        nb = g[max(0, p_ - 3):p_] + g[p_ + 1:p_ + 4]
        a = areas[nb].astype(float)
        nbPen = float((fracC[nb] * a).sum() / max(a.sum(), 1))
        if nbPen < 0.2 and fracC[i] < 0.85:
            printComp[i] = True
for (x0, y0, x1, y1) in lineBoxes:
    printComp |= small & (cx >= x0) & (cx <= x1) & (ys >= y0) & (ys + hs <= y1)

# ---- 6. stroke vote, grow along faint pen ink
compHw = (fracC >= 0.35) & ~printComp; compHw[0] = False
m2, l2, st2, _ = cv2.connectedComponentsWithStats((cand & ~printComp[lbl]).astype(np.uint8), 8)
big = st2[:, 4] >= (k * 2) ** 2; big[0] = False
hw = cand & ~printComp[lbl] & (compHw[lbl] | big[l2])
# small pen marks (accents, dots, commas) next to handwriting
near = cv2.dilate(hw.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * k + 1, 4 * k + 1))) > 0
hw |= (small & ~printComp & (fracC >= 0.15))[lbl] & near & ink
# handwritten lines: group the non-print strokes into loose lines; in a line that is mostly pen,
# strokes only slightly bluer than print are pen too (pen colour varies with pressure)
nonPrint = glyph & ~printComp
hwPixPerComp = np.bincount(lbl.ravel(), weights=hw.ravel(), minlength=n)
np_ids = np.where(nonPrint)[0]; o2 = np_ids[np.argsort(cx[np_ids])]
par2 = {i: i for i in np_ids}
def find2(a):
    while par2[a] != a:
        par2[a] = par2[par2[a]]; a = par2[a]
    return a
ocx2 = cx[o2]
for idx, i in enumerate(o2):
    j = idx + 1
    while j < len(o2) and ocx2[j] - cx[i] < 2.5 * max(hs[i], k * 3):
        o = o2[j]
        if abs((ys[o] + hs[o] / 2) - (ys[i] + hs[i] / 2)) < 0.6 * max(hs[i], hs[o]):
            par2[find2(o)] = find2(i)
        j += 1
hwLines = {}
for i in np_ids: hwLines.setdefault(find2(i), []).append(i)
compRatio = np.full(n, np.nan)
rr = np.nan_to_num(ratio, nan=0) - refMap
sumR = np.bincount(lbl.ravel(), weights=np.where(ink & ~np.isnan(ratio), rr, 0).ravel(), minlength=n)
cntR = np.bincount(lbl.ravel(), weights=(ink & ~np.isnan(ratio)).ravel(), minlength=n)
compRatio = sumR / np.maximum(cntR, 1)
for g in hwLines.values():
    g = np.array(g)
    if len(g) < 2: continue
    if hwPixPerComp[g].sum() >= 0.5 * areas[g].sum():
        weak = g[compRatio[g] < -0.015]
        mark = np.zeros(n, bool); mark[weak] = True
        hw |= mark[lbl] & ink
# neighbourhood consensus: a pen stroke's darkness/colour is uneven, so single pixels flip; an
# ink pixel takes the majority label of the ink around it (≈3 stroke widths), both ways — holes
# inside handwriting close, isolated pen-coloured specks inside print go back to print. Layout
# print (regular lines) keeps its label.
cwin = 3 * k + 1
inkF = ink.astype(np.float32)
layoutPrint = printComp[lbl]
for _ in range(2):
    share = cv2.boxFilter((hw & ink).astype(np.float32), -1, (cwin, cwin)) / np.maximum(cv2.boxFilter(inkF, -1, (cwin, cwin)), 1e-6)
    hw = (ink & (share >= 0.5) & ~layoutPrint) | (hw & ~ink)
# small detached pen fragments (a letter's lead-in curl, a colon, a dash, an accent): colour is
# unreliable on bits this thin, so their NEAREST neighbour decides — a small ink piece outside any
# regular print line that lies closer to handwriting than to any substantial print stroke (and
# within ≈8 stroke widths of it) is handwriting
tinyMax = (2 * k) ** 2      # dots, accents, short dashes
smallMax = (4 * k) ** 2     # lead-in curls, stroke ends
nonHwInk = np.bincount(lbl.ravel(), weights=(ink & ~hw).ravel(), minlength=n)
# print reference = any letter-sized-or-bigger ink that is not handwriting (printed letters
# included), so a printed letter next to a pen stroke still has print right beside it
refComp = (areas > tinyMax) & (nonHwInk >= 0.5 * areas); refComp[0] = False
printRef_ = (refComp | printComp)[lbl] & ink & ~hw
distHw = cv2.distanceTransform((~(hw & ink)).astype(np.uint8), cv2.DIST_L2, 3)
distPr = cv2.distanceTransform((~printRef_).astype(np.uint8), cv2.DIST_L2, 3)
# nearest approach (not the average): a lead-in curl touches its letter at one end
ordr = np.argsort(lbl.ravel()); lab_s = lbl.ravel()[ordr]; bounds = np.searchsorted(lab_s, np.arange(n + 1))
dHw = np.full(n, 1e9); dPr = np.full(n, 1e9)
dh = np.where(ink, distHw, 1e9).ravel()[ordr]; dp = np.where(ink, distPr, 1e9).ravel()[ordr]
ne = bounds[1:] > bounds[:-1]
dHw[ne] = np.minimum.reduceat(dh, bounds[:-1][ne]); dPr[ne] = np.minimum.reduceat(dp, bounds[:-1][ne])
tinyRule = (areas <= tinyMax) & (dHw < dPr) & (dHw <= 8 * k)
midRule = (areas <= smallMax) & (fracC >= 0.25) & (dHw <= 2 * k)
fragment = ~printComp & (tinyRule | midRule); fragment[0] = False
hw |= fragment[lbl] & ink
fragmentMask = fragment[lbl] & ink
# pictures (photos, logos): ink is dense over a wide area, text strokes never are
win = int(longside * 0.03) | 1
dens = cv2.boxFilter(ink.astype(np.float32), -1, (win, win))
dense = (dens > 0.5).astype(np.uint8)
# only sizeable dense regions (a photo), not a bold heading's letters
nd, ld, sd, _ = cv2.connectedComponentsWithStats(dense, 8)
bigDense = sd[:, cv2.CC_STAT_AREA] >= (2 * win) ** 2; bigDense[0] = False
picture = cv2.dilate(bigDense[ld].astype(np.uint8), np.ones((win, win), np.uint8)) > 0
hw &= ~picture
# protect print: confident print ink + faint pixels with print colour
confPrint = (ink & (np.nan_to_num(ratio, nan=0) > refMap - 0.02)) | printComp[lbl] | picture
faintW = ((odm > 0.06) & (odm <= 0.30)).astype(np.float32); kk = (2 * k + 1, 2 * k + 1)
fb = cv2.boxFilter(OD[..., 0] * faintW, -1, kk); fr = cv2.boxFilter(OD[..., 2] * faintW, -1, kk); fw = cv2.boxFilter(faintW, -1, kk)
ratioFaint = np.where(fw > 0.1, fb / (fr + 1e-3), 0)
confPrint |= (faintW > 0) & (ratioFaint > refMap - 0.03) & ~hw
confPrint = cv2.dilate(confPrint.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
allowed = (odm > 0.06) & ~confPrint
na, la = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=8)
seeded = np.zeros(na, bool); seeded[np.unique(la[hw & allowed])] = True; seeded[0] = False
hw |= seeded[la]
m3, l3, st3, _ = cv2.connectedComponentsWithStats(hw.astype(np.uint8), 8)
keep = st3[:, 4] >= max(12, k * k // 4); keep[0] = False; hw = keep[l3] | fragmentMask   # context-placed dots are not noise

# ---- 7. erase
t0 = time.time()
hw8 = hw.astype(np.uint8) * 255
cpNear = cv2.dilate(confPrint.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
faint = (odm > 0.04).astype(np.uint8) * 255
grown = cv2.dilate(hw8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
area = hw8 | (grown & faint & ~cpNear)
strongPrint = confPrint & ~hw & ink
penOD = np.median(odm[hw & ink]) if (hw & ink).any() else 1.0
nearPrint = cv2.dilate(strongPrint.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
under = (hw & nearPrint & (odm > penOD * 1.3)).astype(np.uint8) * 255
halo = cv2.dilate(area, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) & ~cpNear
area = area | halo
restore = area & under; paperA = area & ~restore
wgt = strongPrint.astype(np.float32); win = (61, 61)
num = cv2.boxFilter(f * wgt[..., None], -1, win); den = cv2.boxFilter(wgt, -1, win)
fallback = np.median(img[strongPrint], axis=0) if strongPrint.any() else np.array([40, 40, 40])
local = np.where(den[..., None] > 1e-3, num / np.maximum(den[..., None], 1e-6), fallback).astype(np.uint8)
dst = img.copy(); dst[restore > 0] = local[restore > 0]
dst = cv2.inpaint(dst, paperA, 3, cv2.INPAINT_TELEA)
print(f'k={k} lines={len(lines)} printLines={len(lineBoxes)} globalRef={globalRef:.3f} hwPx={int(hw.sum())} restorePx={int((restore>0).sum())} erase={time.time()-t0:.1f}s')

ov = img.copy(); ov[hw] = (0, 0, 255)
tag = sys.argv[3] if len(sys.argv) > 3 else 'v2'
cv2.imwrite(f'{S}/{tag}_mask_small.png', cv2.resize(ov, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA))
cv2.imwrite(f'{S}/{tag}_erased_small.png', cv2.resize(dst, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA))
cv2.imwrite(f'{S}/{tag}_erased.png', dst)
np.save(f'{S}/{tag}_hw.npy', hw); np.save(f'{S}/{tag}_ref.npy', refMap.astype(np.float32)); np.save(f'{S}/{tag}_print.npy', confPrint); np.save(f'{S}/{tag}_frag.npy', fragmentMask)
