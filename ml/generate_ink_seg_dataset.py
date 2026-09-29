"""Synthetic training data for the 2-channel ink segmenter (ink_segmenter_v2).

Every sample is a synthetic scanned page rendered at inference scale (longer side = 1536 px,
the app resizes each page to that before tiling), with two pixel-aligned binary masks:

  channel 0  print  - printed ink present   (taken from the CLEAN printed document, so it also
                      covers printed ink that is hidden under handwriting)
  channel 1  hw     - handwriting ink present (taken from the handwriting layer alone)

Pipeline per page
  1. paper (tint + fibre texture)
  2. printed layer(s): documents / worksheets / forms / tables / calendars / receipts rendered
     with system fonts (EN + VI with diacritics), ruled lines, checkboxes, logo blocks...
     Each print colour is an alpha layer; print_mask = combined print alpha > 0.5.
  3. handwriting layer(s): EMNIST glyphs composed into words, macOS script fonts with per-glyph
     jitter + elastic distortion, procedural cursive trajectories, and pen marks (circles,
     ticks, crosses, underlines, strike-throughs, scribbles, arrows, signatures, boxes).
     hw_mask = stroke coverage > 0.3. A large share deliberately crosses printed text.
  4. composite with a MULTIPLY blend (ink over ink is darker than either), then degrade
     (geometry applied to image and masks alike; photometric only to the image).
  5. cut into 256x256 tiles (grid with 32 px overlap like inference, plus random crops),
     keeping most tiles with handwriting and ~15% without.

Output (in ml/): inkseg_{train,val,test}.npz with
  images uint8 (N,256,256,3) RGB,  masks uint8 (N,256,256) bit0 = print, bit1 = handwriting
and inkseg_fullpages_test.npz with a few whole test pages (images/masks lists, padded).

Usage: venv/bin/python generate_ink_seg_dataset.py [--pages-train 520 --pages-val 40 --pages-test 60]
"""
from __future__ import annotations

import argparse
import gzip
import math
import os
import random
import string
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ML_DIR = os.path.dirname(os.path.abspath(__file__))
EMNIST_DIR = None
for root, dirs, files in os.walk(os.path.join(ML_DIR, "tfds_data", "downloads", "extracted")):
    if "emnist-byclass-test-images-idx3-ubyte.gz" in files:
        EMNIST_DIR = root
        break

TAG = os.environ.get("INKSEG_TAG", "")  # optional dataset-name tag (for side-by-side experiments)
TILE = 256
OVERLAP = 32
STRIDE = TILE - OVERLAP
LONG_SIDE = 1536

# ----------------------------------------------------------------------------------------------
# fonts
# ----------------------------------------------------------------------------------------------
SYS = "/System/Library/Fonts/"
SUP = "/System/Library/Fonts/Supplemental/"

PRINT_FONT_FILES = [
    SUP + f for f in [
        "Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf", "Arial Narrow.ttf",
        "Arial Narrow Bold.ttf", "Arial Black.ttf", "Arial Rounded Bold.ttf", "Times New Roman.ttf",
        "Times New Roman Bold.ttf", "Times New Roman Italic.ttf", "Georgia.ttf", "Georgia Bold.ttf",
        "Georgia Italic.ttf", "Verdana.ttf", "Verdana Bold.ttf", "Verdana Italic.ttf", "Tahoma.ttf",
        "Tahoma Bold.ttf", "Trebuchet MS.ttf", "Trebuchet MS Bold.ttf", "Courier New.ttf",
        "Courier New Bold.ttf", "Andale Mono.ttf", "Impact.ttf", "Microsoft Sans Serif.ttf",
        "DIN Alternate Bold.ttf", "DIN Condensed Bold.ttf", "Comic Sans MS.ttf", "BigCaslon.ttf",
        "STIXTwoText.ttf", "Futura.ttc", "GillSans.ttc", "Baskerville.ttc", "Cochin.ttc", "Charter.ttc",
        "Didot.ttc", "Hoefler Text.ttc", "Iowan Old Style.ttc", "PTSans.ttc", "PTSerif.ttc", "PTMono.ttc",
        "Seravek.ttc", "Athelas.ttc", "Rockwell.ttc", "AmericanTypewriter.ttc", "Bodoni 72.ttc",
        "Copperplate.ttc", "Kailasa.ttc", "Galvji.ttc",
    ]
] + [
    SYS + f for f in [
        "Helvetica.ttc", "HelveticaNeue.ttc", "Times.ttc", "Courier.ttc", "Palatino.ttc", "Optima.ttc",
        "Avenir.ttc", "Avenir Next.ttc", "Avenir Next Condensed.ttc", "Menlo.ttc", "Monaco.ttf",
        "Geneva.ttf", "NewYork.ttf", "SFNS.ttf", "SFNSMono.ttf", "SFCompact.ttf", "LucidaGrande.ttc",
        "SFNSRounded.ttf",
    ]
] + ["/Library/Fonts/Arial Unicode.ttf"]
MONO_HINTS = ("Courier", "Mono", "Menlo", "Monaco", "Andale")

HW_FONT_FILES = [
    SUP + f for f in [
        "Bradley Hand Bold.ttf", "Chalkboard.ttc", "ChalkboardSE.ttc", "SnellRoundhand.ttc",
        "Apple Chancery.ttf", "Savoye LET.ttc", "Brush Script.ttf", "SignPainter.ttc", "Zapfino.ttf",
        "Trattatello.ttf",
    ]
] + [SYS + "Noteworthy.ttc", SYS + "MarkerFelt.ttc"]

VI_TEST = "ệươđắỹờẫổ"


def _faces(path, max_faces=12):
    out = []
    for i in range(max_faces):
        try:
            ImageFont.truetype(path, 20, index=i)
            out.append((path, i))
        except OSError:
            break
    return out


def _supports(path, index, chars):
    try:
        f = ImageFont.truetype(path, 24, index=index)
    except OSError:
        return False
    try:
        notdef = bytes(f.getmask(chr(0x378)))
        for c in chars:
            m = f.getmask(c)
            if m.getbbox() is None or bytes(m) == notdef:
                return False
    except OSError:
        return False
    return True


_FONT_CACHE: dict = {}


def font(face, size):
    key = (face, int(size))
    f = _FONT_CACHE.get(key)
    if f is None:
        f = ImageFont.truetype(face[0], max(6, int(size)), index=face[1])
        if len(_FONT_CACHE) > 3000:
            _FONT_CACHE.clear()
        _FONT_CACHE[key] = f
    return f


PRINT_FACES: list = []
PRINT_FACES_VI: list = []
MONO_FACES: list = []
HW_FACES: list = []
HW_FACES_VI: list = []


def init_fonts():
    global PRINT_FACES, PRINT_FACES_VI, MONO_FACES, HW_FACES, HW_FACES_VI
    for p in PRINT_FONT_FILES:
        if not os.path.exists(p):
            continue
        faces = _faces(p)
        # .ttc families can have many faces (weights); keep up to 6 so big families don't dominate
        if len(faces) > 6:
            faces = random.Random(p).sample(faces, 6)
        for fc in faces:
            if not _supports(*fc, "Aa0"):
                continue
            PRINT_FACES.append(fc)
            if any(h in p for h in MONO_HINTS):
                MONO_FACES.append(fc)
            if _supports(*fc, VI_TEST):
                PRINT_FACES_VI.append(fc)
    for p in HW_FONT_FILES:
        if not os.path.exists(p):
            continue
        for fc in _faces(p, 4):
            if _supports(*fc, "Aa0"):
                HW_FACES.append(fc)
                if _supports(*fc, VI_TEST):
                    HW_FACES_VI.append(fc)


# ----------------------------------------------------------------------------------------------
# text content
# ----------------------------------------------------------------------------------------------
EN_WORDS = (
    "the of and to in is you that it he was for on are as with his they at be this have from or one "
    "had by word but not what all were we when your can said there use each which she do how their if "
    "will up other about out many then them these so some her would make like him into time has look two "
    "more write go see number no way could people my than first water been call who oil its now find long "
    "down day did get come made may part answer date name school book page line total sum circle underline "
    "complete match true false yes student teacher class homework question exercise read following choose "
    "correct sentence paragraph example note important please sign here customer invoice amount price "
    "quantity item description payment receipt order account balance address phone email city country "
    "report summary meeting project schedule monday tuesday wednesday thursday friday saturday sunday "
    "january february march april may june july august september october november december week month "
    "year score grade level unit lesson chapter section figure table result method data analysis value "
    "average minimum maximum approved signature office manager department company limited service contract "
    "terms conditions policy information required optional checked reviewed science history english math"
).split()

VI_WORDS = (
    "của và những người không được trong đã có một cho này với các là để học sinh trường bài tập số câu "
    "hỏi trả lời đúng sai họ tên lớp ngày tháng năm điểm nhận xét giáo viên phụ huynh địa chỉ điện thoại "
    "tổng cộng thành tiền hóa đơn khách hàng sản phẩm giá lượng Việt Nam Hà Nội tiếng toán văn khoa đề "
    "kiểm tra phần chọn đáp án viết đọc hiểu nghĩa thứ hai ba tư sáu bảy chủ nhật mười trăm nghìn đồng ký "
    "xác nhận công ty cổ phần giám đốc hợp đồng thanh toán ngân hàng tài khoản mã số thuế ghi chú kết quả "
    "bảng dưới đây hãy điền vào chỗ trống khoanh tròn chữ cái đứng trước câu trả lời em quê hương đất nước "
    "mùa xuân buổi sáng thời khóa biểu lịch làm việc cuộc họp báo cáo tuần đơn vị tính số lượng chiết khấu"
).split()


def rand_number(rng):
    r = rng.random()
    if r < 0.3:
        return str(rng.randint(0, 999))
    if r < 0.45:
        return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2030)}"
    if r < 0.6:
        return f"${rng.randint(1, 999)}.{rng.randint(0, 99):02d}"
    if r < 0.72:
        return f"{rng.randint(1, 999)}.{rng.randint(0, 999):03d}đ"
    if r < 0.82:
        return f"{rng.randint(0, 100)}%"
    if r < 0.9:
        return f"0{rng.randint(900000000, 999999999)}"
    return f"{rng.randint(1, 99)}.{rng.randint(0, 9)}"


def rand_word(rng, vi=False):
    r = rng.random()
    if r < 0.12:
        return rand_number(rng)
    w = rng.choice(VI_WORDS if vi else EN_WORDS)
    s = rng.random()
    if s < 0.08:
        w = w.upper()
    elif s < 0.25:
        w = w.capitalize()
    if rng.random() < 0.08:
        w += rng.choice(",.;:")
    return w


def rand_words(rng, n, vi=False):
    return [rand_word(rng, vi) for _ in range(n)]


# ----------------------------------------------------------------------------------------------
# EMNIST glyph bank
# ----------------------------------------------------------------------------------------------
def _read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[2:3], "big")
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


EMNIST = None  # dict char -> list of tight-cropped glyph arrays (uint8 coverage)
EMNIST_SPLIT_AT = 100_000


def init_emnist(split):
    """byclass *test* file (116k glyphs, 62 classes digits+upper+lower). Samples [0,100k) are
    used for train/val pages, [100k, end) for test pages so test handwriting glyphs are unseen."""
    global EMNIST
    imgs = _read_idx(os.path.join(EMNIST_DIR, "emnist-byclass-test-images-idx3-ubyte.gz"))
    labels = _read_idx(os.path.join(EMNIST_DIR, "emnist-byclass-test-labels-idx1-ubyte.gz"))
    mapping = {}
    with open(os.path.join(EMNIST_DIR, "emnist-byclass-mapping.txt")) as f:
        for line in f:
            a, b = line.split()
            mapping[int(a)] = chr(int(b))
    sl = slice(0, EMNIST_SPLIT_AT) if split != "test" else slice(EMNIST_SPLIT_AT, None)
    imgs, labels = imgs[sl], labels[sl]
    bank: dict = {}
    for img, lab in zip(imgs, labels):
        g = img.T  # EMNIST bitmaps are transposed
        ys, xs = np.where(g > 40)
        if len(ys) == 0:
            continue
        g = g[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
        bank.setdefault(mapping[int(lab)], []).append(g)
    EMNIST = bank


# ----------------------------------------------------------------------------------------------
# stroke rendering helpers (all hand-drawn ink is rendered to float coverage patches)
# ----------------------------------------------------------------------------------------------
def catmull_rom(pts, samples_per_seg=10):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 3:
        if len(pts) == 2:
            t = np.linspace(0, 1, samples_per_seg * 2)[:, None]
            return pts[0] * (1 - t) + pts[1] * t
        return pts
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(len(p) - 3):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        seg_len = np.linalg.norm(p2 - p1)
        n = max(2, int(samples_per_seg * max(1.0, seg_len / 8)))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2][None])
    return np.vstack(out)


class StrokeCanvas:
    """Float coverage canvas for hand-drawn ink, rendered at SS x supersampling then box-reduced."""

    SS = 2

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.a = np.zeros((h * self.SS, w * self.SS), np.float32)

    def polyline(self, pts, width, rng, pressure=True):
        pts = np.asarray(pts, np.float32)
        if len(pts) < 2:
            return
        s = self.SS
        n = len(pts)
        # pressure: smooth width modulation along the stroke + taper at the ends
        if pressure:
            k = rng.uniform(0.5, 3.0)
            ph = rng.uniform(0, 6.28)
            t = np.linspace(0, 1, n)
            prof = 0.8 + 0.25 * np.sin(k * 6.28 * t + ph)
            taper = np.clip(np.minimum(t, 1 - t) / 0.08, 0.45, 1.0)
            prof = prof * taper
        else:
            prof = np.ones(n)
        P = (pts * s).astype(np.int32)
        for i in range(n - 1):
            th = max(1, int(round(width * s * prof[i])))
            cv2.line(self.a, tuple(P[i]), tuple(P[i + 1]), 1.0, th, cv2.LINE_AA)

    def dot(self, x, y, r):
        cv2.circle(self.a, (int(x * self.SS), int(y * self.SS)), max(1, int(r * self.SS)), 1.0, -1, cv2.LINE_AA)

    def paste(self, cov, x, y):
        """paste a 1x coverage patch (float 0..1) at (x, y) top-left, max-combine."""
        s = self.SS
        ph, pw = cov.shape
        big = cv2.resize(cov, (pw * s, ph * s), interpolation=cv2.INTER_LINEAR)
        x0, y0 = int(x * s), int(y * s)
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(self.a.shape[1], x0 + big.shape[1]), min(self.a.shape[0], y0 + big.shape[0])
        if xb <= xa or yb <= ya:
            return
        sub = big[ya - y0: yb - y0, xa - x0: xb - x0]
        np.maximum(self.a[ya:yb, xa:xb], sub, out=self.a[ya:yb, xa:xb])

    def result(self):
        return np.clip(cv2.resize(self.a, (self.w, self.h), interpolation=cv2.INTER_AREA), 0, 1)


def elastic(cov, rng, alpha=None, sigma=None):
    h, w = cov.shape
    alpha = alpha if alpha is not None else rng.uniform(1.0, 3.0)
    sigma = sigma if sigma is not None else rng.uniform(5, 10)
    dx = cv2.GaussianBlur((np.random.default_rng(rng.randrange(1 << 30)).random((h, w), np.float32) * 2 - 1), (0, 0), sigma)
    dy = cv2.GaussianBlur((np.random.default_rng(rng.randrange(1 << 30)).random((h, w), np.float32) * 2 - 1), (0, 0), sigma)
    m = max(1e-6, np.abs(dx).max(), np.abs(dy).max())
    dx, dy = dx / m * alpha, dy / m * alpha
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(cov, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=0)


def shear_patch(cov, shear):
    h, w = cov.shape
    extra = int(abs(shear) * h) + 2
    M = np.float32([[1, -shear, extra if shear > 0 else 2], [0, 1, 0]])
    return cv2.warpAffine(cov, M, (w + extra + 2, h), flags=cv2.INTER_LINEAR, borderValue=0)


def rotate_patch(cov, deg):
    h, w = cov.shape
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(cov, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)


def thicken(cov, rng, amount):
    """amount in px (float, may be negative = thinner). Works at 2x."""
    if abs(amount) < 0.25:
        return cov
    h, w = cov.shape
    big = cv2.resize(cov, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    k = max(1, int(round(abs(amount) * 2)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 1, k + 1))
    big = cv2.dilate(big, ker) if amount > 0 else cv2.erode(big, ker)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


# ---- handwriting word sources ---------------------------------------------------------------
XHEIGHT_LOWER = set("acemnorsuvwxz")
DESC_LOWER = set("gjpqy")


def restroke(cov, target_w, rng):
    """Normalise stroke width of a coverage patch to ~target_w px (EMNIST strokes are thick,
    ~15% of the glyph height; real ballpoint is ~6-10%). Distance-transform based thinning /
    thickening at 2x resolution."""
    h, w = cov.shape
    big = cv2.resize(cov, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    m = (big > 0.5).astype(np.uint8)
    if m.sum() < 10:
        return cov
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    ridge = dist[(dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-3)]
    half = float(np.median(ridge)) if len(ridge) else float(dist.max())
    tgt_half = max(1.0, target_w)  # at 2x scale, half width in 2x px == width in 1x px
    if tgt_half < half:
        keep = dist > (half - tgt_half)
        out = keep.astype(np.float32)
    else:
        k = int(round((tgt_half - half) * 2)) + 1
        out = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    out = cv2.GaussianBlur(out, (0, 0), 0.7)
    return np.clip(cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA), 0, 1)


VI_BASE_EXTRA = {"đ": ("d", ["bar"]), "Đ": ("D", ["bar"])}
COMBINING = {"\u0301": "acute", "\u0300": "grave", "\u0309": "hook", "\u0303": "tilde", "\u0323": "dotbelow",
             "\u0302": "circ", "\u0306": "breve", "\u031b": "horn"}


def _decompose(ch):
    import unicodedata
    if ch in VI_BASE_EXTRA:
        return VI_BASE_EXTRA[ch]
    d = unicodedata.normalize("NFD", ch)
    return d[0], [COMBINING[c] for c in d[1:] if c in COMBINING]


def _draw_diacritics(canvas, marks, box, cap_h, t, rng):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2 + rng.uniform(-0.1, 0.1) * cap_h
    top = y0 - cap_h * 0.12
    s = cap_h * rng.uniform(0.18, 0.28)
    J = lambda: rng.uniform(-0.05, 0.05) * cap_h
    def ln(pts):
        pts = np.array(pts, np.float32)
        cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, 1.0, t, cv2.LINE_AA)
    for mk in marks:
        if mk == "circ":
            ln([(cx - s, top), (cx, top - s + J()), (cx + s, top)])
            top -= s * 1.1
        elif mk == "breve":
            ln([(cx - s, top - s), (cx - s * 0.3, top), (cx + s * 0.3, top), (cx + s, top - s)])
            top -= s * 1.1
    for mk in marks:
        if mk == "acute":
            ln([(cx - s * 0.2 + s * 0.6, top - s * 1.1 + J()), (cx + s * 0.1, top)])
        elif mk == "grave":
            ln([(cx - s * 0.6, top - s * 1.1 + J()), (cx, top)])
        elif mk == "hook":
            ln([(cx - s * 0.4, top - s), (cx + s * 0.1, top - s * 1.2), (cx + s * 0.3, top - s * 0.7), (cx, top - s * 0.3), (cx, top)])
        elif mk == "tilde":
            ln([(cx - s, top - s * 0.3), (cx - s * 0.4, top - s * 0.8), (cx + s * 0.3, top - s * 0.3), (cx + s, top - s * 0.8)])
        elif mk == "dotbelow":
            cv2.circle(canvas, (int(cx), int(y1 + cap_h * 0.18)), max(1, int(t * 0.8)), 1.0, -1, cv2.LINE_AA)
        elif mk == "horn":
            ln([(x1 - s * 0.3, y0 + s * 0.5), (x1 + s * 0.3, y0 - s * 0.1), (x1 + s * 0.2, y0 - s * 0.6)])
        elif mk == "bar":
            yb = y0 + (y1 - y0) * rng.uniform(0.25, 0.4)
            ln([(x0 + (x1 - x0) * 0.3, yb), (x1 + s * 0.3, yb + J())])


def emnist_word(text, cap_h, rng, stroke_w=None):
    """Compose EMNIST glyphs into a word patch (float coverage). Returns (patch, baseline_y).
    Vietnamese diacritics are drawn as extra pen strokes on the base glyph."""
    cap_h = max(8, cap_h)
    pad = int(cap_h * 0.8) + 4
    glyphs = []
    marks_per = []
    for ch in text:
        if ch == " ":
            glyphs.append(None)
            marks_per.append([])
            continue
        ch, mk = _decompose(ch)
        marks_per.append(mk)
        key = ch
        if key not in EMNIST:
            if key.upper() in EMNIST:
                key = key.upper()
            elif key.lower() in EMNIST:
                key = key.lower()
            else:
                glyphs.append(("mark", ch))
                continue
        glyphs.append((key, rng.choice(EMNIST[key])))
    W = int(len(text) * cap_h * 1.1) + 2 * pad
    H = int(cap_h * 2.6) + 2 * pad
    canvas = np.zeros((H, W), np.float32)
    base = pad + int(cap_h * 1.7)
    boxes = []
    x = pad
    wobble_amp = rng.uniform(0, 0.12) * cap_h
    wobble_ph = rng.uniform(0, 6.28)
    drift = rng.uniform(-0.05, 0.05)
    spacing = rng.uniform(-0.12, 0.25)
    for i, g in enumerate(glyphs):
        if g is None:
            x += int(cap_h * rng.uniform(0.4, 0.8))
            continue
        key, bm = g
        if key == "mark":
            # simple punctuation drawn as a small blob / slash
            if bm in ".,":
                cv2.circle(canvas, (x + 2, base - 1), max(1, int(cap_h * 0.08)), 1.0, -1, cv2.LINE_AA)
                x += int(cap_h * 0.35)
            elif bm in "/-:+=%$":
                t = max(1, int(cap_h * 0.1))
                if bm == "/":
                    cv2.line(canvas, (x, base), (x + int(cap_h * 0.5), base - cap_h), 1.0, t, cv2.LINE_AA)
                else:
                    cv2.line(canvas, (x, base - cap_h // 2), (x + int(cap_h * 0.5), base - cap_h // 2), 1.0, t, cv2.LINE_AA)
                x += int(cap_h * 0.6)
            else:
                x += int(cap_h * 0.4)
            continue
        is_lower = key.islower()
        if is_lower and key in XHEIGHT_LOWER:
            th = cap_h * rng.uniform(0.55, 0.7)
            top_off = 0
        elif is_lower and key in DESC_LOWER:
            th = cap_h * rng.uniform(0.95, 1.1)
            top_off = th * 0.4  # extends below baseline
        else:
            th = cap_h * rng.uniform(0.92, 1.08)
            top_off = 0
        gh, gw = bm.shape
        sc = th / gh
        nw, nh = max(2, int(gw * sc * rng.uniform(0.85, 1.15))), max(2, int(th))
        gl = cv2.resize(bm.astype(np.float32) / 255.0, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            gl = rotate_patch(gl, rng.uniform(-8, 8))
        gh2, gw2 = gl.shape
        y_bot = base + top_off + wobble_amp * math.sin(wobble_ph + i * 0.9) + drift * (x - pad) + rng.uniform(-0.06, 0.06) * cap_h
        y0 = int(y_bot - gh2)
        y0 = min(max(0, y0), H - gh2)
        x0 = min(x, W - gw2)
        np.maximum(canvas[y0:y0 + gh2, x0:x0 + gw2], gl, out=canvas[y0:y0 + gh2, x0:x0 + gw2])
        boxes.append((i, (x0, y0, x0 + gw2, y0 + gh2)))
        x += int(gw2 * (1 + spacing) + cap_h * 0.05)
    canvas = canvas[:, : min(W, x + pad)]
    if stroke_w is not None:
        canvas = restroke(canvas, stroke_w, rng)
    t = max(1, int(round(stroke_w if stroke_w is not None else cap_h * 0.1)))
    for i, b in boxes:
        if marks_per[i]:
            _draw_diacritics(canvas, marks_per[i], b, cap_h, t, rng)
    return canvas, base


def font_word(text, cap_h, rng, vi=False):
    faces = HW_FACES_VI if (vi and HW_FACES_VI) else HW_FACES
    face = rng.choice(faces)
    size = cap_h / 0.62
    if "Zapfino" in face[0]:
        size *= 0.55
    if "Savoye" in face[0] or "Snell" in face[0]:
        size *= 1.15
    f = font(face, size)
    pad = int(size * 0.6) + 4
    try:
        tw = int(f.getlength(text))
    except Exception:
        tw = int(len(text) * size * 0.6)
    W, H = tw + 2 * pad + int(len(text) * size * 0.1), int(size * 2.2) + 2 * pad
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    base = pad + int(size * 1.1)
    if rng.random() < 0.55:
        # per-glyph jitter
        x = pad
        for i, ch in enumerate(text):
            s2 = size * rng.uniform(0.88, 1.12)
            f2 = font(face, s2)
            gim = Image.new("L", (int(s2 * 2) + 4, int(s2 * 2.4) + 4), 0)
            ImageDraw.Draw(gim).text((2, int(s2 * 0.3)), ch, font=f2, fill=255, anchor="ls" if False else None)
            gim = gim.rotate(rng.uniform(-7, 7), resample=Image.BILINEAR, expand=False)
            y = base - int(s2 * 1.0) + int(rng.uniform(-0.07, 0.07) * size)
            img.paste(255, (x, y), gim)
            x += int(f2.getlength(ch) * rng.uniform(0.9, 1.12))
    else:
        d.text((pad, base - int(size * 0.8)), text, font=f, fill=255)
    cov = np.asarray(img, np.float32) / 255.0
    cov = elastic(cov, rng, alpha=rng.uniform(0.8, 2.5), sigma=rng.uniform(4, 9))
    return cov, base


# cursive letter templates: (x, y) with x in [0,1] of letter width, y in x-heights (0 = baseline)
CURSIVE = {
    "e": [(0, .3), (.6, .65), (.45, 1.0), (.1, .6), (.35, .02), (1, .3)],
    "l": [(0, .3), (.55, 2.2), (.25, 2.0), (.3, 0), (1, .3)],
    "n": [(0, .1), (.15, 1), (.25, 0), (.45, .85), (.65, 1), (.75, 0), (1, .25)],
    "m": [(0, .1), (.1, 1), (.18, 0), (.35, .9), (.45, 1), (.52, 0), (.7, .9), (.8, 1), (.85, 0), (1, .25)],
    "u": [(0, 1), (.1, .1), (.35, 0), (.55, 1), (.6, .1), (1, .3)],
    "o": [(0, .6), (.35, 1), (.6, .5), (.35, 0), (.08, .5), (.35, 1), (1, .8)],
    "a": [(0, .3), (.55, .95), (.2, .8), (.1, .2), (.4, .05), (.6, .9), (.6, .1), (1, .3)],
    "g": [(0, .4), (.5, .95), (.15, .7), (.3, .1), (.6, .9), (.6, -1.2), (.2, -1.0), (1, .25)],
    "y": [(0, 1), (.1, .2), (.4, .1), (.6, 1), (.6, -1.2), (.25, -1.0), (1, .25)],
    "i": [(0, .2), (.4, 1), (.4, 0), (1, .3)],
    "t": [(0, .2), (.4, 1.8), (.4, 0), (1, .3)],
    "h": [(0, .2), (.5, 2.1), (.25, 1.9), (.2, 0), (.45, .9), (.7, .8), (.75, 0), (1, .3)],
    "r": [(0, .2), (.3, 1), (.55, .85), (.6, 0), (1, .3)],
    "s": [(0, .1), (.5, 1), (.3, .5), (.7, .2), (.35, 0), (1, .3)],
    "d": [(0, .3), (.55, .95), (.2, .8), (.1, .2), (.45, .05), (.65, 2.0), (.62, .1), (1, .3)],
    "c": [(0, .3), (.7, .95), (.25, .9), (.1, .3), (.45, 0), (1, .3)],
    "b": [(0, .3), (.45, 2.1), (.2, 1.9), (.2, .1), (.5, .05), (.7, .6), (.4, .7), (1, .6)],
}
CURSIVE_KEYS = list(CURSIVE.keys())


def cursive_word(n_letters, cap_h, width_px, rng, sc: StrokeCanvas | None = None):
    """Procedural pseudo-cursive word as a single pen trajectory. Returns (patch, baseline)."""
    xh = cap_h * rng.uniform(0.5, 0.65)
    lw = xh * rng.uniform(0.7, 1.1)
    slant = rng.uniform(-0.1, 0.45)
    pad = int(cap_h * 1.6) + 4
    W = int(n_letters * lw + 2 * pad)
    H = int(cap_h * 4 + 2 * pad)
    base = pad + cap_h * 2.2
    pts, dots, bars = [], [], []
    x = pad
    for _ in range(n_letters):
        k = rng.choice(CURSIVE_KEYS)
        wl = lw * rng.uniform(0.8, 1.25)
        for (u, v) in CURSIVE[k]:
            vv = v + rng.uniform(-0.08, 0.08)
            px = x + u * wl + slant * vv * xh + rng.uniform(-0.05, 0.05) * wl
            py = base - vv * xh
            pts.append((px, py))
        if k == "i":
            dots.append((x + 0.45 * wl + slant * 1.6 * xh, base - 1.6 * xh))
        if k == "t":
            bars.append(((x + 0.1 * wl + slant * 1.3 * xh, base - 1.3 * xh), (x + 0.8 * wl + slant * 1.3 * xh, base - 1.35 * xh)))
        x += wl
    c = StrokeCanvas(W, H)
    width = width_px if width_px else max(0.8, cap_h * rng.uniform(0.07, 0.13))
    c.polyline(catmull_rom(pts, 6), width, rng)
    for d in dots:
        c.dot(d[0], d[1], width * 0.8)
    for b in bars:
        c.polyline(np.array(b), width, rng, pressure=False)
    return c.result(), base


# ----------------------------------------------------------------------------------------------
# page model
# ----------------------------------------------------------------------------------------------
class Page:
    def __init__(self, w, h, rng):
        self.w, self.h, self.rng = w, h, rng
        self.print_layers: dict = {}   # color -> PIL L image
        self.word_boxes = []            # printed word/element boxes (x0,y0,x1,y1)
        self.line_boxes = []            # printed text line boxes
        self.slots = []                 # empty areas to write into (x0,y0,x1,y1, baseline_y)
        self.checkboxes = []            # (x0,y0,x1,y1)
        self.sign_lines = []
        self.mcq = []                   # boxes of option letters (to be circled / ticked)
        self.vi = rng.random() < 0.35
        self.dark = self._dark_color()
        self.accent = self._accent_color()

    def _dark_color(self):
        g = self.rng.randint(0, 70)
        if self.rng.random() < 0.2:  # navy / blue-ish black print (colour must not be a shortcut)
            return (g // 2, g // 2 + self.rng.randint(0, 8), g // 2 + self.rng.randint(8, 30))
        t = self.rng.randint(-8, 8)
        return (max(0, g + t), g, max(0, g - t))

    def _accent_color(self):
        return self.rng.choice([(20, 60, 160), (170, 25, 30), (10, 110, 60), (90, 30, 120), (0, 90, 140),
                                (200, 90, 0), (60, 60, 60), (30, 30, 90)])

    def draw(self, color=None) -> ImageDraw.ImageDraw:
        color = color or self.dark
        if color not in self.print_layers:
            self.print_layers[color] = Image.new("L", (self.w, self.h), 0)
        return ImageDraw.Draw(self.print_layers[color])

    def face(self, mono=False, bold=None):
        if mono and MONO_FACES:
            return self.rng.choice(MONO_FACES)
        faces = PRINT_FACES_VI if self.vi else PRINT_FACES
        return self.rng.choice(faces)

    # -- primitives ----------------------------------------------------------------------------
    def text_line(self, x, y, words, fc, size, color=None, max_x=None, record=True, justify=False):
        d = self.draw(color)
        f = font(fc, size)
        space = f.getlength(" ") * self.rng.uniform(0.9, 1.3)
        max_x = max_x or self.w - 10
        cx = x
        boxes = []
        for w in words:
            wl = f.getlength(w)
            if cx + wl > max_x:
                break
            d.text((cx, y), w, font=f, fill=255)
            bb = d.textbbox((cx, y), w, font=f)
            boxes.append(bb)
            cx += wl + space
        if record and boxes:
            self.word_boxes.extend(boxes)
            self.line_boxes.append((boxes[0][0], min(b[1] for b in boxes), boxes[-1][2], max(b[3] for b in boxes)))
        return cx, boxes

    def paragraph(self, x0, y, width, fc, size, n_lines, color=None, indent=0.0):
        f = font(fc, size)
        lh = size * self.rng.uniform(1.15, 1.7)
        for i in range(n_lines):
            if y + lh > self.h - 10:
                break
            n = 30
            words = rand_words(self.rng, n, self.vi)
            last = i == n_lines - 1
            mx = x0 + width * (self.rng.uniform(0.3, 0.9) if last else 1.0)
            self.text_line(x0 + (indent * size if i == 0 else 0), y, words, fc, size, color, max_x=mx)
            y += lh
        return y

    def hline(self, x0, x1, y, width=1, color=None, dotted=False):
        d = self.draw(color)
        if dotted:
            step = self.rng.randint(4, 8)
            for xx in range(int(x0), int(x1), step):
                d.line([(xx, y), (xx + max(1, step // 2 - 1), y)], fill=255, width=width)
        else:
            d.line([(x0, y), (x1, y)], fill=255, width=width)

    def rect(self, box, width=1, color=None, fill=None):
        d = self.draw(color)
        d.rectangle(box, outline=255, width=width, fill=fill)

    def blank_line(self, x0, x1, y, size):
        """fill-in line (underline / dotted); registered as a writing slot."""
        style = self.rng.random()
        if style < 0.6:
            self.hline(x0, x1, y, width=self.rng.choice([1, 1, 2]))
        elif style < 0.9:
            self.hline(x0, x1, y, width=self.rng.choice([1, 2]), dotted=True)
        self.slots.append((x0, y - size * 1.6, x1, y + size * 0.2, y - 2))

    # -- blocks (each returns new y) ------------------------------------------------------------
    def block_heading(self, x0, y, width, base):
        size = base * self.rng.uniform(1.3, 2.4)
        fc = self.face()
        col = self.accent if self.rng.random() < 0.35 else None
        words = rand_words(self.rng, self.rng.randint(2, 7), self.vi)
        if self.rng.random() < 0.4:
            words = [w.upper() for w in words]
        if self.rng.random() < 0.3:
            f = font(fc, size)
            tw = sum(f.getlength(w + " ") for w in words)
            x0 = x0 + max(0, (width - tw) / 2)
        self.text_line(x0, y, words, fc, size, col, max_x=x0 + width)
        y += size * 1.5
        if self.rng.random() < 0.4:  # section heading with a printed rule (CV / report style)
            self.hline(x0, x0 + width, y - size * 0.2, width=self.rng.randint(1, 3), color=col)
            y += size * 0.3
        return y

    def block_paragraph(self, x0, y, width, base):
        return self.paragraph(x0, y, width, self.body_face, base, self.rng.randint(2, 9), indent=self.rng.choice([0, 0, 2])) + base * 0.6

    def block_list(self, x0, y, width, base):
        n = self.rng.randint(2, 6)
        style = self.rng.choice(["num", "bullet", "dash", "letter"])
        f = font(self.body_face, base)
        for i in range(n):
            if y > self.h - base * 2:
                break
            lab = {"num": f"{i + 1}.", "bullet": "•", "dash": "-", "letter": f"{'abcdef'[i]})"}[style]
            self.text_line(x0, y, [lab], self.body_face, base)
            self.text_line(x0 + base * 1.8, y, rand_words(self.rng, 20, self.vi), self.body_face, base,
                           max_x=x0 + width * self.rng.uniform(0.5, 1.0))
            y += base * self.rng.uniform(1.3, 1.8)
        return y + base * 0.5

    def block_questions(self, x0, y, width, base):
        n = self.rng.randint(2, 5)
        for i in range(n):
            if y > self.h - base * 4:
                break
            q = [f"{self.rng.randint(1, 40)}." if self.rng.random() < 0.5 else f"Câu {i + 1}:" if self.vi else f"Q{i + 1}."]
            self.text_line(x0, y, q + rand_words(self.rng, 25, self.vi), self.body_face, base, max_x=x0 + width)
            y += base * 1.5
            kind = self.rng.random()
            if kind < 0.4:  # MCQ
                opts = ["A.", "B.", "C.", "D."]
                col_w = width / self.rng.choice([2, 4])
                cx, cy = x0 + base, y
                for j, o in enumerate(opts):
                    _, bxs = self.text_line(cx, cy, [o] + rand_words(self.rng, self.rng.randint(1, 3), self.vi),
                                            self.body_face, base, max_x=cx + col_w - base)
                    if bxs:
                        self.mcq.append(bxs[0])
                    cx += col_w
                    if cx + col_w > x0 + width + 1:
                        cx = x0 + base
                        cy += base * 1.5
                y = cy + base * 1.6
            elif kind < 0.7:  # answer lines
                for _ in range(self.rng.randint(1, 3)):
                    y += base * 1.4
                    self.blank_line(x0 + base, x0 + width, y, base)
                y += base * 0.9
            elif kind < 0.85:  # true/false boxes
                cx = x0 + base
                for lab in (["Đúng", "Sai"] if self.vi else ["True", "False"]):
                    s = int(base * 0.9)
                    self.rect((cx, y, cx + s, y + s), width=self.rng.choice([1, 2]))
                    self.checkboxes.append((cx, y, cx + s, y + s))
                    self.text_line(cx + s * 1.5, y - base * 0.1, [lab], self.body_face, base)
                    cx += base * 7
                y += base * 1.8
            else:  # arithmetic fill-in
                cx = x0 + base
                for _ in range(self.rng.randint(1, 3)):
                    a, b = self.rng.randint(1, 99), self.rng.randint(1, 99)
                    ex, _ = self.text_line(cx, y, [f"{a}", self.rng.choice("+-x"), f"{b}", "="], self.body_face, base)
                    self.blank_line(ex, ex + base * 4, y + base * 1.1, base)
                    cx = ex + base * 6
                    if cx > x0 + width - base * 8:
                        break
                y += base * 2.0
        return y

    def block_form(self, x0, y, width, base):
        labels_en = ["Name", "Date", "Class", "Address", "Phone", "Email", "Signature", "ID", "School", "Total", "Company"]
        labels_vi = ["Họ tên", "Ngày", "Lớp", "Địa chỉ", "Điện thoại", "Email", "Chữ ký", "Mã số", "Trường", "Tổng cộng"]
        labels = labels_vi if self.vi else labels_en
        n = self.rng.randint(2, 6)
        two_col = self.rng.random() < 0.4
        for i in range(n):
            if y > self.h - base * 3:
                break
            cols = 2 if two_col else 1
            cw = width / cols
            for c in range(cols):
                lab = self.rng.choice(labels) + ":"
                cx = x0 + c * cw
                ex, _ = self.text_line(cx, y, [lab], self.body_face, base)
                if self.rng.random() < 0.15:
                    # boxed field
                    self.rect((ex + base * 0.3, y - base * 0.3, cx + cw - base, y + base * 1.4))
                    self.slots.append((ex + base * 0.5, y - base * 0.2, cx + cw - base * 1.2, y + base * 1.3, y + base))
                else:
                    self.blank_line(ex + base * 0.3, cx + cw - base, y + base * 1.15, base)
                if "Chữ ký" in lab or "Signature" in lab:
                    self.sign_lines.append((ex, y - base * 1.5, cx + cw - base, y + base * 1.2))
            y += base * self.rng.uniform(1.9, 2.6)
        if self.rng.random() < 0.5:  # checkbox row
            cx = x0
            for _ in range(self.rng.randint(2, 5)):
                s = int(base * 0.9)
                self.rect((cx, y, cx + s, y + s), width=self.rng.choice([1, 2]))
                self.checkboxes.append((cx, y, cx + s, y + s))
                ex, _ = self.text_line(cx + s * 1.4, y - base * 0.1, rand_words(self.rng, self.rng.randint(1, 2), self.vi), self.body_face, base)
                cx = ex + base * 1.5
                if cx > x0 + width - base * 6:
                    break
            y += base * 2
        return y + base * 0.5

    def block_table(self, x0, y, width, base):
        ncol = self.rng.randint(2, 7)
        nrow = self.rng.randint(2, 12)
        rh = base * self.rng.uniform(1.6, 2.6)
        colw = np.array([self.rng.uniform(0.6, 2.0) for _ in range(ncol)])
        colw = colw / colw.sum() * width
        xs = [x0] + list(x0 + np.cumsum(colw))
        lw = self.rng.choice([1, 1, 2])
        col = self.accent if self.rng.random() < 0.2 else None
        maxrows = int((self.h - y - base) // rh)
        nrow = max(0, min(nrow, maxrows))
        if nrow < 2:
            return y
        ys = [y + i * rh for i in range(nrow + 1)]
        header_fill = self.rng.random() < 0.3
        if header_fill:
            # light shaded header: printed tint (alpha < 0.5 -> not "ink" in the mask)
            d = self.draw(self.accent)
            d.rectangle((xs[0], ys[0], xs[-1], ys[1]), fill=self.rng.randint(25, 90))
        full_grid = self.rng.random() < 0.7
        for yy in ys:
            self.hline(xs[0], xs[-1], yy, lw, col)
        if full_grid:
            d = self.draw(col)
            for xx in xs:
                d.line([(xx, ys[0]), (xx, ys[-1])], fill=255, width=lw)
        empty_p = self.rng.uniform(0.1, 0.6)
        for r in range(nrow):
            for c in range(ncol):
                cx0, cy0, cx1, cy1 = xs[c] + base * 0.3, ys[r], xs[c + 1] - base * 0.3, ys[r + 1]
                ty = cy0 + (rh - base * 1.2) / 2
                if r == 0:
                    self.text_line(cx0, ty, rand_words(self.rng, 3, self.vi), self.body_face, base * 0.95, max_x=cx1)
                elif self.rng.random() < empty_p:
                    self.slots.append((cx0, cy0 + 2, cx1, cy1 - 2, cy1 - rh * 0.25))
                else:
                    ws = [rand_number(self.rng)] if self.rng.random() < 0.5 else rand_words(self.rng, 4, self.vi)
                    self.text_line(cx0, ty, ws, self.body_face, base * 0.9, max_x=cx1)
        return ys[-1] + base * 1.2

    def block_logo(self, x0, y, width, base):
        s = base * self.rng.uniform(2, 4)
        col = self.accent if self.rng.random() < 0.6 else None
        d = self.draw(col)
        kind = self.rng.random()
        if kind < 0.4:
            d.rectangle((x0, y, x0 + s * self.rng.uniform(1, 3), y + s), fill=255)
            # knocked-out letters in the block (paper shows through)
            f = font(self.face(), s * 0.6)
            d.text((x0 + s * 0.15, y + s * 0.15), rand_word(self.rng).upper()[:4], font=f, fill=0)
        elif kind < 0.7:
            d.ellipse((x0, y, x0 + s, y + s), fill=255)
            d.ellipse((x0 + s * 0.25, y + s * 0.25, x0 + s * 0.75, y + s * 0.75), fill=0)
        else:
            d.polygon([(x0, y + s), (x0 + s / 2, y), (x0 + s, y + s)], fill=255)
        self.word_boxes.append((x0, y, x0 + s, y + s))
        fc = self.face()
        self.text_line(x0 + s * 1.3, y + s * 0.2, [w.upper() for w in rand_words(self.rng, self.rng.randint(1, 4), self.vi)],
                       fc, base * self.rng.uniform(1.2, 2.0), col)
        return y + s + base

    def block_image(self, x0, y, width, base):
        """grayscale photo / figure (e.g. CV portrait): printed content -> must never be handwriting.
        Labelled print where its ink alpha > 0.5, like any other printed ink."""
        w = int(width * self.rng.uniform(0.2, 0.6))
        if self.rng.random() < 0.5:
            x0 = x0 + width - w
        h = int(min(self.h - y - base, w * self.rng.uniform(0.5, 0.9)))
        if h < base * 3:
            return y
        img = photo_like(w, h, self.rng)
        circ = self.rng.random() < 0.35
        if circ:  # circular / rounded portrait crop
            yy, xx = np.mgrid[0:h, 0:w]
            img = img * ((((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) <= 1)
        layer = self.print_layers.setdefault(self.dark, Image.new("L", (self.w, self.h), 0))
        arr = np.asarray(layer).copy()
        xi, yi = int(x0), int(y)
        sub = arr[yi:yi + h, xi:xi + w]
        np.maximum(sub, (img[: sub.shape[0], : sub.shape[1]] * 255).astype(np.uint8), out=sub)
        self.print_layers[self.dark] = Image.fromarray(arr)
        if circ:
            lw = self.rng.choice([0, 2, 3, 5])
            if lw:
                self.draw().ellipse((xi - lw, yi - lw, xi + w + lw, yi + h + lw), outline=255, width=lw)
        elif self.rng.random() < 0.7:
            self.rect((xi, yi, xi + w, yi + h))
        self.text_line(x0, y + h + base * 0.3, ["Fig."] + rand_words(self.rng, 8, self.vi), self.body_face, base * 0.8)
        return y + h + base * 2

    def block_shapes(self, x0, y, width, base):
        """printed vector graphics: circles/ellipses, rounded boxes, icons, smooth curves, arrows.
        Hand-drawn marks are made of the same primitives, so the model must learn the regularity
        (perfect geometry, constant width) rather than 'circle = handwriting'."""
        rng = self.rng
        col = self.accent if rng.random() < 0.4 else None
        d = self.draw(col)
        h = base * rng.uniform(2, 6)
        cx = x0
        while cx < x0 + width - base * 2:
            k = rng.random()
            s = base * rng.uniform(0.6, 4)
            lw = rng.choice([1, 2, 2, 3, 4])
            if k < 0.25:
                d.ellipse((cx, y, cx + s * rng.uniform(1, 1.8), y + s), outline=255, width=lw)
            elif k < 0.4:
                d.rounded_rectangle((cx, y, cx + s * rng.uniform(1, 3), y + s), radius=s * 0.25, outline=255, width=lw)
            elif k < 0.6:  # small filled icons next to text (CV contact lines, bullets)
                r = base * rng.uniform(0.3, 0.6)
                shape = rng.choice(["circle", "square", "tri", "pin"])
                if shape == "circle":
                    d.ellipse((cx, y, cx + r * 2, y + r * 2), fill=255)
                elif shape == "square":
                    d.rectangle((cx, y, cx + r * 2, y + r * 1.6), fill=255)
                elif shape == "tri":
                    d.polygon([(cx, y + r * 2), (cx + r, y), (cx + r * 2, y + r * 2)], fill=255)
                else:
                    d.ellipse((cx, y, cx + r * 2, y + r * 2), fill=255)
                    d.polygon([(cx + r * 0.3, y + r * 1.4), (cx + r, y + r * 2.6), (cx + r * 1.7, y + r * 1.4)], fill=255)
                self.text_line(cx + r * 2.8, y - r * 0.2, rand_words(rng, rng.randint(1, 4), self.vi), self.body_face, base, max_x=x0 + width)
                s = base * 8
            elif k < 0.8:  # smooth printed curve (template swoosh)
                pts = catmull_rom([(cx, y + s * rng.uniform(0, 1)), (cx + width * 0.3, y + s * rng.uniform(0, 1)),
                                   (cx + width * 0.6, y + s * rng.uniform(0, 1)), (x0 + width, y + s * rng.uniform(0, 1))], 20)
                d.line([tuple(p) for p in pts], fill=255, width=lw, joint="curve")
                s = width
            else:  # printed arrow
                L = s * 2
                d.line([(cx, y + s / 2), (cx + L, y + s / 2)], fill=255, width=lw)
                d.polygon([(cx + L, y + s / 2 - s * 0.25), (cx + L + s * 0.4, y + s / 2), (cx + L, y + s / 2 + s * 0.25)], fill=255)
                s = L + s * 0.4
            self.word_boxes.append((cx, y, cx + s, y + base))
            cx += s + base * rng.uniform(1, 4)
        return y + h

    def block_cvheader(self, x0, y, width, base):
        """CV-style header: circular portrait (+ printed ring), name, contact icons, rule/curve."""
        rng = self.rng
        s = int(base * rng.uniform(5, 9))
        img = photo_like(s, s, rng)
        yy, xx = np.mgrid[0:s, 0:s]
        img = img * ((((xx - s / 2) / (s / 2)) ** 2 + ((yy - s / 2) / (s / 2)) ** 2) <= 1)
        layer = self.print_layers.setdefault(self.dark, Image.new("L", (self.w, self.h), 0))
        arr = np.asarray(layer).copy()
        xi, yi = int(x0), int(y)
        sub = arr[yi:yi + s, xi:xi + s]
        np.maximum(sub, (img[: sub.shape[0], : sub.shape[1]] * 255).astype(np.uint8), out=sub)
        self.print_layers[self.dark] = Image.fromarray(arr)
        d = self.draw()
        lw = rng.choice([2, 3, 4, 6])
        d.ellipse((xi - lw, yi - lw, xi + s + lw, yi + s + lw), outline=255, width=lw)
        tx = x0 + s * 1.2
        self.text_line(tx, y + s * 0.2, rand_words(rng, 3, self.vi), self.face(), base * rng.uniform(1.4, 2.0), max_x=x0 + width)
        if rng.random() < 0.7:
            pts = catmull_rom([(xi + s, y + s * 0.45), (tx + width * 0.2, y + s * 0.42), (x0 + width, y + s * 0.4 + rng.uniform(-5, 5))], 20)
            d.line([tuple(p) for p in pts], fill=255, width=rng.choice([1, 2, 3]), joint="curve")
        self.block_shapes(tx, y + s * 0.55, width - s * 1.2, base * 0.8)
        return y + s + base * 1.5

    def block_footnote(self, x0, y, width, base):
        size = max(9, base * self.rng.uniform(0.55, 0.75))
        self.hline(x0, x0 + width * 0.3, y, 1)
        return self.paragraph(x0, y + size * 0.6, width, self.body_face, size, self.rng.randint(1, 3)) + size


def layout_flow(page: Page, base):
    rng = page.rng
    mx = int(page.w * rng.uniform(0.05, 0.12))
    my = int(page.h * rng.uniform(0.03, 0.08))
    x0, width = mx, page.w - 2 * mx
    y = my
    page.body_face = page.face(mono=rng.random() < 0.08)
    kind = rng.random()
    if kind < 0.35:
        weights = {"heading": 2, "paragraph": 5, "list": 2, "questions": 1, "form": 1, "table": 1.5, "logo": 0.4, "image": 1.0, "footnote": 0.4, "shapes": 0.7}
    elif kind < 0.65:  # worksheet / exam
        weights = {"heading": 1.5, "paragraph": 1, "list": 0.5, "questions": 6, "form": 1, "table": 0.8, "logo": 0.3, "image": 0.3, "footnote": 0.2, "shapes": 0.5}
    elif kind < 0.85:  # form-ish
        weights = {"heading": 1.5, "paragraph": 1, "list": 0.5, "questions": 0.5, "form": 5, "table": 2, "logo": 0.8, "image": 0.6, "footnote": 0.3, "shapes": 0.6}
    else:  # table-heavy report
        weights = {"heading": 1.5, "paragraph": 1.5, "list": 0.5, "questions": 0.3, "form": 0.5, "table": 5, "logo": 0.3, "image": 0.3, "footnote": 0.3, "shapes": 0.6}
    keys, ws = list(weights), list(weights.values())
    if rng.random() < 0.18:
        y = page.block_cvheader(x0, y, width, base)
    elif rng.random() < 0.7:
        y = page.block_heading(x0, y, width, base)
    two_col = rng.random() < 0.15 and page.w > 900
    cols = [(x0, width)] if not two_col else [(x0, width / 2 - base), (x0 + width / 2 + base, width / 2 - base)]
    for cx, cw in cols:
        yy = y
        guard = 0
        while yy < page.h - my - base * 2 and guard < 40:
            guard += 1
            b = rng.choices(keys, ws)[0]
            yy = getattr(page, "block_" + b)(cx, yy, cw, base) + base * rng.uniform(0.2, 1.2)
    # page number
    if rng.random() < 0.5:
        page.text_line(page.w / 2, page.h - my * 0.7, [str(rng.randint(1, 300))], page.body_face, base * 0.8)


def layout_calendar(page: Page, base):
    rng = page.rng
    page.body_face = page.face()
    mx, my = int(page.w * 0.04), int(page.h * 0.04)
    y = page.block_heading(mx, my, page.w - 2 * mx, base * 1.2)
    days = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"] if page.vi else ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ncol, nrow = 7, rng.choice([5, 6])
    cw = (page.w - 2 * mx) / ncol
    rh = (page.h - y - my - base * 2) / nrow
    for c in range(ncol):
        page.text_line(mx + c * cw + base * 0.4, y, [days[c]], page.body_face, base, page.accent if rng.random() < 0.5 else None)
    y += base * 1.6
    lw = rng.choice([1, 2])
    for r in range(nrow + 1):
        page.hline(mx, page.w - mx, y + r * rh, lw)
    d = page.draw()
    for c in range(ncol + 1):
        d.line([(mx + c * cw, y), (mx + c * cw, y + nrow * rh)], fill=255, width=lw)
    start = rng.randint(0, 6)
    ndays = rng.choice([28, 30, 31])
    for i in range(nrow * ncol):
        r, c = divmod(i, ncol)
        num = i - start + 1
        cx0, cy0 = mx + c * cw, y + r * rh
        if 1 <= num <= ndays:
            page.text_line(cx0 + base * 0.3, cy0 + base * 0.2, [str(num)], page.body_face, base * rng.uniform(0.9, 1.4),
                           page.accent if c == 6 and rng.random() < 0.6 else None)
            if rng.random() < 0.1:
                page.text_line(cx0 + base * 0.3, cy0 + rh - base * 1.3, rand_words(rng, 2, page.vi), page.body_face, base * 0.6, max_x=cx0 + cw - 4)
        page.slots.append((cx0 + base * 0.2, cy0 + base * 1.6, cx0 + cw - base * 0.2, cy0 + rh - base * 0.2, cy0 + rh * 0.7))
        page.word_boxes.append((cx0 + 2, cy0 + 2, cx0 + base * 2, cy0 + base * 1.6))


def layout_receipt(page: Page, base):
    rng = page.rng
    page.body_face = page.face(mono=rng.random() < 0.7)
    mx = int(page.w * 0.06)
    y = int(page.h * 0.03)
    width = page.w - 2 * mx
    y = page.block_heading(mx, y, width, base)
    for _ in range(rng.randint(1, 3)):
        page.text_line(mx, y, rand_words(rng, 8, page.vi), page.body_face, base * 0.85, max_x=mx + width)
        y += base * 1.4
    f = font(page.body_face, base)
    while y < page.h - base * 8:
        r = rng.random()
        if r < 0.12:
            page.hline(mx, mx + width, y + base * 0.6, 1, dotted=rng.random() < 0.6)
            y += base * 1.3
            continue
        _, _ = page.text_line(mx, y, rand_words(rng, rng.randint(1, 4), page.vi), page.body_face, base, max_x=mx + width * 0.6)
        price = rand_number(rng)
        pw = f.getlength(price)
        page.text_line(mx + width - pw, y, [price], page.body_face, base)
        y += base * rng.uniform(1.25, 1.6)
    page.hline(mx, mx + width, y, 2)
    y += base * 0.6
    tl = ["TỔNG", "CỘNG:"] if page.vi else ["TOTAL:"]
    page.text_line(mx, y, tl, page.body_face, base * 1.3)
    tot = rand_number(rng)
    page.text_line(mx + width - font(page.body_face, base * 1.3).getlength(tot), y, [tot], page.body_face, base * 1.3)
    y += base * 2.5
    page.sign_lines.append((mx, y, mx + width, y + base * 3))
    page.slots.append((mx, y, mx + width, y + base * 3, y + base * 2.5))


# ----------------------------------------------------------------------------------------------
# handwriting placement
# ----------------------------------------------------------------------------------------------
PEN_COLORS = {
    "blue": [(20, 40, 160), (30, 60, 190), (10, 30, 110), (40, 80, 200), (25, 35, 90)],
    "black": [(15, 15, 20), (35, 35, 40), (25, 20, 30), (50, 50, 55)],
    # dark blue-black ballpoint: nearly as dark as print, only slightly bluer (low saturation)
    "blueblack": [(28, 30, 48), (20, 24, 40), (35, 38, 60), (24, 26, 34), (40, 42, 70), (30, 34, 55)],
    "pencil": [(95, 95, 100), (115, 115, 118), (80, 80, 85), (130, 128, 125)],
    "red": [(200, 25, 30), (170, 20, 40), (220, 50, 50), (150, 10, 20)],
    "green": [(20, 130, 60), (10, 100, 50)],
    "purple": [(100, 40, 150)],
}
PEN_P = {"blue": 0.24, "blueblack": 0.22, "black": 0.2, "pencil": 0.12, "red": 0.14, "green": 0.04, "purple": 0.04}


class Writer:
    """one 'hand' per page (or a second one, e.g. teacher's red pen)."""

    def __init__(self, rng, kind=None):
        self.rng = rng
        self.kind = kind or rng.choices(list(PEN_P), list(PEN_P.values()))[0]
        self.color = rng.choice(PEN_COLORS[self.kind])
        self.width_scale = rng.uniform(0.5, 1.4) if rng.random() < 0.7 else rng.uniform(1.2, 1.8)  # mostly thin ballpoint
        self.src_w = [rng.uniform(0.2, 1), rng.uniform(0.2, 1), rng.uniform(0.1, 0.6)]  # emnist, font, cursive
        self.slant = rng.uniform(-0.15, 0.35)
        self.opacity = rng.uniform(0.55, 0.9) if self.kind == "pencil" else rng.uniform(0.8, 1.0)
        # "neat ballpoint" hand: upright, thin, near-black/blue-black, print-sized — the hard case
        # (real photos: shape is the only cue, colour/darkness barely differ from print)
        self.neat = kind is None and rng.random() < 0.4
        if self.neat:
            self.kind = rng.choice(["blueblack", "blueblack", "black", "blue"])
            self.color = rng.choice(PEN_COLORS[self.kind][:4])
            self.width_scale = rng.uniform(0.45, 0.95)
            self.slant = rng.uniform(-0.06, 0.12)
            self.src_w = [rng.uniform(0.4, 1), rng.uniform(0.3, 1), rng.uniform(0.1, 0.4)]
            self.opacity = rng.uniform(0.85, 1.0)


def hw_text_patch(text, cap_h, writer: Writer, rng, vi=False):
    src = rng.choices(["emnist", "font", "cursive"], writer.src_w)[0]
    if src == "emnist":
        sw = max(1.0, cap_h * rng.uniform(0.055, 0.12) * writer.width_scale) if rng.random() < 0.8 else None
        cov, base = emnist_word(text, cap_h, rng, stroke_w=sw)
        if sw is None:
            cov = thicken(cov, rng, rng.uniform(-0.6, 1.0) * writer.width_scale)
        if rng.random() < (0.35 if writer.neat else 0.6):
            cov = elastic(cov, rng, alpha=rng.uniform(0.6, 1.5) if writer.neat else None)
    elif src == "font":
        cov, base = font_word(text, cap_h, rng, vi)
        if writer.neat or rng.random() < 0.3:
            cov = restroke(cov, max(1.0, cap_h * rng.uniform(0.055, 0.1) * writer.width_scale), rng)
        else:
            cov = thicken(cov, rng, rng.uniform(-0.3, 0.8) * writer.width_scale)
    else:
        cov, base = cursive_word(max(2, len(text)), cap_h, max(0.9, cap_h * rng.uniform(0.06, 0.12) * writer.width_scale), rng)
    if abs(writer.slant) > 0.03:
        cov = shear_patch(cov, writer.slant + rng.uniform(-0.05, 0.05))
    return cov, base


def _crop_cov(cov):
    ys, xs = np.where(cov > 0.05)
    if len(ys) == 0:
        return None, 0, 0
    return cov[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1], xs.min(), ys.min()


class HWLayer:
    def __init__(self, w, h):
        self.w, self.h = w, h
        # separate coverage per writer colour to keep multiply colouring right
        self.items = []  # (cov_page_region, x, y, writer)

    def add(self, cov, x, y, writer):
        cov, dx, dy = _crop_cov(cov)
        if cov is None:
            return None
        x, y = int(x + dx), int(y + dy)
        # clip to page
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + cov.shape[1]), min(self.h, y + cov.shape[0])
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        self.items.append((cov[y0 - y: y1 - y, x0 - x: x1 - x], x0, y0, writer))
        return (x0, y0, x1, y1)


def place_text_at(layer, page, writer, rng, x, baseline, cap_h, n_words=None, max_x=None):
    n_words = n_words or rng.randint(1, 6)
    boxes = []
    for _ in range(n_words):
        r = rng.random()
        if r < 0.25:
            t = rand_number(rng)
        elif r < 0.35:
            t = str(rng.randint(0, 10))
        else:
            t = rand_word(rng, page.vi and rng.random() < 0.6)
            if rng.random() < 0.3:
                t = t.lower()
        cov, base = hw_text_patch(t, cap_h, writer, rng, page.vi)
        if max_x is not None and x + cov.shape[1] * 0.6 > max_x:
            break
        b = layer.add(cov, x, baseline - base, writer)
        if b:
            boxes.append(b)
            x = b[2] + cap_h * rng.uniform(0.4, 1.0)
    return boxes


def mark_canvas_for(box, pad):
    x0, y0, x1, y1 = box
    return int(x0 - pad), int(y0 - pad), int(x1 - x0 + 2 * pad), int(y1 - y0 + 2 * pad)


def draw_mark(layer, writer, rng, kind, box, base):
    x0, y0, x1, y1 = [float(v) for v in box]
    w, h = max(4.0, x1 - x0), max(4.0, y1 - y0)
    pad = max(w, h) * 0.6 + 20
    ox, oy, cw, ch = mark_canvas_for((x0, y0, x1, y1), pad)
    cw, ch = min(cw, 2200), min(ch, 2200)
    sc = StrokeCanvas(cw, ch)
    lx0, ly0 = x0 - ox, y0 - oy
    cx, cy = lx0 + w / 2, ly0 + h / 2
    width = max(1.0, base * rng.uniform(0.08, 0.16) * writer.width_scale)
    J = lambda s: rng.uniform(-s, s)
    if kind == "circle":
        rx, ry = w / 2 * rng.uniform(1.05, 1.35) + 3, h / 2 * rng.uniform(1.1, 1.7) + 3
        turns = rng.uniform(1.0, 1.35)
        n = 40
        t0 = rng.uniform(0, 6.28)
        pts = []
        for i in range(n + 1):
            t = t0 + i / n * turns * 6.283
            rr = 1 + J(0.06) + 0.04 * i / n
            pts.append((cx + rx * rr * math.cos(t), cy + ry * rr * math.sin(t)))
        sc.polyline(catmull_rom(pts, 4), width, rng)
    elif kind == "underline":
        n = rng.randint(3, 7)
        yy = ly0 + h + rng.uniform(1, 0.35 * h + 2)
        pts = [(lx0 - J(6) + i * w / n, yy + J(1.5) + i * J(0.4)) for i in range(n + 1)]
        sc.polyline(catmull_rom(pts, 4), width, rng)
        if rng.random() < 0.2:
            sc.polyline(catmull_rom([(p[0], p[1] + width * 2.5 + J(1)) for p in pts], 4), width, rng)
    elif kind == "strike":
        n = rng.randint(2, 6)
        yy = ly0 + h * rng.uniform(0.35, 0.65)
        slope = J(0.12)
        pts = [(lx0 - rng.uniform(0, 8) + i * (w + 10) / n, yy + slope * i * w / n + J(1.5)) for i in range(n + 1)]
        sc.polyline(catmull_rom(pts, 4), width, rng)
        if rng.random() < 0.3:
            pts2 = [(p[0], p[1] + J(h * 0.25)) for p in pts]
            sc.polyline(catmull_rom(pts2, 4), width, rng)
    elif kind == "scribble":
        n = int(max(4, w / max(3, h * 0.25)))
        pts = []
        for i in range(n + 1):
            pts.append((lx0 + i * w / n + J(2), ly0 + (h if i % 2 else 0) + J(h * 0.2)))
        sc.polyline(catmull_rom(pts, 3), width * rng.uniform(0.8, 1.4), rng)
    elif kind == "tick":
        s = max(h, base) * rng.uniform(0.8, 1.6)
        px, py = cx + J(w * 0.3), cy + J(h * 0.2)
        pts = [(px - s * 0.4, py - s * 0.05), (px - s * 0.1, py + s * 0.35), (px + s * 0.55 + J(3), py - s * 0.6 + J(3))]
        sc.polyline(catmull_rom(pts, 5), width, rng)
    elif kind == "cross":
        s = max(h, base) * rng.uniform(0.7, 1.4)
        px, py = cx + J(w * 0.2), cy
        sc.polyline(catmull_rom([(px - s / 2, py - s / 2), (px + J(2), py + J(2)), (px + s / 2, py + s / 2)], 4), width, rng)
        sc.polyline(catmull_rom([(px + s / 2 + J(2), py - s / 2), (px - s / 2, py + s / 2 + J(2))], 4), width, rng)
    elif kind == "box":
        m = rng.uniform(2, 8)
        pts = [(lx0 - m, ly0 - m + J(2)), (lx0 + w + m + J(2), ly0 - m), (lx0 + w + m, ly0 + h + m + J(2)),
               (lx0 - m + J(2), ly0 + h + m), (lx0 - m + J(3), ly0 - m + J(3))]
        sc.polyline(np.array(pts), width, rng)
    elif kind == "arrow":
        L = rng.uniform(3, 8) * base
        ang = rng.uniform(0, 6.28)
        sx, sy = cx + math.cos(ang) * L, cy + math.sin(ang) * L
        mx_, my_ = (sx + cx) / 2 + J(L * 0.3), (sy + cy) / 2 + J(L * 0.3)
        ex, ey = cx + math.cos(ang) * (w / 2 + 6), cy + math.sin(ang) * (h / 2 + 6)
        path = catmull_rom([(sx, sy), (mx_, my_), (ex, ey)], 6)
        # canvas might be too small: expand by using page-relative big canvas
        sc2 = StrokeCanvas(int(cw + 2 * L), int(ch + 2 * L))
        off = L
        path = path + off
        sc2.polyline(path, width, rng)
        d = path[-1] - path[-4]
        a = math.atan2(d[1], d[0])
        hl = base * rng.uniform(0.6, 1.2)
        for s_ in (+1, -1):
            aa = a + math.pi + s_ * rng.uniform(0.4, 0.7)
            sc2.polyline(np.array([path[-1], path[-1] + [hl * math.cos(aa), hl * math.sin(aa)]]), width, rng, pressure=False)
        return layer.add(sc2.result(), ox - off, oy - off, writer)
    elif kind == "signature":
        n = rng.randint(6, 14)
        pts = []
        x = lx0
        for i in range(n):
            x += w / n * rng.uniform(0.6, 1.4)
            pts.append((x, cy + J(h * 0.5)))
        sc.polyline(catmull_rom(pts, 8), width, rng)
        if rng.random() < 0.6:
            sc.polyline(catmull_rom([(lx0 + J(10), cy + h * 0.3), (lx0 + w * 0.5, cy + h * 0.2 + J(4)), (lx0 + w, cy + h * 0.35)], 6), width, rng)
    elif kind == "bracket":
        pts = [(lx0 - 8, ly0), (lx0 - 14 + J(2), ly0 + h * 0.2), (lx0 - 14 + J(2), ly0 + h * 0.8), (lx0 - 8, ly0 + h)]
        sc.polyline(catmull_rom(pts, 5), width, rng)
    return layer.add(sc.result(), ox, oy, writer)


def add_handwriting(page: Page, base, rng, split):
    layer = HWLayer(page.w, page.h)
    if rng.random() < 0.12:
        return layer
    writers = [Writer(rng)]
    if rng.random() < 0.35:
        writers.append(Writer(rng, rng.choice(["red", "red", "green", "blue", "black", "pencil"])))
    n_items = rng.randint(4, 30)
    cap_base = base * 0.72  # printed cap height ~ 0.72 * font size
    for _ in range(n_items):
        wr = rng.choice(writers)
        cap_h = int(max(9, cap_base * rng.uniform(0.8, 1.8)))
        r = rng.random()
        if r < 0.30 and page.slots:
            s = rng.choice(page.slots)
            x0, y0, x1, y1, bl = s
            place_text_at(layer, page, wr, rng, x0 + rng.uniform(0, max(1, (x1 - x0) * 0.3)),
                          bl + rng.uniform(-cap_h * 0.3, cap_h * 0.3),
                          int(min(cap_h, max(9, (y1 - y0) * 0.75))), max_x=x1 + cap_h * 2)
        elif r < 0.50 and page.word_boxes:
            # marks on printed words (overlap-heavy)
            b = rng.choice(page.word_boxes)
            if rng.random() < 0.4 and len(page.word_boxes) > 1:
                # span several consecutive words
                i = page.word_boxes.index(b)
                j = min(len(page.word_boxes) - 1, i + rng.randint(1, 4))
                bb = page.word_boxes[j]
                if abs(bb[1] - b[1]) < base:
                    b = (min(b[0], bb[0]), min(b[1], bb[1]), max(b[2], bb[2]), max(b[3], bb[3]))
            kind = rng.choices(["circle", "underline", "strike", "scribble", "box", "arrow", "tick", "cross", "bracket"],
                               [4, 3, 3.5, 1.2, 1, 1, 1, 0.7, 0.5])[0]
            draw_mark(layer, wr, rng, kind, b, base)
        elif r < 0.58 and (page.mcq or page.checkboxes):
            b = rng.choice(page.mcq + page.checkboxes)
            kind = rng.choices(["circle", "tick", "cross"], [4, 3, 1.5])[0]
            draw_mark(layer, wr, rng, kind, b, base)
        elif r < 0.76 and page.line_boxes:
            # handwriting written over / squeezed between printed lines (heavy overlap)
            lb = rng.choice(page.line_boxes)
            x = rng.uniform(lb[0], max(lb[0] + 1, lb[2] - cap_h * 4))
            bl = rng.choice([lb[3], (lb[1] + lb[3]) / 2, lb[1] + cap_h * 0.3, lb[3] + cap_h * 0.6])
            place_text_at(layer, page, wr, rng, x, bl, cap_h, n_words=rng.randint(1, 5))
        elif r < 0.80 and page.sign_lines:
            b = rng.choice(page.sign_lines)
            bw = min(b[2] - b[0], base * rng.uniform(5, 12))
            sx = b[0] + rng.uniform(0, max(1, b[2] - b[0] - bw))
            draw_mark(layer, wr, rng, "signature", (sx, b[1], sx + bw, b[3]), base)
        elif r < 0.9:
            # free annotation anywhere (margins, blank areas, grades like "8/10", "OK!")
            x = rng.uniform(0, page.w - cap_h * 3)
            y = rng.uniform(cap_h * 2, page.h - cap_h)
            if rng.random() < 0.3:
                t = rng.choice([f"{rng.randint(0, 10)}/10", f"{rng.randint(0, 100)}", "OK", "Good!", "Tốt", "?", "!!",
                                f"{rng.randint(1, 9)}.{rng.randint(0, 9)}", "Đạt", "x2", "=", "No"])
                cov, bs = hw_text_patch(t, int(cap_h * rng.uniform(1.0, 2.2)), wr, rng, page.vi)
                layer.add(cov, x, y - bs, wr)
            else:
                place_text_at(layer, page, wr, rng, x, y, cap_h, max_x=page.w)
        else:
            # standalone marks in free space: arrow/star/line/cursive scribble
            x = rng.uniform(0, page.w - 100)
            y = rng.uniform(0, page.h - 60)
            kind = rng.choice(["scribble", "arrow", "underline", "signature", "tick", "circle"])
            draw_mark(layer, wr, rng, kind, (x, y, x + base * rng.uniform(2, 10), y + base * rng.uniform(0.8, 3)), base)
    return layer


# ----------------------------------------------------------------------------------------------
# compositing + degradations
# ----------------------------------------------------------------------------------------------
def photo_like(w, h, rng):
    """grayscale photo-ish texture: multi-octave noise, gradients, soft blobs (a 'head'), edges."""
    nr = np.random.default_rng(rng.randrange(1 << 30))
    img = np.zeros((h, w), np.float32)
    for octave, amp in ((4, 1.0), (12, 0.5), (40, 0.25), (120, 0.12)):
        n = nr.random((max(2, h * octave // max(h, w)), max(2, w * octave // max(h, w)))).astype(np.float32)
        img += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    gx, gy = np.meshgrid(np.linspace(0, 1, w, dtype=np.float32), np.linspace(0, 1, h, dtype=np.float32))
    img += rng.uniform(-1, 1) * gx + rng.uniform(-1, 1) * gy
    if rng.random() < 0.6:
        cx, cy, rx, ry = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.3, 0.5), w * rng.uniform(0.15, 0.3), h * rng.uniform(0.2, 0.35)
        blob = (((gx * w - cx) / rx) ** 2 + ((gy * h - cy) / ry) ** 2 < 1).astype(np.float32)
        img += rng.uniform(-1.2, 1.2) * cv2.GaussianBlur(blob, (0, 0), max(1, min(w, h) * 0.02))
        body = ((gy * h > cy + ry * 0.9) & (np.abs(gx * w - cx) < rx * 2.2)).astype(np.float32)
        img += rng.uniform(-1.2, 1.2) * cv2.GaussianBlur(body, (0, 0), max(1, min(w, h) * 0.02))
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    img = img ** rng.uniform(0.6, 1.6)
    return np.clip(img * rng.uniform(0.6, 1.0), 0, 1)


def add_show_through(img, page: Page, rng):
    """faint mirrored text from the back of the sheet: visible but NOT ink (no label)."""
    w, h = page.w, page.h
    lim = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(lim)
    fc = rng.choice(PRINT_FACES)
    size = rng.uniform(16, 34)
    f = font(fc, size)
    y = rng.uniform(0, h * 0.2)
    while y < h:
        d.text((rng.uniform(w * 0.03, w * 0.12), y), " ".join(rand_words(rng, 25)), font=f, fill=255)
        y += size * rng.uniform(1.2, 1.8)
    a = np.asarray(lim, np.float32)[:, ::-1] / 255.0
    a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.8, 2.5)) * rng.uniform(0.06, 0.28)
    img *= (1.0 - a)[..., None]


def make_paper(w, h, rng):
    nr = np.random.default_rng(rng.randrange(1 << 30))
    lum = rng.uniform(228, 255) if rng.random() < 0.65 else rng.uniform(190, 232)  # white .. greyish photographed paper
    tint = rng.choice([(0, 0, 0), (0, 0, 0), (2, 1, -6), (3, 1, -8), (-3, -1, 2), (1, 1, -3)])  # white / cream / cool
    base = np.minimum(np.array([lum + tint[0], lum + tint[1], lum + tint[2]], np.float32), 255)
    paper = np.ones((h, w, 3), np.float32) * base
    # fibre / grain texture
    amp = rng.uniform(0, 5)
    g = nr.standard_normal((h // 2 + 1, w // 2 + 1)).astype(np.float32)
    g = cv2.resize(cv2.GaussianBlur(g, (0, 0), rng.uniform(0.5, 2)), (w, h))[..., None] * amp
    paper += g
    return paper


def composite(page: Page, layer: HWLayer, rng):
    w, h = page.w, page.h
    img = make_paper(w, h, rng)
    print_alpha = np.zeros((h, w), np.float32)
    for color, lim in page.print_layers.items():
        a = np.asarray(lim, np.float32) / 255.0
        if rng.random() < 0.3:
            a = a * rng.uniform(0.85, 1.0)  # slightly faded toner
        col = np.array(color, np.float32) / 255.0
        img *= 1.0 - a[..., None] * (1.0 - col)
        np.maximum(print_alpha, a, out=print_alpha)
    print_mask = print_alpha > 0.5  # clean-document mask, includes ink later hidden under handwriting
    if rng.random() < 0.3:
        add_show_through(img, page, rng)
    hw_cov = np.zeros((h, w), np.float32)
    nr = np.random.default_rng(rng.randrange(1 << 30))
    for cov, x, y, wr in layer.items:
        ph, pw = cov.shape
        np.maximum(hw_cov[y:y + ph, x:x + pw], cov, out=hw_cov[y:y + ph, x:x + pw])
        ink = cov * wr.opacity
        # pressure / ink-flow variation (low freq) and pencil grain (high freq)
        lf = cv2.resize(nr.random((max(2, ph // 24), max(2, pw // 24))).astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)
        ink = ink * (0.78 + 0.22 * np.clip(lf, 0, 1))
        if wr.kind == "pencil":
            grain = np.clip(nr.normal(0.8, 0.25, (ph, pw)).astype(np.float32), 0.25, 1.0)
            ink = ink * grain
        col = np.array(wr.color, np.float32) / 255.0
        img[y:y + ph, x:x + pw] *= 1.0 - np.clip(ink, 0, 1)[..., None] * (1.0 - col)  # MULTIPLY blend
    hw_mask = hw_cov > 0.3
    masks = print_mask.astype(np.uint8) | (hw_mask.astype(np.uint8) << 1)
    return img, masks


def degrade(img, masks, rng):
    h, w = masks.shape
    nr = np.random.default_rng(rng.randrange(1 << 30))
    # geometry: rotation +-2 deg + slight perspective, same warp for masks (nearest)
    ang = math.radians(rng.uniform(-2, 2))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = 0.012 * min(w, h)
    dst = src + np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    c, s = math.cos(ang), math.sin(ang)
    ctr = np.float32([w / 2, h / 2])
    dst = (dst - ctr) @ np.float32([[c, -s], [s, c]]).T + ctr
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    pap = tuple(float(v) for v in img[5, 5])
    img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=pap)
    masks = cv2.warpPerspective(masks, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    # resolution loss
    if rng.random() < 0.35:
        f = rng.uniform(0.6, 0.9)
        img = cv2.resize(cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.55:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    # mild uneven illumination (the app removes shadows, so keep it gentle)
    if rng.random() < 0.6:
        lf = cv2.resize(nr.random((3, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        img *= (1.0 - rng.uniform(0.02, 0.18) * np.clip(lf, 0, 1))[..., None]
    # white balance
    img *= np.array([rng.uniform(0.96, 1.04) for _ in range(3)], np.float32)
    # the app sharpens before inference
    if rng.random() < 0.45:
        bl = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 2.0))
        k = rng.uniform(0.3, 1.2)
        img = img + k * (img - bl)
    img += nr.normal(0, rng.uniform(0.5, 5), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        q = rng.randint(45, 95)
        ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(img), masks


# ----------------------------------------------------------------------------------------------
# page generation + tiling
# ----------------------------------------------------------------------------------------------
def render_page(seed, split):
    rng = random.Random(seed)
    kind = rng.random()
    # page geometry at inference scale (longer side = 1536)
    if kind < 0.1:
        ptype = "receipt"
        w, h = int(LONG_SIDE * rng.uniform(0.3, 0.5)), LONG_SIDE
    elif kind < 0.2:
        ptype = "calendar"
        w, h = (LONG_SIDE, int(LONG_SIDE * rng.uniform(0.65, 0.75))) if rng.random() < 0.6 else (int(LONG_SIDE * 0.707), LONG_SIDE)
    else:
        ptype = "flow"
        r = rng.random()
        if r < 0.68:
            w, h = int(LONG_SIDE * rng.uniform(0.68, 0.78)), LONG_SIDE          # portrait A4/letter
        elif r < 0.85:
            w, h = LONG_SIDE, int(LONG_SIDE * rng.uniform(0.62, 0.78))          # landscape
        else:
            w, h = (LONG_SIDE, int(LONG_SIDE * rng.uniform(0.4, 0.95))) if rng.random() < 0.5 else (int(LONG_SIDE * rng.uniform(0.45, 0.95)), LONG_SIDE)  # partial page
    page = Page(w, h, rng)
    # body font size in px: cap height ~12-26 px  -> size ~ cap/0.72
    zoom = rng.uniform(1.2, 2.0) if (ptype == "flow" and min(w, h) < 0.62 * LONG_SIDE and rng.random() < 0.6) else 1.0
    base = rng.uniform(12, 26) / 0.72 * zoom
    if ptype == "receipt":
        base = rng.uniform(14, 22) / 0.72
    if ptype == "flow":
        layout_flow(page, base)
    elif ptype == "calendar":
        layout_calendar(page, base * rng.uniform(0.9, 1.3))
    else:
        layout_receipt(page, base)
    layer = add_handwriting(page, base, rng, split)
    img, masks = composite(page, layer, rng)
    img, masks = degrade(img, masks, rng)
    return img, masks


def pad_to_grid(img, masks):
    h, w = masks.shape
    nx = max(1, math.ceil((w - TILE) / STRIDE) + 1)
    ny = max(1, math.ceil((h - TILE) / STRIDE) + 1)
    W, H = (nx - 1) * STRIDE + TILE, (ny - 1) * STRIDE + TILE
    pi = np.full((H, W, 3), 255, np.uint8)
    pm = np.zeros((H, W), np.uint8)
    pi[:h, :w] = img
    pm[:h, :w] = masks
    return pi, pm, nx, ny


def tiles_from_page(img, masks, rng, n_hw_max=14, p_nohw=0.18, n_random=6):
    pi, pm, nx, ny = pad_to_grid(img, masks)
    cands = []
    for j in range(ny):
        for i in range(nx):
            cands.append((j * STRIDE, i * STRIDE))
    H, W = pm.shape
    for _ in range(n_random):
        cands.append((rng.randint(0, H - TILE), rng.randint(0, W - TILE)))
    hw, nohw = [], []
    for (y, x) in cands:
        m = pm[y:y + TILE, x:x + TILE]
        nhw = int(((m & 2) > 0).sum())
        (hw if nhw >= 40 else nohw).append((y, x, int((m & 1).sum())))
    rng.shuffle(hw)
    hw = hw[:n_hw_max]
    out = [(y, x) for y, x, _ in hw]
    # no-handwriting tiles: mostly ones with printed content, few blank
    rng.shuffle(nohw)
    n_no = int(round(p_nohw / (1 - p_nohw) * max(len(hw), 4)))
    printed = [t for t in nohw if t[2] > 200]
    blank = [t for t in nohw if t[2] <= 200]
    pick = printed[: max(0, n_no - 1)] + (blank[:1] if rng.random() < 0.5 else [])
    out += [(y, x) for y, x, _ in pick]
    return [(pi[y:y + TILE, x:x + TILE].copy(), pm[y:y + TILE, x:x + TILE].copy()) for y, x in out]


_WORKER_SPLIT = None


def _init_worker(split):
    global _WORKER_SPLIT
    _WORKER_SPLIT = split
    init_fonts()
    init_emnist(split)
    cv2.setNumThreads(1)


def _work(seed):
    img, masks = render_page(seed, _WORKER_SPLIT)
    rng = random.Random(seed * 7 + 3)
    tiles = tiles_from_page(img, masks, rng)
    return tiles, (img, masks)


def build_split(name, n_pages, seed0, workers, keep_pages=0):
    xs, ys, pages = [], [], []
    with Pool(workers, initializer=_init_worker, initargs=(name,)) as pool:
        for k, (tiles, page) in enumerate(pool.imap(_work, range(seed0, seed0 + n_pages), chunksize=2)):
            for t, m in tiles:
                xs.append(t)
                ys.append(m)
            if k < keep_pages:
                pages.append(page)
            if (k + 1) % 50 == 0:
                print(f"  {name}: {k + 1}/{n_pages} pages, {len(xs)} tiles", flush=True)
    X = np.stack(xs)
    Y = np.stack(ys)
    perm = np.random.default_rng(seed0).permutation(len(X))
    X, Y = X[perm], Y[perm]
    np.savez_compressed(os.path.join(ML_DIR, f"inkseg{TAG}_{name}.npz"), images=X, masks=Y)
    hw_frac = ((Y & 2) > 0).mean()
    pr_frac = ((Y & 1) > 0).mean()
    ov_frac = ((Y & 3) == 3).mean()
    nohw_tiles = (((Y & 2) > 0).reshape(len(Y), -1).sum(1) < 40).mean()
    print(f"saved inkseg{TAG}_{name}.npz: {X.shape}  print px {pr_frac:.3%}  hw px {hw_frac:.3%}  overlap px {ov_frac:.3%}  "
          f"tiles w/o hw {nohw_tiles:.1%}", flush=True)
    if pages:
        np.savez_compressed(os.path.join(ML_DIR, f"inkseg{TAG}_fullpages_{name}.npz"),
                            images=np.array([p[0] for p in pages], dtype=object),
                            masks=np.array([p[1] for p in pages], dtype=object))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-train", type=int, default=520)
    ap.add_argument("--pages-val", type=int, default=40)
    ap.add_argument("--pages-test", type=int, default=60)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    splits = [("train", args.pages_train, 1_000_000), ("val", args.pages_val, 2_000_000), ("test", args.pages_test, 3_000_000)]
    for name, n, s0 in splits:
        if args.only and name != args.only:
            continue
        print(f"Generating {name}: {n} pages", flush=True)
        build_split(name, n, s0, args.workers, keep_pages=6 if name == "test" else 0)


if __name__ == "__main__":
    main()
