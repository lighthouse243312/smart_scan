"""Generates a synthetic printed-vs-handwriting word-crop dataset.

Class 0 = printed (PIL-rendered system fonts). Class 1 = handwriting (EMNIST
`letters` glyphs composed into word-like crops, or a printed word with a small
hand-drawn mark overlaid). No cloud APIs, no sign-up-gated datasets — see
/Users/golfzon/.claude/plans/distributed-pondering-otter.md.

Output: train.npz / val.npz / test.npz in this directory, each holding
`images` (N, IMG_H, IMG_W, 1) uint8 and `labels` (N,) uint8.
"""
from __future__ import annotations

import random
import string

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw, ImageFilter, ImageFont

IMG_H = 64
IMG_W = 128
SEED = 1234

# Generous working canvas for printed text BEFORE the tight-bbox crop below — wide enough
# that longer words aren't clipped by the canvas edge during placement.
WORK_W = IMG_W * 2
WORK_H = IMG_H

# Verified against a real photo (Tesseract word boxes, standing in for ML Kit): median real
# word box is only ~34x16px with a median ~7px gap to the next word. The native classify
# pipeline (MlHandwritingClassifier.kt / .swift) crops each ML Kit box with this much padding
# before resizing to IMG_W x IMG_H — MUST match exactly, since training a model on a
# differently-cropped distribution than what it's fed in production is the whole reason this
# needed fixing in the first place (v1-v3 all trained on synthetic canvases with generous
# unrelated margin, which doesn't match a real tight word box blown up 4x — that mismatch,
# not stroke shape, was why real photos scored ~99% "handwriting" regardless of content).
NATIVE_PADDING_PX = 5.0

PRINTED_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "/System/Library/Fonts/Supplemental/Arial Narrow.ttf",
    "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Italic.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Microsoft Sans Serif.ttf",
    "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf",
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Courier.ttc",
    "/System/Library/Fonts/Palatino.ttc",
    "/System/Library/Fonts/Optima.ttc",
    "/System/Library/Fonts/Avenir.ttc",
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/Avenir Next Condensed.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/LucidaGrande.ttc",
    "/System/Library/Fonts/Geneva.ttf",
    "/System/Library/Fonts/NewYork.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/SFCompact.ttf",
    "/System/Library/Fonts/SFCompactRounded.ttf",
]

_WORD_POOL = (
    "the of and to in is you that it he was for on are as with his they "
    "at be this have from or one had by word but not what all were we "
    "when your can said there use each which she do how their if will up "
    "other about out many then them these so some her would make like him "
    "into time has look two more write go see number no way could people "
    "answer date name school book page line total sum find write circle "
    "underline complete match true false yes"
).split()


def random_word() -> str:
    # Standalone short numbers (calendar dates, page/question numbers, table cells) are common
    # printed content that a word-only pool never covers — verified on a real calendar photo:
    # specific digit shapes ("2", "4", "7") were consistently misread as handwriting regardless
    # of crop size, because the model had essentially never seen a bare 1-3 digit printed token,
    # only occasional digit SUFFIXES on a word. Every digit needs even coverage on its own.
    if random.random() < 0.2:
        return "".join(random.choices(string.digits, k=random.randint(1, 3)))

    word = random.choice(_WORD_POOL)
    style = random.random()
    if style < 0.25:
        word = word.upper()
    elif style < 0.4:
        word = word.capitalize()
    if random.random() < 0.15:
        word = word + random.choice(string.digits)
    if random.random() < 0.1:
        word = "".join(random.choices(string.ascii_letters + string.digits, k=random.randint(2, 7)))
    return word


def _tight_ink_bbox(arr: np.ndarray, threshold: int = 200) -> tuple[int, int, int, int] | None:
    mask = arr < threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_to_native_box(img: Image.Image) -> Image.Image:
    """Reproduces exactly what the native pipeline does: take the ink's own tight bounding box
    (standing in for an ML Kit word box), add NATIVE_PADDING_PX, crop, then bilinear-resize up
    to IMG_W x IMG_H. This — not stroke shape — is the actual input distribution the model sees
    in production, so it must be what generates the training images too.
    """
    arr = np.asarray(img)
    bbox = _tight_ink_bbox(arr)
    if bbox is None:
        return img.resize((IMG_W, IMG_H), resample=Image.BILINEAR)
    x0, y0, x1, y1 = bbox
    pad = max(0.0, NATIVE_PADDING_PX + random.uniform(-2, 2))
    cx0 = max(0, int(x0 - pad))
    cy0 = max(0, int(y0 - pad))
    cx1 = min(img.width, int(x1 + pad))
    cy1 = min(img.height, int(y1 + pad))
    if cx1 <= cx0:
        cx1 = cx0 + 1
    if cy1 <= cy0:
        cy1 = cy0 + 1
    crop = img.crop((cx0, cy0, cx1, cy1))
    return crop.resize((IMG_W, IMG_H), resample=Image.BILINEAR)


def jitter_canvas(img: Image.Image) -> np.ndarray:
    """Rotate (document skew), crop to the ink's own tight box exactly like production
    (see crop_to_native_box), then blur/contrast/noise jitter."""
    angle = random.uniform(-4, 4)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    img = crop_to_native_box(img)

    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    arr = np.asarray(img, dtype=np.float32)

    contrast = random.uniform(0.7, 1.3)
    brightness = random.uniform(-25, 15)
    arr = (arr - 128.0) * contrast + 128.0 + brightness

    noise_sigma = random.uniform(1.0, 8.0)
    arr = arr + np.random.normal(0, noise_sigma, arr.shape)

    return np.clip(arr, 0, 255).astype(np.uint8)


# Real word-detector boxes (ML Kit) are occasionally a few px off — clipping into a
# neighboring word or cutting a glyph short. Applied to a minority of samples of BOTH
# classes so the model doesn't key on "is this a complete, well-formed glyph shape".
EDGE_CLIP_PROB = 0.18


def _edge_clip_offset(text_w: float, text_h: float) -> tuple[int, int]:
    if random.random() >= EDGE_CLIP_PROB:
        return 0, 0
    side = random.choice(["left", "right", "top", "bottom"])
    if side == "left":
        return -int(text_w * random.uniform(0.1, 0.3)), 0
    if side == "right":
        return int(text_w * random.uniform(0.1, 0.3)), 0
    if side == "top":
        return 0, -int(text_h * random.uniform(0.15, 0.4))
    return 0, int(text_h * random.uniform(0.15, 0.4))


def _place_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    max_x = max(2, WORK_W - text_w - 4)
    max_y = max(2, WORK_H - text_h - 4)
    x = random.randint(0, max_x) - bbox[0]
    y = random.randint(0, max_y) - bbox[1]
    dx, dy = _edge_clip_offset(text_w, text_h)
    return x + dx, y + dy


def draw_hand_mark(draw: ImageDraw.ImageDraw, cx: float, cy: float, w: float, h: float) -> None:
    """Draws one small hand-drawn-style annotation (circle/checkmark/tick/scribble-underline)
    near (cx, cy) spanning roughly (w, h) — mimics a grader's pen mark on a printed answer
    (e.g. circling a multiple-choice letter). Built from a few jittered line segments instead
    of a perfect geometric shape, since a real hand never draws a clean ellipse. Deliberately
    NOT guaranteed to survive the later tight-bbox crop if it strays far from the text — a
    real ML Kit box is the PRINTED text's own extent, so a mark that doesn't touch/overlap the
    text may get partially or fully cropped away in production too; the model should learn
    from that reality, not from an idealized always-fully-visible mark.
    """
    ink = random.randint(0, 60)
    width = random.randint(2, 3)
    kind = random.choice(["circle", "check", "tick", "underline_scribble"])

    if kind == "circle":
        rx, ry = w * random.uniform(0.55, 0.75), h * random.uniform(0.55, 0.75)
        n = 14
        pts = []
        for i in range(n + 1):
            t = i / n * 2 * 3.14159
            jitter = random.uniform(-2, 2)
            pts.append((cx + (rx + jitter) * np.cos(t), cy + (ry + jitter) * np.sin(t)))
        draw.line(pts, fill=ink, width=width, joint="curve")
    elif kind == "check":
        p0 = (cx - w * 0.4, cy)
        p1 = (cx - w * 0.1, cy + h * 0.4)
        p2 = (cx + w * 0.45, cy - h * 0.45)
        draw.line([p0, p1, p2], fill=ink, width=width, joint="curve")
    elif kind == "tick":
        p0 = (cx - w * 0.3, cy + h * 0.3)
        p1 = (cx + w * 0.3, cy - h * 0.3)
        draw.line([p0, p1], fill=ink, width=width)
    else:  # underline_scribble
        n = 6
        y0 = cy + h * 0.5
        pts = [(cx - w * 0.5 + i * w / n, y0 + random.uniform(-2, 2)) for i in range(n + 1)]
        draw.line(pts, fill=ink, width=width, joint="curve")


def render_printed(font_path: str, with_overlay_mark: bool = False) -> np.ndarray:
    canvas = Image.new("L", (WORK_W, WORK_H), color=255)
    draw = ImageDraw.Draw(canvas)
    text = random_word()
    size = random.randint(20, 34)
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        font = ImageFont.load_default()

    x, y = _place_text(draw, text, font)
    ink = random.randint(0, 40)
    draw.text((x, y), text, font=font, fill=ink)

    if with_overlay_mark:
        bbox = draw.textbbox((x, y), text, font=font)
        # Center the mark on a random letter within the word, not the whole word — a grader
        # circles one option (e.g. just "B"), not the entire line.
        span = max(1, bbox[2] - bbox[0])
        focus_w = span * random.uniform(0.25, 0.5)
        focus_x = bbox[0] + random.uniform(0, span - focus_w) + focus_w / 2
        cy = (bbox[1] + bbox[3]) / 2
        draw_hand_mark(draw, focus_x, cy, max(focus_w, 14), max(bbox[3] - bbox[1], 14))

    return jitter_canvas(canvas)


def load_emnist_glyphs() -> dict[int, list[np.ndarray]]:
    """Loads EMNIST letters train split into a {label: [28x28 uint8, ...]} map.

    EMNIST bitmaps are stored transposed relative to normal reading orientation
    (a well-known quirk of the original NIST conversion) — fixed with a simple
    transpose of the two spatial axes.
    """
    ds = tfds.load("emnist/letters", split="train", as_supervised=True, data_dir="./tfds_data")
    glyphs: dict[int, list[np.ndarray]] = {}
    for img, label in tfds.as_numpy(ds):
        fixed = np.transpose(img.squeeze(-1), (1, 0))
        glyphs.setdefault(int(label), []).append(fixed)
    return glyphs


def render_handwriting(glyphs: dict[int, list[np.ndarray]]) -> np.ndarray:
    canvas = Image.new("L", (WORK_W, WORK_H), color=255)
    n_chars = random.randint(2, 6)
    glyph_h = random.randint(30, 46)

    clip_dx, clip_dy = _edge_clip_offset(WORK_W * 0.25, WORK_H * 0.5)
    cursor_x = random.randint(2, WORK_W // 3) + clip_dx
    base_y = WORK_H // 2 + random.randint(-4, 4) + clip_dy
    baseline_drift = random.uniform(-0.3, 0.3)

    for i in range(n_chars):
        label = random.randint(1, 26)
        bank = glyphs.get(label) or glyphs[random.choice(list(glyphs.keys()))]
        glyph = random.choice(bank)

        glyph_img = Image.fromarray(255 - glyph)  # EMNIST ink is bright-on-black; invert to dark-on-white
        scale = glyph_h / 28.0 * random.uniform(0.85, 1.15)
        gw, gh = max(4, int(28 * scale)), max(4, int(28 * scale))
        glyph_img = glyph_img.resize((gw, gh), resample=Image.BICUBIC)

        rot = random.uniform(-12, 12)
        glyph_img = glyph_img.rotate(rot, resample=Image.BICUBIC, expand=True, fillcolor=255)

        y = int(base_y - gh / 2 + i * baseline_drift * glyph_h + random.uniform(-3, 3))
        if cursor_x + glyph_img.width > WORK_W - 2:
            break
        canvas.paste(glyph_img, (cursor_x, max(0, y)))
        cursor_x += int(glyph_img.width * random.uniform(0.55, 0.8))

    # Random ink-darkness variation per glyph is already baked in via EMNIST's own
    # anti-aliasing; add an overall darkening pass since EMNIST source ink is lighter
    # than typical pen ink.
    arr = np.asarray(canvas, dtype=np.float32)
    arr = np.clip(arr * random.uniform(0.75, 0.95), 0, 255)
    canvas = Image.fromarray(arr.astype(np.uint8))

    return jitter_canvas(canvas)


def build_split(n_printed: int, n_handwriting: int, glyphs: dict[int, list[np.ndarray]]):
    images = []
    labels = []
    for _ in range(n_printed):
        font_path = random.choice(PRINTED_FONTS)
        images.append(render_printed(font_path))
        labels.append(0)

    # A third of the "handwriting" budget is a printed word with a small hand-drawn mark
    # overlaid (circle/check/tick/scribble) instead of a pure EMNIST-composed word — teaches
    # the "grader circled a printed MCQ letter" case, which pure-class-only training never saw.
    n_overlay = n_handwriting // 3
    n_pure_handwriting = n_handwriting - n_overlay
    for _ in range(n_pure_handwriting):
        images.append(render_handwriting(glyphs))
        labels.append(1)
    for _ in range(n_overlay):
        font_path = random.choice(PRINTED_FONTS)
        images.append(render_printed(font_path, with_overlay_mark=True))
        labels.append(1)

    images = np.stack(images)[..., None]  # (N, H, W, 1)
    labels = np.array(labels, dtype=np.uint8)

    perm = np.random.permutation(len(labels))
    return images[perm], labels[perm]


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading EMNIST letters glyph bank...")
    glyphs = load_emnist_glyphs()
    print(f"  {sum(len(v) for v in glyphs.values())} glyphs across {len(glyphs)} labels")

    splits = {
        "train": (16000, 16000),
        "val": (2000, 2000),
        "test": (2000, 2000),
    }
    for name, (n_printed, n_handwriting) in splits.items():
        print(f"Generating {name}: {n_printed} printed + {n_handwriting} handwriting...")
        images, labels = build_split(n_printed, n_handwriting, glyphs)
        out_path = f"{name}.npz"
        np.savez_compressed(out_path, images=images, labels=labels)
        print(f"  saved {out_path} -> images {images.shape}, labels {labels.shape}")


if __name__ == "__main__":
    main()
