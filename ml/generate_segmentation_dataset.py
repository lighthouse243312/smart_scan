"""Generates training data for a pixel-level "which ink is handwriting" segmentation model —
replaces the hand-tuned redness+dilate+connected-component erase-region heuristic, which
still missed parts of circles/ticks and couldn't generalize beyond red ink.

Each sample is a synthetic scene (1-2 printed words, optionally with a hand-drawn annotation
mark or a handwritten fill-in word placed near them) at SEG_SIZE x SEG_SIZE, with a matching
ground-truth mask: 1 where a pixel belongs to handwriting-class ink (the overlay mark, or an
EMNIST-composed word), 0 everywhere else (printed ink, paper, marks are drawn identically onto
both the image and the mask so they stay pixel-aligned through rotation).

Output: seg_train.npz / seg_val.npz / seg_test.npz, each holding `images` (N, SEG_SIZE,
SEG_SIZE, 1) uint8 and `masks` (N, SEG_SIZE, SEG_SIZE, 1) uint8 (0 or 255).
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import generate_dataset as gd

SEG_SIZE = 128
SEED = 4321


def _draw_mark_on_both(image_draw, mask_draw, cx, cy, w, h):
    """Re-implements generate_dataset.draw_hand_mark's shape choice/geometry so the exact same
    stroke can be drawn onto the image canvas (visible, jittered ink darkness) and the mask
    canvas (solid white — this IS the erase target) without letting the two draws diverge."""
    kind = random.choice(["circle", "check", "tick", "underline_scribble"])
    width = random.randint(2, 4)
    image_ink = random.randint(0, 60)

    if kind == "circle":
        rx, ry = w * random.uniform(0.55, 0.75), h * random.uniform(0.55, 0.75)
        n = 14
        pts = []
        for i in range(n + 1):
            t = i / n * 2 * 3.14159
            jitter = random.uniform(-2, 2)
            pts.append((cx + (rx + jitter) * np.cos(t), cy + (ry + jitter) * np.sin(t)))
    elif kind == "check":
        pts = [(cx - w * 0.4, cy), (cx - w * 0.1, cy + h * 0.4), (cx + w * 0.45, cy - h * 0.45)]
    elif kind == "tick":
        pts = [(cx - w * 0.3, cy + h * 0.3), (cx + w * 0.3, cy - h * 0.3)]
    else:
        n = 6
        y0 = cy + h * 0.5
        pts = [(cx - w * 0.5 + i * w / n, y0 + random.uniform(-2, 2)) for i in range(n + 1)]

    image_draw.line(pts, fill=image_ink, width=width, joint="curve")
    mask_draw.line(pts, fill=255, width=width + 2, joint="curve")  # +2: mask slightly generous so thin AA edges are still covered


def _place_word(draw, work_w, work_h, avoid_rect=None):
    text = gd.random_word()
    size = random.randint(18, 30)
    font = ImageFont.truetype(random.choice(gd.PRINTED_FONTS), size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    for _ in range(20):
        x = random.randint(0, max(1, work_w - tw - 4)) - bbox[0]
        y = random.randint(0, max(1, work_h - th - 4)) - bbox[1]
        rect = (x + bbox[0], y + bbox[1], x + bbox[2], y + bbox[3])
        if avoid_rect is None or not _overlaps(rect, avoid_rect):
            return text, font, x, y, rect
    return text, font, x, y, rect


def _overlaps(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def render_scene(glyphs: dict[int, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    image = Image.new("L", (SEG_SIZE, SEG_SIZE), color=255)
    mask = Image.new("L", (SEG_SIZE, SEG_SIZE), color=0)
    idraw = ImageDraw.Draw(image)
    mdraw = ImageDraw.Draw(mask)

    n_words = random.randint(1, 3)
    placed_rects = []
    word_rects = []
    for _ in range(n_words):
        avoid = random.choice(placed_rects) if placed_rects else None
        text, font, x, y, rect = _place_word(idraw, SEG_SIZE, SEG_SIZE, avoid)
        ink = random.randint(0, 40)
        idraw.text((x, y), text, font=font, fill=ink)
        placed_rects.append(rect)
        word_rects.append(rect)

    if random.random() < 0.7 and word_rects:
        target = random.choice(word_rects)
        span = max(1, target[2] - target[0])
        span_h = max(1, target[3] - target[1])
        if random.random() < 0.6:
            # overlay mark centered on (part of) a random printed word
            focus_w = span * random.uniform(0.3, 0.6)
            focus_x = target[0] + random.uniform(0, max(1, span - focus_w)) + focus_w / 2
            cy = (target[1] + target[3]) / 2
            _draw_mark_on_both(idraw, mdraw, focus_x, cy, max(focus_w, 14), max(span_h, 14))
        else:
            # a pure handwritten word nearby (fill-in-blank style)
            hw_canvas = Image.new("L", (SEG_SIZE, SEG_SIZE), color=255)
            hw_arr_before = np.asarray(hw_canvas)
            _paste_handwriting(hw_canvas, glyphs, near_rect=target)
            hw_arr = np.asarray(hw_canvas)
            ink_pixels = hw_arr < 200
            image_arr = np.asarray(image).copy()
            image_arr[ink_pixels] = np.minimum(image_arr[ink_pixels], hw_arr[ink_pixels])
            image = Image.fromarray(image_arr)
            mask_arr = np.asarray(mask).copy()
            mask_arr[ink_pixels] = 255
            mask = Image.fromarray(mask_arr)

    angle = random.uniform(-4, 4)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    mask = mask.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)

    if random.random() < 0.5:
        scale = random.uniform(0.5, 0.95)
        sw, sh = max(8, int(SEG_SIZE * scale)), max(8, int(SEG_SIZE * scale))
        image = image.resize((sw, sh), Image.BILINEAR).resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)

    if random.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))

    arr = np.asarray(image, dtype=np.float32)
    contrast = random.uniform(0.75, 1.25)
    brightness = random.uniform(-20, 15)
    arr = (arr - 128.0) * contrast + 128.0 + brightness
    arr = arr + np.random.normal(0, random.uniform(1.0, 6.0), arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    mask_arr = (np.asarray(mask) > 127).astype(np.uint8) * 255
    return arr, mask_arr


def _paste_handwriting(canvas: Image.Image, glyphs: dict[int, list[np.ndarray]], near_rect):
    draw_x = near_rect[2] + random.randint(4, 20)
    draw_y = (near_rect[1] + near_rect[3]) // 2
    n_chars = random.randint(2, 5)
    glyph_h = random.randint(16, 26)
    cursor_x = draw_x
    for i in range(n_chars):
        label = random.randint(1, 26)
        bank = glyphs.get(label) or glyphs[random.choice(list(glyphs.keys()))]
        glyph = random.choice(bank)
        glyph_img = Image.fromarray(255 - glyph)
        scale = glyph_h / 28.0 * random.uniform(0.85, 1.15)
        gw, gh = max(4, int(28 * scale)), max(4, int(28 * scale))
        glyph_img = glyph_img.resize((gw, gh), resample=Image.BICUBIC)
        rot = random.uniform(-10, 10)
        glyph_img = glyph_img.rotate(rot, resample=Image.BICUBIC, expand=True, fillcolor=255)
        y = draw_y - gh // 2 + random.randint(-3, 3)
        if cursor_x + glyph_img.width > SEG_SIZE - 2 or y < 0 or y + glyph_img.height > SEG_SIZE:
            break
        canvas.paste(glyph_img, (cursor_x, max(0, y)))
        cursor_x += int(glyph_img.width * random.uniform(0.55, 0.8))


def build_split(n: int, glyphs) -> tuple[np.ndarray, np.ndarray]:
    images, masks = [], []
    for _ in range(n):
        img, msk = render_scene(glyphs)
        images.append(img)
        masks.append(msk)
    images = np.stack(images)[..., None]
    masks = np.stack(masks)[..., None]
    return images, masks


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading EMNIST letters glyph bank...")
    glyphs = gd.load_emnist_glyphs()
    print(f"  {sum(len(v) for v in glyphs.values())} glyphs across {len(glyphs)} labels")

    splits = {"train": 12000, "val": 1500, "test": 1500}
    for name, n in splits.items():
        print(f"Generating seg_{name}: {n} scenes...")
        images, masks = build_split(n, glyphs)
        out_path = f"seg_{name}.npz"
        np.savez_compressed(out_path, images=images, masks=masks)
        print(f"  saved {out_path} -> images {images.shape}, masks {masks.shape}")


if __name__ == "__main__":
    main()
