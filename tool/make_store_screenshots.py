"""Store screenshots for Google Play and the App Store, composed from real app screenshots.

Usage: ml/venv/bin/python tool/make_store_screenshots.py <shots dir> <out dir> [lang ...]
<shots dir>/<lang>/ holds full-resolution device screenshots (app set to that language) named in
SLIDES below; output goes to <out dir>/<lang>/<size>/. Languages: vi en ja ko (default: all).
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SHOTS_ROOT, OUT_ROOT = sys.argv[1], sys.argv[2]
LANGS = sys.argv[3:] or ["vi", "en", "ja", "ko"]
SHOTS = SHOTS_ROOT  # set per language below
FONT = "/System/Library/Fonts/SFNS.ttf"
JA_FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W{}.ttc"
KO_FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
LANG = "vi"

INK = (40, 56, 54)
MUTED = (96, 116, 112)
TEAL = (78, 136, 128)
PEACH = (232, 150, 118)

# name -> (width, height, device frame)
SIZES = {
    "app_store_iphone_6.9": (1320, 2868, "iphone"),
    "app_store_iphone_6.5": (1284, 2778, "iphone"),
    "app_store_iphone_6.3": (1206, 2622, "iphone"),
    "app_store_ipad_13": (2064, 2752, "iphone"),
    "google_play": (1080, 1920, "android"),
}

# (key, screenshot, title, subtitle) per language; "hero" is the before / after slide
SLIDES_BY_LANG = {
    "vi": [
        ("hero", None, "Xoá chữ viết tay\nchỉ một chạm", "Trả lại trang giấy sạch như mới"),
        ("detect", "s3_detect.png", "AI nhận diện\nnét bút chính xác", "Tự phân biệt chữ viết tay và chữ in"),
        ("process", "s2_process.png", "Làm nét, xoá bóng\ntự động", "Ảnh chụp thành bản scan rõ ràng"),
        ("home", "s1_home.png", "Quét, chọn ảnh\nhoặc nhập PDF", "Tự nhận khung và cắt trang"),
        ("export", "s4_export.png", "Xuất PDF, Word, PNG\nchia sẻ tức thì", "Lưu lại cả bản gốc và bản đã xử lý"),
    ],
    "en": [
        ("hero", None, "Erase handwriting\nin one tap", "Get back a clean, fresh page"),
        ("detect", "s3_detect.png", "AI spots every\npen stroke", "Tells handwriting apart from print"),
        ("process", "s2_process.png", "Sharpen & remove\nshadows, automatically", "Photos become crisp scans"),
        ("home", "s1_home.png", "Scan, pick photos\nor import a PDF", "Edges found and cropped for you"),
        ("export", "s4_export.png", "Export PDF, Word, PNG\nshare instantly", "Keeps both the original and the clean copy"),
    ],
    "ja": [
        ("hero", None, "手書き文字を\nワンタップで消去", "まっさらなページがよみがえる"),
        ("detect", "s3_detect.png", "AIがペン跡を\n正確に検出", "手書きと印刷文字を自動で区別"),
        ("process", "s2_process.png", "鮮明化と影の除去を\n自動で", "写真がくっきりスキャンに"),
        ("home", "s1_home.png", "スキャン・写真・\nPDFから取り込み", "枠を自動検出して切り抜き"),
        ("export", "s4_export.png", "PDF・Word・PNGで\nすぐに共有", "元の画像も処理済みも保存"),
    ],
    "ko": [
        ("hero", None, "손글씨를\n한 번에 지우기", "깨끗한 새 페이지로 되돌려요"),
        ("detect", "s3_detect.png", "AI가 펜 자국을\n정확하게 찾아요", "손글씨와 인쇄 글자를 자동 구분"),
        ("process", "s2_process.png", "선명하게,\n그림자는 자동 제거", "사진이 깔끔한 스캔본으로"),
        ("home", "s1_home.png", "스캔, 사진 선택,\nPDF 가져오기", "테두리를 찾아 자동으로 자르기"),
        ("export", "s4_export.png", "PDF, Word, PNG로\n바로 공유", "원본과 처리본을 모두 저장"),
    ],
}
BEFORE_AFTER = {"vi": ("Trước", "Sau"), "en": ("Before", "After"), "ja": ("ビフォー", "アフター"), "ko": ("전", "후")}


def font(size, weight="Bold"):
    if LANG == "ja":
        w = {"Bold": 7, "Semibold": 6, "Medium": 5}.get(weight, 5)
        return ImageFont.truetype(JA_FONT.format(w), size)
    if LANG == "ko":
        return ImageFont.truetype(KO_FONT, size, index={"Bold": 6, "Semibold": 4, "Medium": 2}.get(weight, 2))
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_name(weight)
    return f


def background(W, H, flip=False):
    import numpy as np
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.clip(yy / H * 0.85 + (xx / W if not flip else 1 - xx / W) * 0.15, 0, 1)[..., None]
    top, bottom = np.array((236, 246, 242), np.float32), np.array((176, 222, 211), np.float32)
    img = Image.fromarray((top * (1 - t) + bottom * t).astype(np.uint8), "RGB").convert("RGBA")
    # soft blobs
    blob = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(blob)
    bx = W * (0.85 if not flip else 0.15)
    d.ellipse([bx - W * 0.45, H * 0.32 - W * 0.45, bx + W * 0.45, H * 0.32 + W * 0.45], fill=(255, 226, 210, 120))
    d.ellipse([W * 0.5 - W * 0.6, H * 0.95 - W * 0.4, W * 0.5 + W * 0.6, H * 0.95 + W * 0.4], fill=(120, 184, 172, 90))
    img.alpha_composite(blob.filter(ImageFilter.GaussianBlur(W * 0.12)))
    return img


def clean_status_bar(shot):
    """Replaces the device status bar with a neutral one (9:41, signal, wifi, battery)."""
    shot = shot.convert("RGB").copy()
    w, h = shot.size
    bar = int(h * 0.036)
    bg = shot.getpixel((w // 2, bar + 6))
    d = ImageDraw.Draw(shot)
    d.rectangle([0, 0, w, bar], fill=bg)
    s = bar / 120
    d.text((int(70 * s), bar // 2), "9:41", font=font(int(46 * s), "Semibold"), fill=INK, anchor="lm")
    # battery
    bx, by = w - int(130 * s), bar // 2
    d.rounded_rectangle([bx, by - int(18 * s), bx + int(66 * s), by + int(18 * s)], radius=int(8 * s), outline=INK, width=max(2, int(4 * s)))
    d.rounded_rectangle([bx + int(7 * s), by - int(11 * s), bx + int(56 * s), by + int(11 * s)], radius=int(4 * s), fill=INK)
    d.rectangle([bx + int(68 * s), by - int(7 * s), bx + int(74 * s), by + int(7 * s)], fill=INK)
    # signal bars
    sx = bx - int(150 * s)
    for i in range(4):
        hh = int((10 + i * 8) * s)
        d.rounded_rectangle([sx + i * int(14 * s), by + int(16 * s) - hh, sx + i * int(14 * s) + int(9 * s), by + int(16 * s)], radius=int(2 * s), fill=INK)
    # wifi
    wx, wy = bx - int(50 * s), by + int(14 * s)
    for r in (34, 22, 10):
        rr = int(r * s)
        d.arc([wx - rr, wy - rr, wx + rr, wy + rr], start=225, end=315, fill=INK, width=max(2, int(6 * s)))
    return shot


def phone(shot, width, style="android"):
    """Device frame around [shot] at [width] px: Android (punch-hole camera) or iPhone (Dynamic
    Island, thinner bezel)."""
    sw, sh = shot.size
    bezel = int(width * (0.035 if style == "android" else 0.028))
    inner_w = width - 2 * bezel
    inner_h = int(sh * inner_w / sw)
    H = inner_h + 2 * bezel
    r_out, r_in = int(width * 0.13), int(width * 0.105)
    body = Image.new("RGBA", (width, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(body)
    d.rounded_rectangle([0, 0, width - 1, H - 1], radius=r_out, fill=(28, 36, 35, 255))
    d.rounded_rectangle([3, 3, width - 4, H - 4], radius=r_out - 3, outline=(70, 82, 80, 255), width=3)
    screen = shot.resize((inner_w, inner_h), Image.LANCZOS).convert("RGBA")
    mask = Image.new("L", (inner_w, inner_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, inner_w - 1, inner_h - 1], radius=r_in, fill=255)
    body.paste(screen, (bezel, bezel), mask)
    # camera punch-hole
    if style == "android":
        cx, cy, cr = width // 2, bezel + int(inner_h * 0.016), int(width * 0.018)
        d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(16, 20, 20, 255))
    else:
        # inside the status bar (3.6% of the screen), between the clock and the icons
        bar = int(inner_h * 0.036)
        ih = int(bar * 0.72)
        iw = int(ih * 3.3)
        iy = bezel + (bar - ih) // 2
        d.rounded_rectangle([width // 2 - iw // 2, iy, width // 2 + iw // 2, iy + ih], radius=ih // 2, fill=(8, 10, 10, 255))
        # side buttons
        d.rounded_rectangle([width - 4, int(H * 0.2), width + 2, int(H * 0.3)], radius=3, fill=(40, 50, 49, 255))
    return body


def shadow(layer, blur, offset, alpha=90):
    sh = Image.new("RGBA", layer.size, (20, 60, 55, 0))
    sh.putalpha(layer.getchannel("A").point(lambda v: v * alpha // 255))
    pad = blur * 3
    big = Image.new("RGBA", (layer.width + 2 * pad, layer.height + 2 * pad), (20, 60, 55, 0))
    big.paste(sh, (pad, pad + offset))
    return big.filter(ImageFilter.GaussianBlur(blur)), pad


def place(canvas, layer, x, y, blur, offset):
    sh, pad = shadow(layer, blur, offset)
    canvas.alpha_composite(sh, (x - pad, y - pad)) if x - pad >= 0 and y - pad >= 0 else canvas.paste(sh, (x - pad, y - pad), sh)
    canvas.paste(layer, (x, y), layer)


def page_crop(path, box):
    im = Image.open(os.path.join(SHOTS, path)).convert("RGB")
    w, h = im.size
    return im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))


def card(img, width, label, color):
    img = img.resize((width, int(img.height * width / img.width)), Image.LANCZOS)
    pad = int(width * 0.035)
    c = Image.new("RGBA", (width + 2 * pad, img.height + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(c)
    d.rounded_rectangle([0, 0, c.width - 1, c.height - 1], radius=int(width * 0.05), fill=(255, 255, 255, 255))
    c.paste(img, (pad, pad))
    f = font(int(width * 0.075), "Bold")
    tw = d.textlength(label, font=f)
    ph, pw = int(width * 0.12), int(tw + width * 0.1)
    d.rounded_rectangle([pad * 2, pad * 2, pad * 2 + pw, pad * 2 + ph], radius=ph // 2, fill=color)
    d.text((pad * 2 + pw // 2, pad * 2 + ph // 2), label, font=f, fill=(255, 255, 255), anchor="mm")
    return c


def caption(canvas, W, H, title, sub):
    d = ImageDraw.Draw(canvas)
    U = min(W, H * 0.46)  # type scale: canvas width on phones, limited on wide canvases
    tf = font(int(U * 0.082), "Bold")
    sf = font(int(U * 0.04), "Medium")
    y = int(H * 0.065)
    for line in title.split("\n"):
        d.text((W // 2, y), line, font=tf, fill=INK, anchor="mt")
        y += int(U * 0.1)
    y += int(U * 0.015)
    d.text((W // 2, y), sub, font=sf, fill=MUTED, anchor="mt")
    # accent dash
    d.rounded_rectangle([W // 2 - int(U * 0.04), int(H * 0.04), W // 2 + int(U * 0.04), int(H * 0.04) + int(U * 0.012)],
                        radius=int(U * 0.006), fill=PEACH)
    return y + int(U * 0.06)


def render(slide, W, H, idx, style):
    key, shot, title, sub = slide
    canvas = background(W, H, flip=idx % 2 == 1)
    top = caption(canvas, W, H, title, sub)
    if key == "hero":
        before = page_crop("s2_process.png", (0.09, 0.13, 0.925, 0.69))
        after = page_crop("s4_export.png", (0.1, 0.11, 0.9, 0.505))
        cw = int(min(W * 0.62, H * 0.3))
        b = card(before, cw, BEFORE_AFTER[LANG][0], (201, 107, 92)).rotate(4, expand=True, resample=Image.BICUBIC)
        a = card(after, cw, BEFORE_AFTER[LANG][1], TEAL).rotate(-3, expand=True, resample=Image.BICUBIC)
        room = H - top - int(H * 0.04)
        yb = top + int(H * 0.02)
        ya = max(yb + int(room * 0.36), H - a.height - int(H * 0.05))
        place(canvas, b, int(W * 0.04), yb, int(W * 0.03), int(W * 0.02))
        place(canvas, a, W - a.width - int(W * 0.04), ya, int(W * 0.035), int(W * 0.025))
    else:
        s = clean_status_bar(Image.open(os.path.join(SHOTS, shot)))
        avail = H - top - int(H * 0.03)
        pw = min(int(W * (0.74 if H / W > 2 else 0.66)), int(avail / (s.height / s.width) * 0.94))
        p = phone(s, pw, style)
        place(canvas, p, (W - pw) // 2, top + int(H * 0.025), int(W * 0.04), int(W * 0.03))
    return canvas.convert("RGB")


for LANG in LANGS:
    SHOTS = os.path.join(SHOTS_ROOT, LANG)
    for name, (W, H, style) in SIZES.items():
        d = os.path.join(OUT_ROOT, LANG, name)
        os.makedirs(d, exist_ok=True)
        for i, slide in enumerate(SLIDES_BY_LANG[LANG]):
            render(slide, W, H, i, style).save(os.path.join(d, f"{i + 1:02d}_{slide[0]}.png"))
    print("done", LANG)
