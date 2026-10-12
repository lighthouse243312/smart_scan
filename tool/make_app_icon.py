"""Smart Scan app icon: a white page inside scanner corner brackets, a glowing peach scan line
crossing it - crisp print above, handwriting dissolving into particles below - and a sparkle.
Drawn at 4096 px and downsampled. Writes iOS AppIcon set + Android legacy and adaptive icons."""
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

ROOT = sys.argv[1]
OUT_PREVIEW = sys.argv[2]
S = 4096

TEAL_DARK = (58, 136, 126)
TEAL = (78, 136, 128)
SAGE = (150, 210, 197)
INK = (47, 59, 58)
PEACH = (240, 166, 134)
PEACH_LIGHT = (255, 214, 192)
WHITE = (255, 255, 255)


def u(v):  # unit -> px
    return int(round(v * S))


def background():
    import numpy as np
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    t = np.clip((xx * 0.55 + yy * 0.75) / 1.3, 0, 1)[..., None]
    top, bottom = np.array(SAGE, np.float32), np.array(TEAL_DARK, np.float32)
    rgb = top * (1 - t) + bottom * t
    # soft light from the top-left
    d = np.sqrt((xx - 0.18) ** 2 + (yy - 0.12) ** 2)
    light = np.clip(1 - d / 0.75, 0, 1)[..., None] ** 2 * 0.3
    rgb = rgb * (1 - light) + 255 * light
    return Image.fromarray(rgb.astype(np.uint8), "RGB").convert("RGBA")


def rounded_line(d, x0, y0, x1, h, fill):
    d.rounded_rectangle([x0, y0 - h // 2, x1, y0 + h // 2], radius=h // 2, fill=fill)


def star(d, cx, cy, r, fill, pinch=0.62):
    pts = []
    for i in range(64):
        a = 2 * math.pi * i / 64
        # astroid-like 4-point sparkle
        k = (abs(math.cos(a)) ** pinch + abs(math.sin(a)) ** pinch) ** (-1 / pinch)
        pts.append((cx + r * k * math.cos(a) * 1.0, cy + r * k * math.sin(a)))
    d.polygon(pts, fill=fill)


def foreground(scale):
    """Everything but the background, centred, sized by [scale] (1 = full iOS composition)."""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    def t(v, axis_center=0.5):  # scale a unit coordinate about the centre
        return u(axis_center + (v - axis_center) * scale)

    # page geometry
    L, T, R, B = 0.30, 0.20, 0.70, 0.80
    rad = u(0.055 * scale)
    fold = 0.12

    # shadow under the page
    sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([t(L), t(T + 0.03), t(R), t(B + 0.035)], radius=rad, fill=(20, 70, 64, 105))
    sh = sh.filter(ImageFilter.GaussianBlur(u(0.03 * scale)))
    layer.alpha_composite(sh)

    # page with a folded top-right corner
    page = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    pd = ImageDraw.Draw(page)
    pd.rounded_rectangle([t(L), t(T), t(R), t(B)], radius=rad, fill=WHITE + (255,))
    # cut the corner
    pd.polygon([(t(R - fold), t(T) - 2), (t(R) + 2, t(T) - 2), (t(R) + 2, t(T + fold))], fill=(0, 0, 0, 0))
    layer.alpha_composite(page)
    d = ImageDraw.Draw(layer)
    # fold flap with a soft shade
    flap = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    fd = ImageDraw.Draw(flap)
    fd.polygon([(t(R - fold), t(T)), (t(R - fold), t(T + fold)), (t(R), t(T + fold))], fill=(10, 40, 38, 70))
    flap = flap.filter(ImageFilter.GaussianBlur(u(0.008 * scale)))
    layer.alpha_composite(flap)
    d.polygon([(t(R - fold), t(T)), (t(R - fold) + u(0.012 * scale), t(T + fold) - u(0.012 * scale)), (t(R), t(T + fold))],
              fill=(220, 236, 232, 255))

    # crisp printed lines above the scan line
    lh = u(0.028 * scale)
    for y, x1 in [(0.30, 0.565), (0.37, 0.645), (0.44, 0.61)]:
        rounded_line(d, t(0.355), t(y), t(x1), lh, INK + (255,))

    # below the scan line: handwriting dissolving into particles
    hw = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hw)
    pts = []
    for i in range(80):
        s = i / 79
        x = 0.355 + s * 0.17
        y = 0.645 + 0.035 * math.sin(s * math.pi * 2.2) - 0.01 * s
        pts.append((t(x), t(y)))
    hd.line(pts, fill=(201, 107, 92, 255), width=u(0.024 * scale), joint="curve")
    for (x, y) in [pts[0], pts[-1]]:
        r = u(0.012 * scale)
        hd.ellipse([x - r, y - r, x + r, y + r], fill=(201, 107, 92, 255))
    # fade the stroke out to the right
    import numpy as np
    xs = np.arange(S, dtype=np.float32)
    x0, x1 = t(0.44), t(0.53)
    ramp = np.clip(1 - (xs - x0) / max(1, x1 - x0), 0, 1)
    alpha = np.asarray(hw.getchannel("A"), np.float32) * ramp[None, :]
    hw.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    layer.alpha_composite(hw)
    for i, (x, y, r, a) in enumerate([(0.555, 0.632, 0.011, 200), (0.585, 0.618, 0.008, 160), (0.61, 0.640, 0.006, 120),
                                      (0.632, 0.622, 0.0045, 90), (0.65, 0.635, 0.003, 70)]):
        rr = u(r * scale)
        m = a / 255
        col = tuple(int(c * m + 255 * (1 - m)) for c in (201, 107, 92))
        d.ellipse([t(x) - rr, t(y) - rr, t(x) + rr, t(y) + rr], fill=col + (255,))
    # ghost of the next printed line
    rounded_line(d, t(0.355), t(0.73), t(0.58), lh, (220, 228, 226, 255))

    # glowing scan line across the page
    sy = 0.535
    glow = Image.new("RGBA", (S, S), PEACH_LIGHT + (0,))
    gd = ImageDraw.Draw(glow)
    gd.rounded_rectangle([t(0.25), t(sy) - u(0.018 * scale), t(0.75), t(sy) + u(0.018 * scale)],
                         radius=u(0.018 * scale), fill=PEACH_LIGHT + (150,))
    glow = glow.filter(ImageFilter.GaussianBlur(u(0.014 * scale)))
    layer.alpha_composite(glow)
    # light band trailing above the line (the part just scanned)
    band = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    for k in range(40):
        a = int(55 * (1 - k / 40) ** 2)
        y0 = t(sy) - u(0.004 * scale * k)
        bd.rectangle([t(L), y0 - u(0.004 * scale), t(R), y0], fill=PEACH_LIGHT + (a,))
    band_mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(band_mask).rounded_rectangle([t(L), t(T), t(R), t(B)], radius=rad, fill=255)
    band.putalpha(Image.composite(band.getchannel("A"), Image.new("L", (S, S), 0), band_mask))
    layer.alpha_composite(band)
    rounded_line(d, t(0.24), t(sy), t(0.76), u(0.014 * scale), PEACH + (255,))
    rounded_line(d, t(0.30), t(sy), t(0.70), u(0.005 * scale), PEACH_LIGHT + (255,))

    # scanner corner brackets
    bw = u(0.032 * scale)
    BL, BT, BR, BB = 0.215, 0.125, 0.785, 0.875
    arm = 0.115
    for (cx, cy, sx, sy2) in [(BL, BT, 1, 1), (BR, BT, -1, 1), (BL, BB, 1, -1), (BR, BB, -1, -1)]:
        r = u(0.04 * scale)  # rounded inner corner
        pts = [(t(cx + sx * arm), t(cy))]
        for k in range(17):
            ang = math.pi / 2 * k / 16
            pts.append((t(cx) + sx * (r - r * math.sin(ang)), t(cy) + sy2 * (r - r * math.cos(ang))))
        pts.append((t(cx), t(cy + sy2 * arm)))
        d.line(pts, fill=WHITE + (240,), width=bw, joint="curve")
        for (x, y) in (pts[0], pts[-1]):
            d.ellipse([x - bw // 2, y - bw // 2, x + bw // 2, y + bw // 2], fill=WHITE + (240,))

    # sparkle
    sp = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sp)
    star(sd, t(0.705), t(0.735), u(0.085 * scale), PEACH + (255,))
    star(sd, t(0.705), t(0.735), u(0.035 * scale), PEACH_LIGHT + (255,))
    star(sd, t(0.615), t(0.805), u(0.035 * scale), PEACH + (255,))
    halo_src = Image.new("RGBA", (S, S), PEACH + (0,))
    halo_src.putalpha(sp.getchannel("A").point(lambda v: v * 0.8))
    halo = halo_src.filter(ImageFilter.GaussianBlur(u(0.02 * scale)))
    layer.alpha_composite(halo)
    layer.alpha_composite(sp)
    return layer


def down(img, size):
    return img.resize((size, size), Image.LANCZOS)


bg = background()
full = bg.copy()
full.alpha_composite(foreground(1.0))
master = full.convert("RGB")
master.resize((1024, 1024), Image.LANCZOS).save(OUT_PREVIEW)

# iOS: opaque squares (iOS applies its own rounded mask)
ios_dir = os.path.join(ROOT, "ios/Runner/Assets.xcassets/AppIcon.appiconset")
for f in os.listdir(ios_dir):
    if f.endswith(".png"):
        n = Image.open(os.path.join(ios_dir, f)).size[0]
        down(master, n).save(os.path.join(ios_dir, f))

# Android legacy icons (rounded square) + adaptive layers (content within the 66% safe zone)
res = os.path.join(ROOT, "android/app/src/main/res")
legacy_mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(legacy_mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=u(0.22), fill=255)
legacy = Image.new("RGBA", (S, S), (0, 0, 0, 0))
legacy.paste(master, (0, 0), legacy_mask)
fg = foreground(0.62)
for dens, n in {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}.items():
    d = os.path.join(res, f"mipmap-{dens}")
    down(legacy, n).save(os.path.join(d, "ic_launcher.png"))
    a = n * 108 // 48  # adaptive layers are 108dp
    down(bg, a).save(os.path.join(d, "ic_launcher_background.png"))
    down(fg, a).save(os.path.join(d, "ic_launcher_foreground.png"))
print("done")
