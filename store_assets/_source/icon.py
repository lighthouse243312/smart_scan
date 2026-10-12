"""Smart Scan app icon: a clean page whose handwriting turns to sparkle, on sage teal."""
import math, sys
from PIL import Image, ImageDraw

K = 4  # supersampling
N = 1024 * K

def c(h):
    h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

TEAL_TOP, TEAL_BOT = c('6E9F98'), c('3E6963')
INK, PEACH, FOLD, PEN = c('2F3B3A'), c('EBA084'), c('DCEBE7'), c('C96B5C')

def background(size):
    img = Image.new('RGB', (size, size))
    d = ImageDraw.Draw(img)
    for y in range(size):
        t = y / (size - 1)
        d.line([(0, y), (size, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(TEAL_TOP, TEAL_BOT)))
    return img

def star(d, cx, cy, r, fill, pinch=0.16):
    """A 4-point sparkle: tips joined by quadratic curves pulled in towards the centre."""
    tips = [(r, 0), (0, -r), (-r, 0), (0, r)]
    pts = []
    for k in range(4):
        (ax, ay), (bx, by) = tips[k], tips[(k + 1) % 4]
        qx, qy = (ax + bx) * pinch, (ay + by) * pinch
        for i in range(60):
            t = i / 60
            x = (1 - t) ** 2 * ax + 2 * (1 - t) * t * qx + t ** 2 * bx
            y = (1 - t) ** 2 * ay + 2 * (1 - t) * t * qy + t ** 2 * by
            pts.append((cx + x, cy + y))
    d.polygon(pts, fill=fill)

def foreground(scale=1.0, offset=(0, 0)):
    """Page + marks on a transparent layer, in 1024-unit coordinates scaled by `scale`."""
    layer = Image.new('RGBA', (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    u = lambda v: v * K * scale
    ox, oy = offset[0] * K, offset[1] * K
    P = lambda x, y: (ox + u(x - 512) + N / 2, oy + u(y - 512) + N / 2)
    # page with a folded top-right corner
    x0, y0, x1, y1, r, f = 268, 196, 756, 828, 56, 132
    shadow = Image.new('RGBA', (N, N), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([P(x0, y0 + 18), P(x1, y1 + 18)], radius=u(r), fill=(20, 45, 42, 70))
    from PIL import ImageFilter
    shadow = shadow.filter(ImageFilter.GaussianBlur(u(22)))
    layer.alpha_composite(shadow)
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([P(x0, y0), P(x1, y1)], radius=u(r), fill=(255, 255, 255, 255))
    d.polygon([P(x1 - f, y0 - 1), P(x1 + 1, y0 - 1), P(x1 + 1, y0 + f)], fill=(0, 0, 0, 0))
    # punch the corner out (polygon above only paints transparent pixels onto RGBA: do it properly)
    mask = Image.new('L', (N, N), 0)
    ImageDraw.Draw(mask).polygon([P(x1 - f, y0 - 2), P(x1 + 2, y0 - 2), P(x1 + 2, y0 + f)], fill=255)
    layer.putalpha(Image.composite(Image.new('L', (N, N), 0), layer.getchannel('A'), mask))
    d = ImageDraw.Draw(layer)
    d.polygon([P(x1 - f, y0), P(x1 - f, y0 + f - 18), P(x1 - f + 18, y0 + f), P(x1, y0 + f)], fill=FOLD + (255,))
    # printed lines
    w = 30
    for y, xe in [(318, 690), (394, 640), (470, 690)]:
        d.line([P(334, y), P(xe, y)], fill=INK + (255,), width=round(u(w)))
        for x in (334, xe):
            cx, cy = P(x, y); d.ellipse([cx - u(w / 2), cy - u(w / 2), cx + u(w / 2), cy + u(w / 2)], fill=INK + (255,))
    # handwriting: a pen squiggle that dissolves into dots, then sparkle
    pts = []
    for i in range(0, 61):
        t = i / 60
        x = 334 + t * 230
        y = 600 + 26 * math.sin(t * math.pi * 3.2) - 18 * t
        pts.append((x, y))
    for i in range(len(pts) - 1):
        t = i / (len(pts) - 1)
        if t > 0.62: break
        d.line([P(*pts[i]), P(*pts[i + 1])], fill=PEN + (255,), width=round(u(30)))
        cx, cy = P(*pts[i]); d.ellipse([cx - u(15), cy - u(15), cx + u(15), cy + u(15)], fill=PEN + (255,))
    # ...dissolving: smaller, paler dots (tints, opaque — alpha would show the teal through)
    for t, rad, mix in [(0.71, 12, 0.35), (0.80, 9, 0.55), (0.88, 6, 0.72)]:
        x, y = pts[round(t * 60)]
        col = tuple(round(a + (255 - a) * mix) for a in PEN) + (255,)
        cx, cy = P(x, y); d.ellipse([cx - u(rad), cy - u(rad), cx + u(rad), cy + u(rad)], fill=col)
    # sparkle
    sx, sy = P(690, 700); star(d, sx, sy, u(150), PEACH + (255,))
    sx, sy = P(566, 760); star(d, sx, sy, u(52), PEACH + (255,))
    return layer

def full_icon(size, rounded=False):
    img = background(N).convert('RGBA')
    img.alpha_composite(foreground())
    img = img.resize((size, size), Image.LANCZOS)
    if rounded:
        m = Image.new('L', (size * 4, size * 4), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, size * 4 - 1, size * 4 - 1], radius=size * 4 * 0.2237, fill=255)
        img.putalpha(m.resize((size, size), Image.LANCZOS))
    return img

if __name__ == "__main__" and len(sys.argv) == 2:
    out = sys.argv[1]
    full_icon(1024).convert('RGB').save(out + '/icon_1024.png')
    full_icon(512, rounded=True).save(out + '/preview_rounded.png')


def export_all(project, store):
    import json, os
    master = background(N).convert('RGBA'); master.alpha_composite(foreground())
    def sized(px):
        return master.resize((px, px), Image.LANCZOS)
    # iOS: every slot of the asset catalog, opaque (the App Store rejects an icon with alpha)
    cat = os.path.join(project, 'ios/Runner/Assets.xcassets/AppIcon.appiconset')
    for im in json.load(open(os.path.join(cat, 'Contents.json')))['images']:
        px = round(float(im['size'].split('x')[0]) * int(im['scale'][0]))
        sized(px).convert('RGB').save(os.path.join(cat, im['filename']))
    # Android legacy launcher icon (pre-8.0): rounded square with transparent corners
    res = os.path.join(project, 'android/app/src/main/res')
    dens = {'mdpi': 1, 'hdpi': 1.5, 'xhdpi': 2, 'xxhdpi': 3, 'xxxhdpi': 4}
    for name, f in dens.items():
        px = round(48 * f)
        big = sized(px * 4)
        m = Image.new('L', big.size, 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, big.size[0] - 1, big.size[1] - 1], radius=big.size[0] * 0.18, fill=255)
        big.putalpha(m)
        big.resize((px, px), Image.LANCZOS).save(os.path.join(res, f'mipmap-{name}', 'ic_launcher.png'))
    # Android 8+ adaptive icon: 108dp layers, artwork kept inside the 66dp safe circle
    fg = foreground(scale=0.70, offset=(-10, 0))
    bg = background(N).convert('RGBA')
    for name, f in dens.items():
        px = round(108 * f)
        fg.resize((px, px), Image.LANCZOS).save(os.path.join(res, f'mipmap-{name}', 'ic_launcher_foreground.png'))
        bg.resize((px, px), Image.LANCZOS).save(os.path.join(res, f'mipmap-{name}', 'ic_launcher_background.png'))
    os.makedirs(os.path.join(res, 'mipmap-anydpi-v26'), exist_ok=True)
    open(os.path.join(res, 'mipmap-anydpi-v26', 'ic_launcher.xml'), 'w').write(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n'
        '    <background android:drawable="@mipmap/ic_launcher_background" />\n'
        '    <foreground android:drawable="@mipmap/ic_launcher_foreground" />\n'
        '</adaptive-icon>\n')
    # stores: App Store 1024 (opaque, square — Apple rounds it), Play 512 (square — Play masks it)
    os.makedirs(store, exist_ok=True)
    sized(1024).convert('RGB').save(os.path.join(store, 'appstore_icon_1024.png'))
    sized(512).save(os.path.join(store, 'playstore_icon_512.png'))
    # previews of the adaptive icon under a circle mask (as on a Pixel launcher)
    prev = bg.copy(); prev.alpha_composite(fg); prev = prev.resize((432, 432), Image.LANCZOS)
    m = Image.new('L', (432, 432), 0); ImageDraw.Draw(m).ellipse([0, 0, 431, 431], fill=255); prev.putalpha(m)
    prev.save(os.path.join(os.path.dirname(store.rstrip('/')) if False else store, '_preview_adaptive_circle.png'))
if __name__ == "__main__" and len(sys.argv) == 3: export_all(sys.argv[1], sys.argv[2])
