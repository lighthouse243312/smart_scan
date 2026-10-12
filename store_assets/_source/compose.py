"""Store screenshots + Play feature graphic from the raw device captures."""
import os, sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
FONT = '/System/Library/Fonts/SFNS.ttf'

def font(size, weight='Bold'):
    f = ImageFont.truetype(FONT, size); f.set_variation_by_name(weight); return f

def c(h): h = h.lstrip('#'); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
TOP, BOT, PEACH, IVORY, INK = c('6E9F98'), c('3E6963'), c('F4C2AD'), c('F8F6F2'), c('2F3B3A')

COPY = {
    'en': {
        'hero': ('Erase handwriting', 'in one tap'),
        'review': ('AI finds the handwriting', 'Fix the mask with a brush if needed'),
        'process': ('Sharp, shadow-free pages', 'Cleaned up automatically'),
        'export': ('Export to PDF, Word or PNG', 'Every page at once'),
        'home': ('Scan, pick photos', 'or import a PDF'),
        'before': 'Before', 'after': 'After',
        'tagline': 'Scan. Clean. Erase handwriting.',
    },
    'vi': {
        'hero': ('Xoá chữ viết tay', 'chỉ với một chạm'),
        'review': ('AI tự tìm chữ viết tay', 'Tô sửa vùng xoá nếu cần'),
        'process': ('Trang rõ nét, hết bóng', 'Tự động làm sạch'),
        'export': ('Xuất PDF, Word hoặc PNG', 'Tất cả các trang một lần'),
        'home': ('Quét, chọn ảnh', 'hoặc nhập file PDF'),
        'before': 'Trước', 'after': 'Sau',
        'tagline': 'Quét. Làm sạch. Xoá chữ viết tay.',
    },
}

def gradient(w, h):
    g = Image.new('RGB', (1, 256))
    for y in range(256):
        t = y / 255; g.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOT)))
    return g.resize((w, h), Image.BICUBIC)

def rounded(im, r):
    m = Image.new('L', (im.width * 2, im.height * 2), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, m.width - 1, m.height - 1], radius=r * 2, fill=255)
    im = im.convert('RGBA'); im.putalpha(m.resize(im.size, Image.LANCZOS)); return im

def paste_card(canvas, card, x, y, r, shadow=True):
    card = rounded(card, r)
    if shadow:
        sh = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([x, y + r // 3, x + card.width, y + card.height + r // 3], radius=r, fill=(10, 30, 28, 110))
        canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(r // 2)))
    canvas.alpha_composite(card, (x, y))

def fit_text(draw, text, size, maxw, weight='Bold'):
    while size > 10:
        f = font(size, weight)
        if draw.textlength(text, font=f) <= maxw: return f
        size -= 2
    return font(size, weight)

def captions(canvas, title, sub, W, top):
    d = ImageDraw.Draw(canvas)
    ft = fit_text(d, title, int(W * 0.078), W * 0.88)
    d.text((W / 2, top), title, font=ft, fill='white', anchor='ma')
    fs = fit_text(d, sub, int(W * 0.045), W * 0.86, 'Medium')
    d.text((W / 2, top + ft.size * 1.25), sub, font=fs, fill=PEACH, anchor='ma')
    return top + ft.size * 1.25 + fs.size * 1.6

def crop_bars(raw, tablet):
    im = Image.open(raw).convert('RGB')
    top, bot = (112, 60) if tablet else (140, 100)
    return im.crop((0, top, im.width, im.height - bot))

def before_after(w, h, lang):
    """The worksheet before / after the erase (the app's own output), split down the middle."""
    before = Image.open(f'{S}/real_clean.png').convert('RGB')
    after = Image.open(f'{S}/real_final_final.png').convert('RGB')
    # the sheet itself: off the table edge / fingers around it
    box = (int(before.width * 0.08), int(before.height * 0.04), int(before.width * 0.93), int(before.height * 0.86))
    before, after = before.crop(box), after.crop(box)
    s = min(w / before.width, h / before.height)
    size = (round(before.width * s), round(before.height * s))
    before, after = before.resize(size, Image.LANCZOS), after.resize(size, Image.LANCZOS)
    page = before.copy(); half = size[0] // 2
    page.paste(after.crop((half, 0, size[0], size[1])), (half, 0))
    d = ImageDraw.Draw(page)
    lw = max(4, size[0] // 160)
    d.rectangle([half - lw // 2, 0, half + lw // 2, size[1]], fill='white')
    rr = size[0] // 22; cy = size[1] // 2
    d.ellipse([half - rr, cy - rr, half + rr, cy + rr], fill='white', outline=TOP, width=lw)
    d.polygon([(half - rr * 0.55, cy), (half - rr * 0.15, cy - rr * 0.35), (half - rr * 0.15, cy + rr * 0.35)], fill=TOP)
    d.polygon([(half + rr * 0.55, cy), (half + rr * 0.15, cy - rr * 0.35), (half + rr * 0.15, cy + rr * 0.35)], fill=TOP)
    fl = font(max(20, size[0] // 24), 'Semibold')
    for x, label, anchor in [(size[0] * 0.03, COPY[lang]['before'], 'la'), (size[0] * 0.97, COPY[lang]['after'], 'ra')]:
        bb = d.textbbox((x, size[1] * 0.025), label, font=fl, anchor=anchor)
        pad = fl.size * 0.4
        d.rounded_rectangle([bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad], radius=pad * 1.5, fill=INK)
        d.text((x, size[1] * 0.025), label, font=fl, fill='white', anchor=anchor)
    return page

def shot(W, H, lang, key, raw, tablet):
    canvas = gradient(W, H).convert('RGBA')
    title, sub = COPY[lang][key]
    y = captions(canvas, title, sub, W, int(H * 0.055))
    margin = int(W * (0.10 if tablet else 0.085))
    if key == 'hero':
        hm = int(W * 0.05)
        page = before_after(W - 2 * hm, H - y - int(H * 0.04), lang)
        paste_card(canvas, page, (W - page.width) // 2, int(y + (H - y - page.height) * 0.22), max(12, W // 60))
    else:
        im = crop_bars(raw, tablet)
        cw = W - 2 * margin
        im = im.resize((cw, round(im.height * cw / im.width)), Image.LANCZOS)
        r = int(W * 0.045)
        top = int(y + H * 0.01)
        # runs off the bottom edge like a phone rising out of the frame
        if top + im.height > H: im = im.crop((0, 0, im.width, H - top + r))
        paste_card(canvas, im, margin, top, r)
    return canvas.convert('RGB')

SETS = [
    # (folder, size, tablet)
    ('appstore_iphone_6.9', (1320, 2868), False),
    ('appstore_ipad_13', (2064, 2752), True),
    ('googleplay_phone', (1080, 1920), False),
]
ORDER = ['hero', 'review', 'process', 'export', 'home']

for lang in ('en', 'vi'):
    for folder, (W, H), tablet in SETS:
        d = os.path.join(OUT, 'screenshots', lang, folder); os.makedirs(d, exist_ok=True)
        # the tablet home shows recent documents (a third-party flyer among them): left out
        for i, key in enumerate([k for k in ORDER if not (tablet and k == 'home')], 1):
            raw = f'{S}/raw_{key}_{lang}{"_tab" if tablet else ""}.png'
            shot(W, H, lang, key, raw, tablet).save(os.path.join(d, f'{i:02d}_{key}.png'))

# Google Play feature graphic 1024x500
for lang in ('en', 'vi'):
    W, H = 1024, 500
    g = gradient(W, H).convert('RGBA')
    icon = Image.open(os.path.join(OUT, 'appstore_icon_1024.png')).resize((150, 150), Image.LANCZOS)
    paste_card(g, icon, 60, 90, 34)
    d = ImageDraw.Draw(g)
    d.text((60, 270), 'Smart Scan', font=font(64), fill='white')
    d.text((60, 352), COPY[lang]['tagline'], font=fit_text(d, COPY[lang]['tagline'], 30, 470, 'Medium'), fill=PEACH)
    page = before_after(400, 440, lang)
    paste_card(g, page, W - page.width - 50, (H - page.height) // 2 + 10, 14)
    g.convert('RGB').save(os.path.join(OUT, f'googleplay_feature_graphic_{lang}.png'))
print('ok')
