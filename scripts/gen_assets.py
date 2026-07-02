"""Procedurally generate all Anima Forge UI art with Pillow. Idempotent.

Run standalone:  python3.10\\python.exe scripts\\gen_assets.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

GOLD = (212, 175, 55)
GOLD_HI = (244, 209, 96)
GOLD_DK = (138, 90, 18)
EMBER = (255, 122, 24)
SILVER = (198, 198, 206)
BLACK = (10, 10, 11)
PANEL = (20, 19, 18)

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _radial(size, inner, outer, center=None, radius=None, scale=4):
    """Smooth radial gradient: render small then upscale (fast + smooth)."""
    w, h = size
    sw, sh = max(1, w // scale), max(1, h // scale)
    cx = (sw / 2) if center is None else center[0] / scale
    cy = (sh / 2) if center is None else center[1] / scale
    rad = (max(sw, sh) / 2) if radius is None else radius / scale
    small = Image.new("RGB", (sw, sh))
    px = small.load()
    for y in range(sh):
        for x in range(sw):
            d = min(1.0, math.hypot(x - cx, y - cy) / rad)
            px[x, y] = _lerp(inner, outer, d)
    return small.resize((w, h), Image.BILINEAR)


def _add_sparks(img, count, color, seedy=12345):
    """Deterministic ember sparks (no randomness -> idempotent output)."""
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    s = seedy
    for _ in range(count):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        x = s % w
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        y = s % h
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        r = 1 + (s % 3)
        a = 120 + (s % 120)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color + (a,))
    return img


def _flame(draw, cx, base_y, h, w, color):
    """A pointed teardrop flame silhouette centered on cx, tip pointing up."""
    pts = [
        (cx, base_y - h),                       # tip
        (cx + w * 0.55, base_y - h * 0.55),
        (cx + w, base_y - h * 0.12),
        (cx + w * 0.6, base_y),                 # bottom-right shoulder
        (cx, base_y + h * 0.06),                # rounded base
        (cx - w * 0.6, base_y),
        (cx - w, base_y - h * 0.12),
        (cx - w * 0.55, base_y - h * 0.55),
    ]
    draw.polygon(pts, fill=color)


def _warm_glow(size, center, radius):
    """RGBA warm glow that fades to fully transparent at the edges (no grey halo)."""
    glow = _radial(size, EMBER, BLACK, center=center, radius=radius).convert("RGBA")
    mask = _radial(size, (255, 255, 255), (0, 0, 0), center=center, radius=int(radius * 1.1)).convert("L")
    glow.putalpha(mask.point(lambda v: int(v * 0.6)))
    return glow


def _anvil(draw, ox, oy, scale, color):
    """Stylized anvil silhouette as filled polygons."""
    def P(pts):
        return [(ox + px * scale, oy + py * scale) for px, py in pts]
    # top face + horn
    draw.polygon(P([(0, 0), (34, 0), (44, 6), (30, 10), (6, 10), (0, 6)]), fill=color)
    # waist
    draw.polygon(P([(10, 10), (24, 10), (22, 20), (12, 20)]), fill=color)
    # base
    draw.polygon(P([(4, 20), (30, 20), (34, 30), (0, 30)]), fill=color)


def make_emblem_png(path: Path) -> Path:
    size = (256, 256)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    # warm flame glow behind, fading to transparent
    img = Image.alpha_composite(img, _warm_glow(size, (128, 96), 120))
    draw = ImageDraw.Draw(img, "RGBA")
    # flame: nested teardrops gold->ember, tip up
    _flame(draw, 128, 118, 96, 44, GOLD_DK + (235,))
    _flame(draw, 128, 116, 76, 33, EMBER + (240,))
    _flame(draw, 128, 114, 54, 22, GOLD + (245,))
    _flame(draw, 128, 112, 32, 12, GOLD_HI + (255,))
    # anvil in gold, centered low
    _anvil(draw, 106, 150, 1.4, GOLD)
    # silver ground line
    draw.line([(56, 196), (200, 196)], fill=SILVER + (180,), width=3)
    img.save(path)
    return path


def make_emblem_svg(path: Path) -> Path:
    g = "#%02x%02x%02x" % GOLD
    e = "#%02x%02x%02x" % EMBER
    s = "#%02x%02x%02x" % SILVER
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <defs><radialGradient id="f" cx="50%" cy="40%" r="60%">
    <stop offset="0%" stop-color="{e}"/><stop offset="100%" stop-color="#0a0a0b" stop-opacity="0"/>
  </radialGradient></defs>
  <circle cx="128" cy="96" r="110" fill="url(#f)"/>
  <path d="M128 36 C150 70 150 96 128 110 C106 96 106 70 128 36 Z" fill="{g}"/>
  <path d="M106 150 h48 l14 8 -14 6 h-34 z M116 164 h26 l-2 14 h-22 z M110 178 h36 l6 14 h-48 z" fill="{g}"/>
  <line x1="56" y1="196" x2="200" y2="196" stroke="{s}" stroke-width="3"/>
</svg>'''
    path.write_text(svg, encoding="utf-8")
    return path


def make_icon(png_path: Path, ico_path: Path):
    size = (256, 256)
    tile = _radial(size, PANEL, BLACK, radius=200).convert("RGBA")
    tile = Image.alpha_composite(tile, _warm_glow(size, (128, 104), 110))
    draw = ImageDraw.Draw(tile, "RGBA")
    _flame(draw, 128, 120, 82, 36, EMBER + (240,))
    _flame(draw, 128, 116, 58, 23, GOLD + (245,))
    _flame(draw, 128, 112, 34, 12, GOLD_HI + (255,))
    _anvil(draw, 100, 150, 1.6, GOLD)
    tile.save(png_path)
    tile.save(ico_path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return png_path, ico_path


def make_hero(path: Path) -> Path:
    size = (1200, 360)
    img = _radial(size, _lerp(EMBER, GOLD_DK, 0.4), BLACK, center=(300, 300), radius=520).convert("RGBA")
    core = _radial(size, GOLD, BLACK, center=(300, 320), radius=240).convert("RGBA")
    core.putalpha(150)
    img = Image.alpha_composite(img, core)
    _add_sparks(img, 90, EMBER)
    _add_sparks(img, 40, GOLD_HI, seedy=999)
    rgb = img.convert("RGB")
    # real edge vignette: blend toward near-black using a radial mask
    vig = _radial(size, (255, 255, 255), (30, 28, 24), radius=780).convert("L")
    rgb = Image.composite(rgb, Image.new("RGB", size, BLACK), vig)
    rgb = rgb.filter(ImageFilter.GaussianBlur(0.6))
    rgb.save(path)
    return path


def make_embers(path: Path) -> Path:
    size = (512, 512)
    img = _radial(size, _lerp(BLACK, GOLD_DK, 0.15), BLACK, center=(120, 460), radius=560).convert("RGBA")
    _add_sparks(img, 70, EMBER)
    img = img.filter(ImageFilter.GaussianBlur(1.0))
    img.convert("RGB").save(path)
    return path


def make_panel_metal(path: Path) -> Path:
    size = (256, 256)
    base = Image.new("RGB", size, PANEL)
    draw = ImageDraw.Draw(base)
    for y in range(0, 256, 2):
        shade = 18 + ((y * 37) % 10)
        draw.line([(0, y), (256, y)], fill=(shade, shade - 1, shade - 2))
    base = base.filter(ImageFilter.GaussianBlur(0.4))
    base.save(path)
    return path


def make_nav_icon(path: Path, kind: str) -> Path:
    size = (48, 48)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    c = GOLD + (255,)
    if kind == "home":
        _anvil(d, 8, 12, 0.85, GOLD)
    elif kind == "setup":
        d.ellipse([14, 14, 34, 34], outline=c, width=3)
        for a in range(0, 360, 45):
            x = 24 + 16 * math.cos(math.radians(a))
            y = 24 + 16 * math.sin(math.radians(a))
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c)
        d.ellipse([20, 20, 28, 28], fill=(10, 10, 11, 255))
    elif kind == "dataset":
        d.rectangle([10, 12, 38, 34], outline=c, width=3)
        d.ellipse([16, 18, 22, 24], fill=c)
        d.polygon([(14, 32), (24, 22), (30, 28), (34, 24), (36, 32)], fill=c)
    elif kind == "train":
        d.polygon([(10, 30), (20, 12), (22, 24), (32, 10), (30, 26), (40, 22), (24, 40)], fill=GOLD_HI + (255,))
    elif kind == "batch":
        for i in range(3):
            y = 12 + i * 9
            d.rectangle([10, y, 38, y + 6], outline=c, width=2)
    elif kind == "presets":
        # three vertical slider rails with offset knobs (preset mixer glyph)
        for x, ky in ((13, 18), (24, 32), (35, 24)):
            d.line([(x, 10), (x, 38)], fill=c, width=3)
            d.ellipse([x - 5, ky - 5, x + 5, ky + 5], fill=c)
            d.ellipse([x - 2, ky - 2, x + 2, ky + 2], fill=(10, 10, 11, 255))
    elif kind == "characters":
        # two head + shoulder silhouettes
        d.ellipse([11, 10, 23, 22], outline=c, width=3)
        d.arc([7, 20, 27, 42], 200, 340, fill=c, width=3)
        d.ellipse([27, 14, 37, 24], outline=c, width=2)
        d.arc([24, 24, 42, 44], 200, 340, fill=c, width=2)
    img.save(path)
    return path


def generate_all(assets_dir: Path) -> list[Path]:
    assets_dir = Path(assets_dir)
    (assets_dir / "nav").mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    out.append(make_emblem_png(assets_dir / "emblem.png"))
    out.append(make_emblem_svg(assets_dir / "emblem.svg"))
    png, ico = make_icon(assets_dir / "icon.png", assets_dir / "icon.ico")
    out.extend([png, ico])
    out.append(make_hero(assets_dir / "hero_forge.png"))
    out.append(make_embers(assets_dir / "bg_embers.png"))
    out.append(make_panel_metal(assets_dir / "panel_metal.png"))
    for kind in ("home", "setup", "dataset", "characters", "train", "batch", "presets"):
        out.append(make_nav_icon(assets_dir / "nav" / f"{kind}.png", kind))
    return out


if __name__ == "__main__":
    paths = generate_all(ASSETS)
    print(f"Wrote {len(paths)} assets to {ASSETS}")
