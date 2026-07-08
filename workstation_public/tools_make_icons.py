"""Generate PWA icons (192/512 PNG) + favicon for the dual-arm workstation UI.
Brand: primary #1a3a6e bg, dual robot arms in accent #EF9F27 / #FFD66B reaching
a shared workpiece. Run once: `python tools_make_icons.py`. Pure Pillow, no net."""
import os
from PIL import Image, ImageDraw

OUT_IMG = os.path.join(os.path.dirname(__file__), "static", "images")
OUT_STATIC = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(OUT_IMG, exist_ok=True)

PRIMARY = (26, 58, 110)      # #1a3a6e
PRIMARY_D = (18, 42, 82)     # darker for subtle vignette
ACCENT = (239, 159, 39)      # #EF9F27
ACCENT2 = (255, 214, 107)    # #FFD66B
CREAM = (255, 248, 232)


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def disc(d, cx, cy, r, fill):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def seg(d, p0, p1, w, fill):
    """Thick rounded segment between two points."""
    d.line([p0, p1], fill=fill, width=int(w))
    disc(d, p0[0], p0[1], w / 2, fill)
    disc(d, p1[0], p1[1], w / 2, fill)


def render(size, content_scale=1.0):
    """content_scale<1 shrinks the motif toward center for maskable safe-zone."""
    ss = 4
    S = size * ss
    img = Image.new("RGB", (S, S), PRIMARY)
    d = ImageDraw.Draw(img)

    # subtle radial vignette (top-light) via concentric discs
    for i in range(60, 0, -1):
        t = i / 60.0
        r = S * (0.95 * t)
        col = lerp(PRIMARY, PRIMARY_D, (1 - t) * 0.55)
        d.ellipse([S * 0.5 - r, S * 0.30 - r, S * 0.5 + r, S * 0.30 + r], fill=col)

    def P(x, y):
        cx = 0.5 + (x - 0.5) * content_scale
        cy = 0.45 + (y - 0.45) * content_scale  # shrink toward visual center
        return (S * cx, S * cy)

    wseg = S * 0.052 * content_scale
    # workpiece (shared target) at top-center
    wp = P(0.50, 0.295)

    # Left arm: base -> elbow -> gripper (ACCENT)
    aL = [P(0.215, 0.815), P(0.315, 0.545), P(0.452, 0.345)]
    # Right arm: base -> elbow -> gripper (ACCENT2)
    aR = [P(0.785, 0.815), P(0.685, 0.545), P(0.548, 0.345)]

    # bases (rounded plates)
    for (bx, by), col in ((aL[0], ACCENT), (aR[0], ACCENT2)):
        bw = S * 0.135 * content_scale
        bh = S * 0.045 * content_scale
        d.rounded_rectangle([bx - bw / 2, by, bx + bw / 2, by + bh],
                            radius=bh / 2, fill=col)

    # arm segments
    for chain, col in ((aL, ACCENT), (aR, ACCENT2)):
        seg(d, chain[0], chain[1], wseg, col)
        seg(d, chain[1], chain[2], wseg * 0.92, col)
        # elbow joint ring
        disc(d, chain[1][0], chain[1][1], wseg * 0.78, col)
        disc(d, chain[1][0], chain[1][1], wseg * 0.40, PRIMARY)
        # gripper claw toward workpiece
        gx, gy = chain[2]
        disc(d, gx, gy, wseg * 0.62, col)

    # connecting "energy" lines from grippers to workpiece
    for chain, col in ((aL, ACCENT2), (aR, ACCENT)):
        d.line([chain[2], wp], fill=lerp(col, CREAM, 0.3),
               width=max(1, int(S * 0.012 * content_scale)))

    # workpiece glow
    disc(d, wp[0], wp[1], S * 0.075 * content_scale, lerp(ACCENT2, CREAM, 0.2))
    disc(d, wp[0], wp[1], S * 0.050 * content_scale, CREAM)

    img = img.resize((size, size), Image.LANCZOS)
    return img


for sz in (192, 512):
    render(sz).save(os.path.join(OUT_IMG, f"icon-{sz}.png"), optimize=True)
    print(f"wrote icon-{sz}.png")

# maskable variants: motif shrunk to ~72% so Android circle/squircle masks
# never clip the arms or workpiece (safe-zone = center 80%).
for sz in (192, 512):
    render(sz, content_scale=0.72).save(
        os.path.join(OUT_IMG, f"icon-maskable-{sz}.png"), optimize=True)
    print(f"wrote icon-maskable-{sz}.png")

# 32px favicon.png as raster fallback
render(64).resize((32, 32), Image.LANCZOS).save(
    os.path.join(OUT_IMG, "favicon-32.png"), optimize=True)
print("wrote favicon-32.png")

# also a classic favicon.ico (multi-size)
ico = render(64)
ico.save(os.path.join(OUT_STATIC, "favicon.ico"),
         sizes=[(16, 16), (32, 32), (48, 48)])
print("wrote favicon.ico")
