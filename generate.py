"""
ASCII Eyes — procedurally rendered, looping GIF for a GitHub profile README.

No source video: every frame is computed from math (eyelid curves, iris
striations, gaze saccades, blinks, pupil dilation), mapped to a character
density ramp, tinted per cell, and composited from a pre-rendered glyph atlas
with numpy.

    python generate.py                       # writes eyes.gif
    python generate.py --caption "> hello, world_" --out banner.gif
    python generate.py --theme light --out eyes-light.gif
"""
import argparse
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- config
COLS, ROWS = 150, 26          # character grid
CW, CH = 8, 16                # pixel size of one character cell
W, H = COLS * CW, ROWS * CH   # output size (1200 x 416)
FONT_SIZE = 14
GLOW = 0.0                    # >0 adds a soft backlight behind cells
FPS = 12
SECONDS = 8
N = FPS * SECONDS
BG = np.array([13, 17, 23], np.float32)      # set by set_theme()
FONT_PATHS = ["C:/Windows/Fonts/consolab.ttf", "C:/Windows/Fonts/consola.ttf",
              "DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf"]

# density ramp: each level is a pool of glyphs of similar "ink"
LEVELS = [" ", " .`", ".:-',", "=+;~:", "*?ivxc", "oaezsun", "XYUJCLQ0", "#%&8B", "@WM$"]
GLYPHS = sorted(set("".join(LEVELS)) | set("/\\|_>$abcdefghijklmnopqrstuvwxyz"
                                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'@#:;-=()[]{}<>+*&%"))
GI = {g: i for i, g in enumerate(GLYPHS)}
LEVEL_IDX = [np.array([GI[g] for g in lvl]) for lvl in LEVELS]

EYES = [(W / 2 - 185.0, 190.0), (W / 2 + 185.0, 190.0)]      # eye centres (px)
EW, HU, HL = 140.0, 84.0, 62.0              # half width, upper / lower lid height
IRIS_R = 60.0
GAZE_X, GAZE_Y = 60.0, 22.0



# colour themes. dark = light emitted (dense glyph = bright); light = ink on paper
# (dense glyph = dark), so the eye interior's density is inverted in light mode.
THEMES = {
    "dark": dict(bg=(13, 17, 23), pink=(255, 105, 180), violet=(123, 47, 247),
                 brow=(150, 60, 170), crease=(200, 80, 170), lid=(230, 90, 175),
                 lash=(170, 90, 240), lower=(200, 75, 170), sclera=(255, 255, 255),
                 iris=((255, 150, 210), (255, 70, 170), (123, 47, 247)),
                 pupil=(70, 25, 110), highlight=(255, 248, 255),
                 caption=(255, 105, 180), cursor=(255, 220, 240)),
    "light": dict(bg=(255, 255, 255), pink=(220, 20, 130), violet=(100, 20, 230),
                  brow=(75, 10, 110), crease=(200, 20, 130), lid=(215, 0, 120),
                  lash=(90, 10, 200), lower=(200, 20, 130), sclera=(85, 65, 120),
                  iris=((240, 30, 150), (210, 0, 120), (95, 10, 220)),
                  pupil=(45, 15, 70), highlight=(255, 255, 255),
                  caption=(215, 40, 140), cursor=(100, 30, 205)),
}
T = None
LIGHT = False


def C(key):
    return np.array(T[key], np.float32)


def set_theme(name):
    global T, LIGHT, BG
    T, LIGHT = THEMES[name], name == "light"
    BG = C("bg")


# ---------------------------------------------------------------- helpers
def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def ease(t):
    return t * t * (3 - 2 * t)


def keyframes(t, keys):
    """Piecewise eased interpolation through (time, value...) keys."""
    for (t0, *a), (t1, *b) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            k = ease((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            return [lerp(x, y, k) for x, y in zip(a, b)]
    return list(keys[-1][1:])


# saccades: quick moves, long holds — ends where it starts so the loop is seamless
GAZE_KEYS = [
    (0.00, 0.0, 0.0), (0.14, 0.0, 0.0),
    (0.18, -0.95, 0.15), (0.34, -0.95, 0.15),
    (0.38, 0.95, -0.05), (0.54, 0.95, -0.05),
    (0.58, 0.35, -0.9), (0.70, 0.35, -0.9),
    (0.74, 0.0, 0.0), (1.00, 0.0, 0.0),
]
BLINKS = [(0.37, 0.045), (0.80, 0.04), (0.865, 0.04)]   # (centre, half-width)


def build_atlas():
    font = None
    for p in FONT_PATHS:
        try:
            font = ImageFont.truetype(p, FONT_SIZE)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    atlas = np.zeros((len(GLYPHS), CH, CW), np.float32)
    for i, g in enumerate(GLYPHS):
        im = Image.new("L", (CW, CH), 0)
        d = ImageDraw.Draw(im)
        l, t, r, b = d.textbbox((0, 0), g, font=font)
        d.text(((CW - (r - l)) / 2 - l, (CH - FONT_SIZE) / 2 - 1), g, fill=255, font=font)
        atlas[i] = np.asarray(im, np.float32) / 255.0
    return atlas


# cell centres in pixel space
YY, XX = np.mgrid[0:ROWS, 0:COLS].astype(np.float32)
PX, PY = (XX + 0.5) * CW, (YY + 0.5) * CH
RNG = np.random.default_rng(7)
CELL_SEED = RNG.integers(0, 1 << 30, (ROWS, COLS))


def frame(fi, caption):
    t = fi / N
    tau = 2 * math.pi * t
    gx, gy = keyframes(t, GAZE_KEYS)
    gx += 0.03 * math.sin(tau * 9) + 0.02 * math.sin(tau * 23)      # micro-saccades
    gy += 0.03 * math.cos(tau * 11)
    blink = max([max(0.0, 1 - abs(t - c) / w) for c, w in BLINKS] + [0.0])
    openness = 1 - ease(blink)
    pupil = 22 + 5 * math.cos(tau)                                   # dilates when facing you
    brow_lift = 16 * max(0.0, -gy) + 6 * blink

    bright = np.zeros((ROWS, COLS), np.float32)
    color = np.zeros((ROWS, COLS, 3), np.float32)
    forced = np.full((ROWS, COLS), -1, np.int64)                     # explicit glyphs (lashes)
    glow = np.zeros((ROWS, COLS), np.float32)                        # soft per-cell backlight

    # skin: warm, faint, fading away from the eyes
    fall = np.zeros_like(PX)
    for cx, cy in EYES:
        fall = np.maximum(fall, np.exp(-(((PX - cx) / 260) ** 2 + ((PY - cy) / 170) ** 2)))
    tex = 0.5 + 0.5 * np.sin(PX * 0.045 + PY * 0.02) * np.cos(PY * 0.06 - PX * 0.013)
    bright[:] = fall * (0.13 + 0.07 * tex)
    # ambient tint follows the banner gradient: hot pink -> violet, left to right
    gx01 = (PX / W)[..., None]
    color[:] = lerp(C("pink"), C("violet"), gx01)

    for side, (cx, cy) in enumerate(EYES):
        u = (PX - cx) / EW
        s = np.clip(1 - u ** 2, 0, None)
        bot = cy + HL * s ** 0.9
        top0 = cy - HU * s ** 0.75 - 6 * u * (1 if side else -1)         # slight tilt, mirrored
        top = bot + (top0 - bot) * openness
        inside = (np.abs(u) < 1) & (PY > top) & (PY < bot)

        # brow
        bu = (PX - cx) / (EW * 1.15)
        brow_y = cy - HU - 44 - brow_lift + 26 * bu ** 2 + 8 * bu * (1 if side else -1)
        brow = (np.abs(bu) < 1) & (np.abs(PY - brow_y) < 10 * (1 - 0.6 * bu ** 2))
        bright[brow] = 0.55 + 0.15 * tex[brow]
        color[brow] = C("brow")

        # lid crease + lashes
        crease = (np.abs(u) < 0.95) & (np.abs(PY - (top0 - 24)) < 5) & (openness > 0.3)
        bright[crease] = np.maximum(bright[crease], 0.32)
        color[crease] = C("crease")
        lid = (np.abs(u) < 1) & (PY <= top) & (PY > top - 12)
        bright[lid] = 0.62
        color[lid] = C("lid")
        lash = (np.abs(u) < 0.9) & (PY <= top - 8) & (PY > top - 28) & (np.sin(u * 46) > -0.2)
        bright[lash] = 0.7
        color[lash] = C("lash")
        forced[lash] = np.where(u[lash] < -0.25, GI["\\"], np.where(u[lash] > 0.25, GI["/"], GI["|"]))
        lower = (np.abs(u) < 1) & (PY >= bot) & (PY < bot + 10)
        bright[lower] = 0.45
        color[lower] = C("lower")

        # sclera with lid shadow and corner darkening
        edge = np.minimum(PY - top, bot - PY)
        scl = (0.30 + 0.50 * smooth(0, 28, edge)) * (1 - 0.45 * smooth(0.55, 1.0, np.abs(u)))
        bright[inside] = scl[inside]
        color[inside] = C("sclera")
        glow[inside] = 0.05 * scl[inside]

        # iris
        ix, iy = cx + gx * GAZE_X, cy + gy * GAZE_Y
        dx, dy = PX - ix, PY - iy
        d = np.sqrt(dx * dx + dy * dy) / IRIS_R
        ang = np.arctan2(dy, dx)
        iris = inside & (d < 1)
        streak = 0.72 + 0.28 * np.sin(ang * 26 + 2.5 * np.sin(ang * 7 + 1.3)) \
            * (0.6 + 0.4 * np.sin(ang * 61 + d * 9))
        ib = (1.0 - 0.40 * smooth(0.75, 1.0, d)) * (0.35 + 0.65 * streak)
        ib *= 0.55 + 0.45 * smooth(0, 22, PY - top)                     # upper-lid shadow
        bright[iris] = ib[iris]
        gold, orange, rim = (np.array(c, np.float32) for c in T["iris"])
        k1 = smooth(0.30, 0.60, d)[..., None]
        k2 = smooth(0.70, 1.0, d)[..., None]
        icol = lerp(lerp(gold, orange, k1), rim, k2)
        color[iris] = icol[iris]
        glow[iris] = 0.30 * ib[iris]

        # pupil
        pr = np.sqrt(dx * dx + dy * dy)
        pup = inside & (pr < pupil)
        bright[pup] = 0.06 + 0.06 * smooth(pupil * 0.7, pupil, pr[pup])
        color[pup] = C("pupil")
        glow[pup] = 0.0

        # specular highlights (fixed light source, so they ride on the iris)
        hl_any = np.zeros_like(inside)
        for hx, hy, hr, hb in ((-0.34, -0.40, 0.20, 1.0), (0.30, 0.32, 0.08, 0.85)):
            hd = np.sqrt((PX - ix - hx * IRIS_R) ** 2 + (PY - iy - hy * IRIS_R) ** 2) / (IRIS_R * hr)
            hl = inside & (hd < 1)
            bright[hl] = np.maximum(bright[hl], hb * (1 - 0.3 * hd[hl]))
            color[hl] = C("highlight")
            glow[hl] = 0.25
            hl_any |= hl

        if LIGHT:   # ink on paper: pale whites, mid-tone iris, solid pupil, blank glint
            scl_only = inside & ~iris
            bright[scl_only] = 0.22 + 0.40 * (1 - scl[scl_only])
            bright[iris] = 0.62 + 0.35 * (1 - ib[iris])
            bright[pup] = 0.92
            bright[hl_any] = 0.0

    # glyph choice: density level + slow shimmer
    lvl = np.clip((bright * (len(LEVELS) - 1) + 0.5).astype(int), 0, len(LEVELS) - 1)
    # eyes shimmer every 2 frames, the faint background only every 6 -> far fewer changed pixels
    tick = fi // np.where(bright > 0.25, 2, 6)
    h = (CELL_SEED * 2654435761 + tick * 40503 + (CELL_SEED >> 7) * tick) & 0xFFFFFFF
    idx = np.zeros((ROWS, COLS), np.int64)
    for L, pool in enumerate(LEVEL_IDX):
        m = lvl == L
        idx[m] = pool[h[m] % len(pool)]
    idx = np.where(forced >= 0, forced, idx)
    floor = 0.85 if LIGHT else 0.50                                  # ink stays saturated on white
    tint = BG + (color - BG) * np.clip(floor + 0.60 * bright, 0, 1.0)[..., None]

    # caption: typed in, held with a blinking cursor, erased — loops cleanly
    if caption:
        n = len(caption)
        if t < 0.08:
            shown = 0
        elif t < 0.40:
            shown = int(n * (t - 0.08) / 0.32)
        elif t < 0.86:
            shown = n
        else:
            shown = max(0, int(n * (1 - (t - 0.86) / 0.10)))
        text = caption[:shown]
        cursor = (fi // 4) % 2 == 0 or 0.08 <= t < 0.40 or t >= 0.86
        row = ROWS - 2
        c0 = (COLS - n - 1) // 2
        for j, ch in enumerate(text + ("_" if cursor else "")):
            if ch == " ":
                continue
            idx[row, c0 + j] = GI.get(ch, GI["?"])
            tint[row, c0 + j] = C("caption") if j < len(text) else C("cursor")

    return idx, np.clip(tint, 0, 255), glow


def compose(atlas, idx, tint, glow):
    m = atlas[idx]                                   # R, C, CH, CW
    back = BG + (tint - BG) * (GLOW * glow)[..., None]        # R, C, 3
    back = back[:, :, None, None, :]
    img = back + (tint[:, :, None, None, :] - back) * m[..., None]
    img = img.transpose(0, 2, 1, 3, 4).reshape(H, W, 3)
    # faint CRT scanlines
    img[1::2] = BG + (img[1::2] - BG) * 0.88
    return img.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eyes.gif")
    ap.add_argument("--caption", default="> thanks for stopping by. don't hesitate to reach out")
    ap.add_argument("--theme", choices=list(THEMES), default="dark")
    ap.add_argument("--frames-dir", default=None, help="also dump PNG frames here")
    a = ap.parse_args()
    set_theme(a.theme)

    atlas = build_atlas()
    frames = [compose(atlas, *frame(i, a.caption)) for i in range(N)]

    # one shared palette across all frames -> no colour flicker between frames
    sample = np.concatenate(frames[:: max(1, N // 12)], axis=0)
    K = 96                                            # palette size; index K = transparent
    pal = Image.fromarray(sample).quantize(colors=K, method=Image.Quantize.MEDIANCUT)
    idx = [np.asarray(Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE))
           for f in frames]
    # inter-frame delta: pixels unchanged since the previous frame become transparent,
    # so each frame only stores what moved (LZW squeezes the transparent runs to almost nothing)
    rgb = pal.getpalette()[: 3 * K]
    # snap the background's palette entry to the exact page colour (no faint box on GitHub)
    p = np.array(rgb, np.float32).reshape(-1, 3)
    p[idx[0][0, 0]] = BG                              # top-left pixel is always background
    rgb = [int(v) for v in p.ravel()] + [0, 0, 0]
    gif = []
    for i, cur in enumerate(idx):
        out = cur.copy()
        if i:
            out[cur == idx[i - 1]] = K
        im = Image.fromarray(out.astype(np.uint8), "P")
        im.putpalette(rgb)
        gif.append(im)
    if a.frames_dir:
        import os
        os.makedirs(a.frames_dir, exist_ok=True)
        for i in (0, 24, 45, 60, 77, 92):
            Image.fromarray(frames[i]).save(f"{a.frames_dir}/f{i:03d}.png")
    gif[0].save(a.out, save_all=True, append_images=gif[1:], duration=1000 // FPS,
                loop=0, optimize=False, disposal=1, transparency=K)
    print(f"wrote {a.out}: {W}x{H}, {N} frames")


if __name__ == "__main__":
    main()
