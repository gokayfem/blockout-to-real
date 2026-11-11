"""Time-lapse of a finished case: the photo on the left, the scene on the right, built up step by step.

    python -m b2r.timelapse --case CASE [--fps 24] [--width 1920] [--skip-render] [--render-only]
                            [--samples 64] [--scale F] [--orbit-deg 6] [--orbit-frames 37]
                            [--relight-frames 13] [--only reveal,lights,orbit,relight]

Step 1 (Blender, blender/timelapse_frames.py; skipped with --skip-render) renders CASE/timelapse/frames/:
  reveal/f_0000.png ...   state k: the first k objects carry their materials, the rest are grey; fitted lights
  lights/f_0001.png ...   each fitted light alone (the basis the light fit works with), strongest first
  orbit/f_0001.png ...    the camera orbiting the view pivot; the middle frame is the case camera
  relight/f_0001.png ...  the fitted rig fading to a warm evening
  frames.json             counts, the object order, the light order with shares, the orbit amplitude
Step 2 composes CASE/timelapse/timelapse.mp4 and thumb.jpg (frame 0, a finished title card):
  blockout -> photo imagined -> maps -> geometry edits per pass -> materials object by object ->
  lights one at a time -> re-render against the photo (LPIPS) -> camera moves -> relit.
A step whose inputs are missing is left out and named on stdout; nothing stands in for it.

Every picture on the right is a Cycles render of the case's own .blend files; the photo and its maps stay
on the left. Nothing from the photo is composited into a render. Needs ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import functools
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import common

W0, H0 = 1920, 1080
WHITE, CYAN, GREY, SOFT = (255, 255, 255), (32, 211, 238), (150, 163, 186), (214, 223, 237)
PANEL, INK = (16, 23, 38), (6, 14, 24)
S = 1.0   # output width / 1920, set in compose()

FONT_FILES = [  # (regular, bold or None when the file is a variable font)
    ("/System/Library/Fonts/SFNS.ttf", None),
    ("/System/Library/Fonts/Helvetica.ttc", None),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
]


def px(v):
    return int(round(v * S))


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ss(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


@functools.lru_cache(maxsize=None)
def font(size, bold):
    size = max(6, int(size))
    for reg, bld in FONT_FILES:
        if not os.path.exists(reg):
            continue
        path = bld if (bold and bld and os.path.exists(bld)) else reg
        f = ImageFont.truetype(path, size)
        try:  # a variable font: set weight and optical size
            vals = []
            for ax in f.get_variation_axes():
                n = ax["name"].decode() if isinstance(ax["name"], bytes) else ax["name"]
                if n == "Weight":
                    vals.append(min(max(650 if bold else 400, ax["minimum"]), ax["maximum"]))
                elif n == "Optical Size":
                    vals.append(min(max(size, ax["minimum"]), ax["maximum"]))
                else:
                    vals.append(ax["default"])
            f.set_variation_by_axes(vals)
        except Exception:
            pass
        return f
    return ImageFont.load_default(size)


@functools.lru_cache(maxsize=2048)
def tmask(text, size, bold):
    f = font(size, bold)
    l, t, r, b = f.getbbox(text or " ")
    pad = 4
    im = Image.new("L", (max(1, int(math.ceil(r - min(l, 0))) + 2 * pad), max(1, int(b) + 2 * pad)), 0)
    ImageDraw.Draw(im).text((pad - min(l, 0), pad), text, font=f, fill=255)
    return im


def text_w(text, size, bold=False):
    return font(max(6, int(size)), bold).getlength(text)


def fade(mask, a):
    return mask if a >= 0.999 else mask.point(lambda v: int(v * a + 0.5))


def draw_text(c, text, x, y, size, bold=False, color=WHITE, a=1.0, anchor="l"):
    """x, y and size in 1920-wide units; y is the top of the line."""
    if not text or a <= 0.003:
        return
    sz = max(6, px(size))
    m = tmask(text, sz, bold)
    w = text_w(text, sz, bold)
    x0 = px(x) - 4 - (w if anchor == "r" else w / 2 if anchor == "c" else 0)
    c.paste(color, (int(round(x0)), px(y) - 4), fade(m, a))


@functools.lru_cache(maxsize=256)
def rmask(w, h, r):
    k = 4
    im = Image.new("L", (w * k, h * k), 0)
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * k - 1, h * k - 1), r * k, fill=255)
    return im.resize((w, h), Image.LANCZOS)


def rect(c, x, y, w, h, r, color, a=1.0):
    w, h = max(1, px(w)), max(1, px(h))
    c.paste(color, (px(x), px(y)), fade(rmask(w, h, min(px(r), w // 2, h // 2)), a))


@functools.lru_cache(maxsize=8)
def shadow(w, h, r):
    pad = max(8, px(60))
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    ImageDraw.Draw(im).rounded_rectangle((pad, pad, pad + w - 1, pad + h - 1), r, fill=150)
    return im.filter(ImageFilter.GaussianBlur(max(2, px(26)))), pad


def paste_panel(c, img, x, y):
    """img is already at panel size; x, y in 1920-wide units."""
    w, h = img.size
    r = max(2, px(14))
    sm, pad = shadow(w, h, r)
    c.paste((1, 2, 5), (px(x) - pad, px(y) - pad + px(16)), sm)
    c.paste(img, (px(x), px(y)), rmask(w, h, r))


def tag(c, text, x, y):
    if not text:
        return
    w = text_w(text, max(6, px(22)), True) / S + 28
    rect(c, x, y, w, 40, 10, (8, 12, 20), 0.74)
    draw_text(c, text, x + 14, y + 7, 22, True, WHITE)


def wipe(a, b, p):
    if p <= 0:
        return a
    if p >= 1:
        return b
    w, h = a.size
    x = int(round(p * w))
    out = a.copy()
    out.paste(b.crop((0, 0, x, h)), (0, 0))
    ImageDraw.Draw(out).rectangle((max(0, x - 1), 0, x, h), fill=(235, 250, 255))
    return out


def make_bg(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = yy / max(h - 1, 1)
    top, bot = np.array([6, 9, 15], np.float32), np.array([15, 20, 31], np.float32)
    img = top[None, None] * (1 - v[..., None]) + bot[None, None] * v[..., None]
    d = np.sqrt(((xx - w / 2) / (w * 0.62)) ** 2 + ((yy - h * 0.98) / (h * 0.9)) ** 2)
    img += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array([4, 5, 9], np.float32)
    img += np.random.default_rng(7).uniform(-0.6, 0.6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


@functools.lru_cache(maxsize=96)
def load(path, w, h):
    """An image file covering (w, h), centre-cropped."""
    im = Image.open(path)
    if im.mode in ("I;16", "I", "F"):
        a = np.asarray(im).astype(np.float32)
        im = Image.fromarray((a / max(float(a.max()), 1e-6) * 255).astype(np.uint8))
    im = im.convert("RGB")
    k = max(w / im.width, h / im.height)
    nw, nh = max(w, round(im.width * k)), max(h, round(im.height * k))
    im = im.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return im.crop((x0, y0, x0 + w, y0 + h))


def depth_png(exr: Path, out: Path):
    """The target's metric depth (EXR) as a near-is-bright PNG, or None when it cannot be read."""
    if out.exists():
        return out
    try:
        import cv2
        d = cv2.imread(str(exr), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        if d.ndim == 3:
            d = d[..., 0]
        ok = np.isfinite(d) & (d > 0)
        inv = np.zeros_like(d, np.float32)
        inv[ok] = 1.0 / d[ok]
        lo, hi = np.percentile(inv[ok], [1, 99])
        v = np.clip((inv - lo) / max(hi - lo, 1e-9), 0, 1) ** 0.85
        v[~ok] = 0
        a, b = np.array([10, 15, 28], np.float32), np.array([236, 244, 255], np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((a * (1 - v[..., None]) + b * v[..., None]).astype(np.uint8)).save(out)
        return out
    except Exception:
        return None


def seq(folder: Path, first: int, count: int):
    paths = [folder / f"f_{i:04d}.png" for i in range(first, first + count)]
    return paths if count > 0 and all(p.exists() for p in paths) else []


def pingpong(k, n):
    if n <= 1:
        return 0
    k %= 2 * n - 2
    return k if k < n else 2 * n - 2 - k


# ------------------------------------------------------------------ what a case has
def gather(root: Path) -> dict:
    """Everything the film needs, with None / [] for what the case does not have."""
    fr = root / "timelapse" / "frames"
    meta = common.read_json(fr / "frames.json") if (fr / "frames.json").exists() else {}
    summ = common.read_json(root / "final" / "summary.json") if (root / "final" / "summary.json").exists() else {}
    iters = [k for k in range(common.MAX_ITERATIONS) if (root / "target" / f"iter{k}" / "target.png").exists()]
    if not iters:
        raise SystemExit(f"{root}: no target/iter<k>/target.png; run the pipeline first")
    kf = int(meta.get("iteration", summ.get("final_iteration", iters[-1])))
    g = {"root": root, "kf": kf, "name": root.name, "meta": meta}
    cfg = common.read_json(root / "case.json") if (root / "case.json").exists() else {}
    g["aspect"] = cfg.get("size", {}).get("width", 16) / max(cfg.get("size", {}).get("height", 9), 1)
    g["beauty"] = [p for p in (root / "gbuffers" / f"iter{k}" / "beauty.png" for k in range(kf + 1)) if p.exists()]
    g["target"] = root / "target" / f"iter{kf}" / "target.png"
    dec = root / "decompose" / f"iter{kf}"
    g["maps"] = [(n, p) for n, p in (("albedo", dec / "albedo.png"), ("roughness", dec / "roughness.png"),
                                     ("normals", dec / "normal_cam.png")) if p.exists()]
    dp = depth_png(dec / "depth_moge.exr", root / "timelapse" / "depth_vis.png") if (dec / "depth_moge.exr").exists() else None
    if dp:
        g["maps"].append(("metric depth", dp))
    g["reveal"] = seq(fr / "reveal", 0, int(meta.get("reveal", {}).get("count", 0)))
    g["order"] = [str(x) for x in meta.get("reveal", {}).get("order", [])]
    g["lights"] = seq(fr / "lights", 1, int(meta.get("lights", {}).get("count", 0)))
    g["light_order"] = meta.get("lights", {}).get("order", [])
    g["orbit"] = seq(fr / "orbit", 1, int(meta.get("orbit", {}).get("count", 0)))
    g["orbit_deg"] = meta.get("orbit", {}).get("amplitude_deg")
    g["relight"] = seq(fr / "relight", 1, int(meta.get("relight", {}).get("count", 0)))
    g["relight_look"] = meta.get("relight", {}).get("look", "relit")
    final = root / "final" / "render_final.png"
    rr = root / "residual" / f"iter{kf}" / "rerender.png"
    g["final"] = g["reveal"][-1] if g["reveal"] else final if final.exists() else rr if rr.exists() else None
    it = summ.get("iterations", {}).get(str(kf), {})
    g["lpips"] = (it.get("blockout_vs_target", {}).get("lpips_alex"), it.get("headline_lpips_alex"))
    ch = root / "target" / f"iter{kf}" / "choice.json"
    g["level"] = common.read_json(ch).get("chosen_level") if ch.exists() else None
    ob = root / "blockout" / f"iter{kf}" / "objects.json"
    g["n_objects"] = len(common.read_json(ob)) if ob.exists() else None
    return g


def lpips_text(g):
    a, b = g["lpips"]
    return f"LPIPS to the photo {a:.3f} → {b:.3f}" if a is not None and b is not None else ""


# ------------------------------------------------------------------ segments
def segments(g, fps, pw, ph):
    """A list of (chip, seconds, fn(t) -> dict(title, meta, left, right, rtag))."""
    L = lambda p: load(str(p), pw, ph)   # noqa: E731
    empty = Image.new("RGB", (pw, ph), (13, 18, 30))
    target = L(g["target"])
    out, skipped = [], []
    beauty = [L(p) for p in g["beauty"]]
    b0 = beauty[0] if beauty else empty
    nobj = f"{g['n_objects']} objects · " if g["n_objects"] else ""
    if beauty:
        out.append(("Blockout", 1.8, lambda t: dict(title="The grey blockout", meta=f"{nobj}one fixed camera",
                                                    left=None, right=b0, rtag="blockout")))
    else:
        skipped.append("blockout (no gbuffers/iter<k>/beauty.png)")
    lvl = f"conditioning level {g['level']}" if g["level"] else "the same view"
    out.append(("Photo", 2.4, lambda t: dict(title="Imagine the photo", meta=lvl, left=wipe(empty, target, ss(t / 0.9)),
                                             right=b0, rtag="blockout")))
    if g["maps"]:
        maps = [(n, L(p)) for n, p in g["maps"]]
        step = 1.0

        def f_maps(t):
            i = int(clamp(math.floor(t / step), 0, len(maps)))
            prev = target if i == 0 else maps[i - 1][1]
            cur, name = (maps[i][1], maps[i][0]) if i < len(maps) else (target, "back to the photo")
            return dict(title="Take the photo apart", meta=name, left=wipe(prev, cur, ss((t - i * step) / 0.5)),
                        right=b0, rtag="blockout", ltag=name if i < len(maps) else "")
        out.append(("Maps", step * len(maps) + 0.7, f_maps))
    else:
        skipped.append("maps (no decompose maps)")
    if len(beauty) > 1:
        def f_edit(t):
            i = int(clamp(math.floor(t / 1.5), 0, len(beauty) - 2))
            return dict(title="Geometry edits from the reviewer", meta=f"pass {i + 2}", left=target,
                        right=wipe(beauty[i], beauty[i + 1], ss((t - i * 1.5) / 0.8)), rtag=f"blockout, pass {i + 2}")
        out.append(("Edits", 1.5 * (len(beauty) - 1), f_edit))
    final = L(g["final"]) if g["final"] else None
    if g["reveal"]:
        n = len(g["reveal"]) - 1
        hold = max(2, round(fps * min(0.25, 7.0 / (n + 1)))) / fps

        def f_rev(t):
            st = int(clamp(math.floor(t / hold), 0, n))
            nm = g["order"][st - 1].replace("_", " ") if 0 < st <= len(g["order"]) else ""
            return dict(title="Write it back, object by object", meta=f"{st} / {n} objects" + (f" · {nm}" if nm else ""),
                        left=target, right=L(g["reveal"][st]), rtag=f"materials {st}/{n}")
        out.append(("Materials", (n + 1) * hold + 0.6, f_rev))
    else:
        skipped.append("materials (no timelapse/frames/reveal)")
    if g["lights"]:
        nl = len(g["lights"])

        def f_lights(t):
            j = int(math.floor(t / 0.9))
            if j >= nl:
                return dict(title="The fitted lights, one at a time", meta="all of them", left=target, right=final,
                            rtag="all fitted lights")
            o = g["light_order"][j] if j < len(g["light_order"]) else {}
            nm = str(o.get("name", f"light {j + 1}")).replace("_", " ")
            share = f" · {round(100 * o['energy_share'])}% of the energy" if o.get("energy_share") is not None else ""
            return dict(title="The fitted lights, one at a time", meta=nm + share + (" · virtual" if o.get("virtual") else ""),
                        left=target, right=L(g["lights"][j]), rtag=f"{nm} alone")
        out.append(("Lights", 0.9 * nl + 1.1, f_lights))
    else:
        skipped.append("lights (no timelapse/frames/lights)")
    if final is not None:
        out.append(("Compare", 2.4, lambda t: dict(title="The re-render against the photo", meta=lpips_text(g),
                                                   left=target, right=final, rtag="Cycles re-render")))
    if g["orbit"]:
        m = len(g["orbit"])
        deg = f"orbit, {g['orbit_deg']:g}° each way · " if g["orbit_deg"] else ""

        def f_orbit(t):
            k = m // 2 + int(round(t * fps))
            return dict(title="Still a scene: move the camera", meta=f"{deg}no photo exists for these views",
                        left=target, right=L(g["orbit"][pingpong(k, m)]), rtag="camera moved")
        out.append(("Camera", (2 * m - 2) / fps, f_orbit))
    else:
        skipped.append("camera (no timelapse/frames/orbit)")
    if g["relight"]:
        r = len(g["relight"])

        def f_relight(t):
            pos = (r - 1) * ss(clamp((0.5 - abs(t / 3.4 - 0.5)) / 0.4))
            i = int(math.floor(pos))
            im = L(g["relight"][i])
            if pos - i > 0.01 and i + 1 < r:
                im = Image.blend(im, L(g["relight"][i + 1]), pos - i)
            return dict(title="Still a scene: change the light", meta=f"the fitted rig fading to a {g['relight_look']}",
                        left=target, right=im, rtag="relit" if pos > 0.5 else "as fitted")
        out.append(("Relight", 3.4, f_relight))
    else:
        skipped.append("relight (no timelapse/frames/relight)")
    if final is not None:
        out.append(("", 1.6, lambda t: dict(title=g["name"], meta=lpips_text(g), left=target, right=final,
                                            rtag="final scene")))
    return out, skipped


def title_card(g, bg):
    """Frame 0: blockout | final scene, darkened, under the title. A finished thumbnail."""
    w, h = bg.size
    c = bg.copy()
    a = load(str(g["beauty"][0]), w, h) if g["beauty"] else None
    b = load(str(g["final"]), w, h) if g["final"] else load(str(g["target"]), w, h)
    full = b.copy()
    if a is not None:
        full.paste(a.crop((0, 0, w // 2, h)), (0, 0))
    c.paste(full.point([int(v * 0.46) for v in range(256)] * 3), (0, 0))
    ImageDraw.Draw(c).rectangle((w // 2 - 1, 0, w // 2, h), fill=(120, 128, 140))
    parts = [("Blockout ", WHITE), ("2", CYAN), (" Real", WHITE)]
    sz = 150
    total = sum(text_w(p, px(sz), True) for p, _ in parts) / S
    x = W0 / 2 - total / 2
    for p, col in parts:
        draw_text(c, p, x, 362, sz, True, col)
        x += text_w(p, px(sz), True) / S
    draw_text(c, f"{g['name']} · time-lapse", W0 / 2, 566, 40, False, (200, 210, 230), 1, "c")
    if lpips_text(g):
        draw_text(c, lpips_text(g), W0 / 2, 630, 28, False, GREY, 1, "c")
    draw_text(c, "Blockout", 60, H0 - 76, 28, True)
    draw_text(c, "Re-render", W0 - 60, H0 - 76, 28, True, WHITE, 1, "r")
    return c


def compose(case: str | Path, fps: int = 24, width: int = 1920, out: Path | None = None, title_seconds: float = 1.6) -> Path:
    global S
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH: the time-lapse needs it to write the mp4")
    root = common.case_dir(case)
    width -= width % 2
    S = width / W0
    w, h = width, int(round(H0 * S / 2)) * 2
    for cache in (font, tmask, rmask, shadow, load):
        cache.cache_clear()
    g = gather(root)
    bg = make_bg(w, h)
    # two panels side by side, as large as the frame allows
    gap, mx, top, room = 24, 100, 196, 700
    pw0 = (W0 - 2 * mx - gap) / 2
    ph0 = min(pw0 / g["aspect"], room)
    pw0 = ph0 * g["aspect"]
    pw, ph = px(pw0), px(ph0)
    xl = (W0 - 2 * pw0 - gap) / 2
    xr = xl + pw0 + gap
    y0 = top + (room - ph0) / 2
    segs, skipped = segments(g, fps, pw, ph)
    for s_ in skipped:
        print("[timelapse] left out:", s_)
    chips = [c_ for c_, _, _ in segs if c_]
    total = title_seconds + sum(d for _, d, _ in segs)
    n = int(round(total * fps))
    dst = Path(out) if out else root / "timelapse" / "timelapse.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    card = title_card(g, bg)
    card.convert("RGB").save(dst.parent / "thumb.jpg", quality=92)
    empty = Image.new("RGB", (pw, ph), (13, 18, 30))

    def frame(k):
        t = k / fps
        if t < title_seconds:
            return card
        t -= title_seconds
        for i, (chip, dur, fn) in enumerate(segs):
            if t < dur or i == len(segs) - 1:
                st = fn(min(t, dur))
                break
            t -= dur
        c = bg.copy()
        draw_text(c, st["title"], mx, 58, 46, True)
        draw_text(c, st.get("meta", ""), W0 - mx, 79, 25, False, GREY, 1, "r")
        draw_text(c, "Photo", xl + 4, y0 - 48, 28, True)
        draw_text(c, "Scene", xr + 4, y0 - 48, 28, True)
        left = st.get("left")
        paste_panel(c, left if left is not None else empty, xl, y0)
        if left is None:
            draw_text(c, "not imagined yet", xl + pw0 / 2, y0 + ph0 / 2 - 14, 26, False, GREY, 1, "c")
        tag(c, st.get("ltag", ""), xl + 16, y0 + 16)
        paste_panel(c, st["right"] if st.get("right") is not None else empty, xr, y0)
        tag(c, st.get("rtag", ""), xr + 16, y0 + 16)
        # step chips and a progress line
        ws = [text_w(c_, px(24), True) / S + 40 for c_ in chips]
        x = W0 / 2 - (sum(ws) + 12 * (len(chips) - 1)) / 2
        for c_, wc in zip(chips, ws):
            on = c_ == chip
            rect(c, x, 946, wc, 46, 23, CYAN if on else PANEL, 1.0 if on else 0.9)
            draw_text(c, c_, x + 20, 955, 24, True, INK if on else SOFT)
            x += wc + 12
        rect(c, mx, 1030, W0 - 2 * mx, 4, 2, (36, 46, 66))
        rect(c, mx, 1030, max(4, (W0 - 2 * mx) * k / max(n - 1, 1)), 4, 2, CYAN)
        return c

    cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264",
           "-preset", "medium", "-crf", "17", "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-movflags", "+faststart", str(dst)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(n):
        p.stdin.write(frame(k).convert("RGB").tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed writing the time-lapse")
    common.write_json(dst.parent / "timelapse.json", {
        "video": str(dst), "frames": n, "fps": fps, "size": [w, h], "seconds": round(n / fps, 2),
        "steps": chips, "left_out": skipped, "final_iteration": g["kf"],
        "lpips_blockout_vs_target": g["lpips"][0], "lpips_rerender_vs_target": g["lpips"][1]})
    print(f"[timelapse] {dst} ({n} frames, {n / fps:.1f} s; steps: {', '.join(chips)})")
    return dst


def render_frames(case: str | Path, passthrough: list[str]) -> None:
    blender = common.find_blender()
    if not blender:
        raise SystemExit("Blender not found: set $BLENDER (or pass --skip-render to compose from existing frames)")
    root = common.case_dir(case)
    log = root / "logs" / "timelapse_frames.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [blender, "-b", "--factory-startup", "--disable-autoexec", "--python", str(common.BLENDER_DIR / "timelapse_frames.py"), "--",
           "--case", str(root), *passthrough]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(common.SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
        raise SystemExit(f"timelapse frames failed (exit {rc}); log: {log}\n{tail}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="b2r.timelapse", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-render", action="store_true", help="compose from the frames already in CASE/timelapse/frames")
    ap.add_argument("--render-only", action="store_true", help="render the frames, do not compose")
    args, passthrough = ap.parse_known_args(argv)
    if not args.skip_render:
        render_frames(args.case, passthrough)
    elif passthrough:
        raise SystemExit(f"unknown options with --skip-render: {' '.join(passthrough)}")
    if not args.render_only:
        compose(args.case, args.fps, args.width, Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
