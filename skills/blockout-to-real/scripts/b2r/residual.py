"""Residual: how far the Cycles re-render is from the target, globally and per object.

    python -m b2r.residual --case CASE --iter K [--rerender PNG] [--target PNG] [--blockout PNG]
                           [--tag T] [--calibrate] [--device auto|mps|cuda|cpu]

Inputs (all case.json "size", pixel-registered; nothing is resampled):
    rerender  default residual/iterK/rerender.png   (Cycles re-render of the lit scene)
    target    default target/iterK/target.png       (the photoreal target)
    blockout  default gbuffers/iterK/beauty.png     (grey baseline)
    ids       gbuffers/iterK/id.png; names from blockout/iterK/objects.json and
              gbuffers/iterK/objects_visible.json

Outputs in residual/iterK/ (with --tag T every name gets "_T" before the extension, and the
untagged files are never touched):
    lpips.json          LPIPS-Alex / LPIPS-VGG / PSNR / SSIM of (rerender, target) and (blockout, target).
                        Headline = lpips_alex(rerender, target).
    error_map.png/.npy  spatial LPIPS-Alex(rerender, target), magma, FIXED scale 0..ERROR_MAP_MAX.
    per_object.json     one row per object id in the image, plus row "0" for open sky (pixels that hit no
                        object, e.g. the view through an opening): pixels, mean spatial LPIPS of re-render and
                        blockout, gap ratio, dE76, signed dL*, class small / large.
    contact_sheet.png   blockout | target | re-render | error map + the per-object table.
    detail_height.png   16-bit relative height for "small" objects (+ detail.json), read by the detail stage.
    calibration.json    only with --calibrate: reference degradations of the target.

Classification rule (CLASS_RULE): an object whose eroded mask has fewer than MIN_PIXELS pixels is "small";
otherwise "large" if coarse_gap_ratio > GAP_RATIO_MAX or coarse dE76 > DELTA_E_MAX, else "small".
Large = needs a geometry / material / exterior edit (the critique's job); small = detail map's job.
The open-sky row is classified by the same rule; it has no surface, so it never gets detail, and when it is
large the fix is geometry outside the opening (category "exterior").

detail_height.png is a HEURISTIC, not measured geometry: band-passed (about 2-12 px) luminance detail that the
target has and the re-render lacks, read as relief lit from the front. It also turns print, model noise or
misregistration into "relief"; the detail stage keeps it only on objects where it measurably helps.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import common

common.setup_model_env()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# ---------------------------------------------------------------- constants (calibrated in the experiment)
ERROR_MAP_MAX = 1.0          # spatial LPIPS value at the top of the colour map
MASK_ERODE_PX = 4            # erode object masks before per-object stats (LPIPS bleeds across edges)
MIN_PIXELS = 1500            # below this eroded pixel count an object cannot be "large"
GAP_DENOM_FLOOR = 0.05       # floor on the blockout LPIPS in the gap-ratio denominator
COARSE_SIGMA = 3.0           # px; classification compares blurred images so missing fine detail is not "large"
GAP_RATIO_MAX = 0.60
DELTA_E_MAX = 12.0
SKY_ID = 0
SKY_NAME = "(open sky: no object)"
CLASS_RULE = (
    f"large if eroded_pixels >= {MIN_PIXELS} and (coarse_gap_ratio > {GAP_RATIO_MAX} or "
    f"coarse_mean_dE76 > {DELTA_E_MAX}); else small. coarse_* = computed after Gaussian blur "
    f"sigma {COARSE_SIGMA} px of all images; coarse_gap_ratio = coarse_lpips_alex_obj(rerender,target) / "
    f"max(coarse_lpips_alex_obj(blockout,target), {GAP_DENOM_FLOOR}); object masks eroded by "
    f"{MASK_ERODE_PX} px; objects under the pixel minimum are small (flagged below_min_pixels). "
    f"Row '0' is open sky (id 0), classified by the same rule; it never receives detail."
)

DETAIL_SIGMA_FINE = 0.7      # removes per-pixel noise (< ~2 px)
DETAIL_SIGMA_COARSE = 3.5    # removes shading at > ~12 px
DETAIL_GAIN = 2.5            # height offset per unit of band-passed luma (0..1)
DETAIL_ERODE_PX = 3          # keep detail away from silhouettes
DETAIL_ACTIVE_EPS = 1.0 / 255.0

FONT_CANDIDATES = ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]
MONO_CANDIDATES = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Courier.ttc",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVuSansMono.ttf"]


# ---------------------------------------------------------------- io
def load_rgb(path: Path, size: tuple[int, int] | None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != tuple(size):
        raise SystemExit(f"{path}: size {im.size} != case size {tuple(size)}; inputs must be "
                         "pixel-registered, refusing to resample")
    return np.asarray(im, dtype=np.uint8)


def object_names(case: Path, k: int) -> dict[int, dict]:
    """id -> {"name", "category"} from objects.json (blockout) and objects_visible.json (gbuffers)."""
    out: dict[int, dict] = {}
    p = case / "blockout" / f"iter{k}" / "objects.json"
    if p.exists():
        data = common.read_json(p)
        rows = data.get("objects", data) if isinstance(data, dict) else data
        for o in rows:
            out[int(o["id"])] = {"name": o.get("name", f"id{o['id']}"), "category": o.get("category", "")}
    p = case / "gbuffers" / f"iter{k}" / "objects_visible.json"
    if p.exists():
        for key, v in common.read_json(p).items():
            e = out.setdefault(int(key), {"name": v.get("name", f"id{key}"), "category": v.get("category", "")})
            e.setdefault("name", v.get("name", f"id{key}"))
    return out


def load_ids(path: Path, size: tuple[int, int] | None) -> np.ndarray:
    ids = np.asarray(Image.open(path)).astype(np.int32)
    if ids.ndim == 3:
        ids = ids[..., 0]
    if size is not None and ids.shape != (size[1], size[0]):
        raise SystemExit(f"{path}: shape {ids.shape} != case size {(size[1], size[0])}")
    return ids


def tagged(name: str, tag: str | None) -> str:
    if not tag:
        return name
    stem, ext = name.rsplit(".", 1)
    return f"{stem}_{tag}.{ext}"


def rel(case: Path, p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(case))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- metrics
def pick_device(req: str):
    import torch
    if req == "cpu":
        return torch.device("cpu")
    if req in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if req in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if req in ("mps", "cuda"):
        raise SystemExit(f"{req} requested but not available")
    return torch.device("cpu")


class Metrics:
    """LPIPS (Alex, Alex spatial, VGG), PSNR and SSIM. torch + lpips are imported here only."""

    def __init__(self, device):
        import warnings
        warnings.filterwarnings("ignore", message=".*(pretrained|weight enum).*")
        import lpips
        self.torch = __import__("torch")
        self.device = device
        self.alex = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.alex_sp = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()
        self.vgg = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    def _t(self, img: np.ndarray):
        t = self.torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        return (t * 2.0 - 1.0).to(self.device)

    def pair(self, a: np.ndarray, b: np.ndarray, spatial: bool = True):
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        with self.torch.no_grad():
            ta, tb = self._t(a), self._t(b)
            out = {
                "lpips_alex": float(self.alex(ta, tb).item()),
                "lpips_vgg": float(self.vgg(ta, tb).item()),
                "psnr_db": float(peak_signal_noise_ratio(b, a, data_range=255)),
                "ssim": float(structural_similarity(b, a, channel_axis=2, data_range=255)),
            }
            smap = None
            if spatial:
                smap = self.alex_sp(ta, tb)[0, 0].float().cpu().numpy()
                if smap.shape != a.shape[:2]:
                    smap = cv2.resize(smap, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out, smap


def lab(img: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2lab
    with np.errstate(all="ignore"):  # numpy + Accelerate raises spurious FP flags in this matmul
        out = rgb2lab(img.astype(np.float64) / 255.0)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite CIELAB values")
    return out


def coarse(img: np.ndarray) -> np.ndarray:
    return np.clip(cv2.GaussianBlur(img.astype(np.float32), (0, 0), COARSE_SIGMA) + 0.5, 0, 255).astype(np.uint8)


class Eval:
    """Per-pixel quantities for one (image, target) pair, fine and coarse."""

    def __init__(self, m, img, target, target_c=None, lab_tg=None, lab_tg_c=None):
        self.g, self.sp = m.pair(img, target)
        tc = coarse(target) if target_c is None else target_c
        _, self.sp_c = m.pair(coarse(img), tc, spatial=True)
        lt = lab(target) if lab_tg is None else lab_tg
        ltc = lab(tc) if lab_tg_c is None else lab_tg_c
        li, lic = lab(img), lab(coarse(img))
        self.dE = np.sqrt(((li - lt) ** 2).sum(-1))
        self.dE_c = np.sqrt(((lic - ltc) ** 2).sum(-1))
        self.dL = li[..., 0] - lt[..., 0]


# ---------------------------------------------------------------- per-object rule (pure numpy)
def classify(pixels_eroded: int, coarse_gap_ratio: float, coarse_de: float) -> tuple[str, list[str]]:
    """The small / large rule. Returns (class, reasons)."""
    reasons = []
    if coarse_gap_ratio > GAP_RATIO_MAX:
        reasons.append(f"coarse_gap_ratio {coarse_gap_ratio:.2f} > {GAP_RATIO_MAX}")
    if coarse_de > DELTA_E_MAX:
        reasons.append(f"coarse_dE76 {coarse_de:.1f} > {DELTA_E_MAX}")
    large = pixels_eroded >= MIN_PIXELS and bool(reasons)
    return ("large" if large else "small"), (reasons if large else [])


def per_object(ids: np.ndarray, names: dict, ev, ev_bl) -> dict:
    """ev / ev_bl need .sp, .sp_c, .dE, .dE_c, .dL arrays (re-render and blockout against the target)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_ERODE_PX + 1,) * 2)
    objs = {}
    for oid in sorted(int(i) for i in np.unique(ids)):
        full = ids == oid
        er = cv2.erode(full.astype(np.uint8), kernel) > 0
        used = er if er.sum() >= 50 else full
        n_er = int(er.sum())
        l_re, l_bl = float(ev.sp[used].mean()), float(ev_bl.sp[used].mean())
        lc_re, lc_bl = float(ev.sp_c[used].mean()), float(ev_bl.sp_c[used].mean())
        ratio_fine = l_re / max(l_bl, GAP_DENOM_FLOOR)
        ratio = lc_re / max(lc_bl, GAP_DENOM_FLOOR)
        de_fine, de = float(ev.dE[used].mean()), float(ev.dE_c[used].mean())
        cls, reasons = classify(n_er, ratio, de)
        dl = float(ev.dL[used].mean())
        sky = oid == SKY_ID
        meta = names.get(oid, {})
        objs[str(oid)] = {
            "name": SKY_NAME if sky else meta.get("name", f"id{oid}"),
            "category": "open_sky" if sky else meta.get("category", ""),
            "is_open_sky": sky,
            "pixels": int(full.sum()),
            "pixels_eroded": n_er,
            "stats_mask": "eroded" if used is er else "full (eroded mask too small)",
            "lpips_rerender_target": l_re,
            "lpips_blockout_target": l_bl,
            "gap_ratio": ratio_fine,
            "mean_dE76": de_fine,
            "coarse_lpips_rerender_target": lc_re,
            "coarse_lpips_blockout_target": lc_bl,
            "coarse_gap_ratio": ratio,
            "coarse_mean_dE76": de,
            "mean_signed_dL": dl,
            "brightness": "too dark" if dl < -3 else "too bright" if dl > 3 else "ok (|dL*|<=3)",
            "below_min_pixels": n_er < MIN_PIXELS,
            "class": cls,
            "large_reasons": reasons,
        }
    return objs


# ---------------------------------------------------------------- detail map
def detail_height(target, rerender, ids, objs):
    def luma(img):
        f = img.astype(np.float32) / 255.0
        return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]

    def band(y):
        return cv2.GaussianBlur(y, (0, 0), DETAIL_SIGMA_FINE) - cv2.GaussianBlur(y, (0, 0), DETAIL_SIGMA_COARSE)

    resid = band(luma(target)) - band(luma(rerender))
    small_ids = [int(k) for k, v in objs.items() if v["class"] == "small" and not v.get("is_open_sky")]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DETAIL_ERODE_PX + 1,) * 2)
    mask = np.zeros(ids.shape, np.uint8)
    for oid in small_ids:
        mask |= cv2.erode((ids == oid).astype(np.uint8), kernel)
    w = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0) * mask
    h = np.clip(0.5 + DETAIL_GAIN * resid * w, 0.0, 1.0)
    active = np.abs(h - 0.5) > DETAIL_ACTIVE_EPS
    n, nm = ids.size, max(int((mask > 0).sum()), 1)
    info = {
        "method": ("height = 0.5 + gain * [DoG(Y_target) - DoG(Y_rerender)] on eroded masks of 'small' objects; "
                   "Y = Rec.709 luma of sRGB values in 0..1; DoG = G(sigma_fine) - G(sigma_coarse). Heuristic "
                   "shape-from-shading (bright = higher, dark = lower), not measured geometry."),
        "band_px": {"sigma_fine": DETAIL_SIGMA_FINE, "sigma_coarse": DETAIL_SIGMA_COARSE,
                    "approx_feature_size_px": [2, 12]},
        "gain": DETAIL_GAIN,
        "neutral_value": "0.5 (32768 in 16-bit) = no change",
        "encoding": "16-bit grey PNG, value/65535 = relative height in 0..1, clipped",
        "mask_erode_px": DETAIL_ERODE_PX,
        "small_object_ids": small_ids,
        "small_object_names": [objs[str(i)]["name"] for i in small_ids],
        "masked_fraction_of_image": float((mask > 0).sum() / n),
        "active_fraction_of_image": float(active.sum() / n),
        "active_fraction_of_masked": float(active.sum() / nm),
        "clipped_fraction_of_masked": float(((h <= 0) | (h >= 1)).sum() / nm),
        "residual_band_std_in_mask": float(resid[mask > 0].std()) if mask.any() else 0.0,
        "per_object_residual_std": {objs[str(i)]["name"]: float(resid[ids == i].std()) for i in small_ids},
    }
    return (h * 65535.0 + 0.5).astype(np.uint16), info


# ---------------------------------------------------------------- visuals
def font(size, mono=False):
    for p in (MONO_CANDIDATES if mono else FONT_CANDIDATES):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def colorize(smap: np.ndarray) -> np.ndarray:
    v = np.clip(smap / ERROR_MAP_MAX, 0, 1)
    bgr = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def contact_sheet(blockout, target, rerender, err_rgb, ids, objs, metrics, title, out, panel_w=575):
    H0, W0 = ids.shape
    pw = min(panel_w, W0)
    ph = round(pw * H0 / W0)
    gap, margin = 8, 16
    W = 4 * pw + 3 * gap + 2 * margin
    f_title, f_lab, f_txt, f_mono = font(30), font(22), font(18), font(16, mono=True)

    def panel(img):
        return cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)

    err_p = panel(err_rgb).copy()
    for k, o in objs.items():
        m = cv2.resize((ids == int(k)).astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if o["class"] == "large":
            cv2.drawContours(err_p, cs, -1, (255, 40, 40), 3, lineType=cv2.LINE_AA)
        elif not o.get("is_open_sky"):
            cv2.drawContours(err_p, cs, -1, (90, 220, 255), 1, lineType=cv2.LINE_AA)

    rows = sorted(objs.items(), key=lambda kv: -kv[1]["pixels"])
    line_h, header_h = 22, 96
    table_h = 40 + line_h * (len(rows) + 1)
    Ht = header_h + 34 + ph + 16 + 46 + table_h + margin
    sheet = Image.new("RGB", (W, Ht), (250, 250, 248))
    d = ImageDraw.Draw(sheet)
    d.text((margin, 12), title, font=f_title, fill=(20, 20, 20))
    r, b = metrics["rerender_vs_target"], metrics["blockout_vs_target"]
    d.text((margin, 54),
           f"re-render vs target:  LPIPS-Alex {r['lpips_alex']:.4f}   LPIPS-VGG {r['lpips_vgg']:.4f}   "
           f"PSNR {r['psnr_db']:.2f} dB   SSIM {r['ssim']:.4f}          "
           f"baseline blockout vs target:  LPIPS-Alex {b['lpips_alex']:.4f}   LPIPS-VGG {b['lpips_vgg']:.4f}   "
           f"PSNR {b['psnr_db']:.2f} dB   SSIM {b['ssim']:.4f}", font=f_txt, fill=(40, 40, 40))
    labels = ["blockout (grey baseline)", "target", "re-render", f"error: spatial LPIPS-Alex, fixed 0..{ERROR_MAP_MAX:g}"]
    y0 = header_h + 34
    for i, (lab_, im) in enumerate(zip(labels, [panel(blockout), panel(target), panel(rerender), err_p])):
        x = margin + i * (pw + gap)
        d.text((x, header_h + 4), lab_, font=f_lab, fill=(20, 20, 20))
        sheet.paste(Image.fromarray(im), (x, y0))
    x = margin + 3 * (pw + gap)
    yb = y0 + ph + 10
    bar = colorize(np.tile(np.linspace(0, ERROR_MAP_MAX, max(pw - 80, 10))[None], (14, 1)))
    sheet.paste(Image.fromarray(bar), (x + 40, yb))
    d.text((x, yb - 2), "0", font=f_txt, fill=(40, 40, 40))
    d.text((x + pw - 34, yb - 2), f"{ERROR_MAP_MAX:g}", font=f_txt, fill=(40, 40, 40))
    d.text((margin, yb + 20), "outline: red = large (edit geometry / material / exterior), cyan = small (detail map); "
           "row 0 = open sky (no object)", font=f_txt, fill=(40, 40, 40))
    ty = yb + 52
    d.text((margin, ty), "per object (eroded masks)   rule: " + CLASS_RULE[:CLASS_RULE.index(";")],
           font=f_txt, fill=(20, 20, 20))
    ty += 30
    hdr = (f"{'id':>3}  {'name':<22}{'pixels':>8}  {'LPIPS re':>8}  {'LPIPS bl':>8}  {'gap':>5}  {'dE76':>5}  "
           f"{'dL*':>6}  | {'coarse gap':>10}  {'coarse dE':>9}  class")
    d.text((margin, ty), hdr, font=f_mono, fill=(20, 20, 20))
    for j, (k, o) in enumerate(rows):
        s = (f"{k:>3}  {o['name'][:21]:<22}{o['pixels']:>8}  {o['lpips_rerender_target']:>8.3f}  "
             f"{o['lpips_blockout_target']:>8.3f}  {o['gap_ratio']:>5.2f}  {o['mean_dE76']:>5.1f}  "
             f"{o['mean_signed_dL']:>+6.1f}  | {o['coarse_gap_ratio']:>10.2f}  {o['coarse_mean_dE76']:>9.1f}  "
             f"{o['class']}" + ("  (tiny: below min pixels)" if o["below_min_pixels"] else "")
             + (f"  <- {'; '.join(o['large_reasons'])}" if o["large_reasons"] else ""))
        d.text((margin, ty + line_h * (j + 1)), s, font=f_mono,
               fill=(200, 20, 20) if o["class"] == "large" else (30, 30, 30))
    sheet.save(out)


# ---------------------------------------------------------------- calibration
def calibrate(m, target, blockout, ids, names, ev_bl, tg_c, lab_tg, lab_tg_c):
    f = target.astype(np.float32)
    variants = {
        "blur_sigma1": cv2.GaussianBlur(target, (0, 0), 1.0),
        "blur_sigma2": cv2.GaussianBlur(target, (0, 0), 2.0),
        "exposure_+10pct": np.clip(f * 1.10, 0, 255).astype(np.uint8),
        "exposure_-15pct": np.clip(f * 0.85, 0, 255).astype(np.uint8),
        "warm_shift_rgb_+6_0_-6": np.clip(f + np.array([6, 0, -6]), 0, 255).astype(np.uint8),
        "blur1_exposure_-10pct": np.clip(cv2.GaussianBlur(f, (0, 0), 1.0) * 0.9, 0, 255).astype(np.uint8),
        "blockout": blockout,
    }
    refs = {}
    for name, img in variants.items():
        ev = Eval(m, img, target, tg_c, lab_tg, lab_tg_c)
        objs = per_object(ids, names, ev, ev_bl)
        big = [o for o in objs.values() if not o["below_min_pixels"]] or list(objs.values())
        refs[name] = {
            "global": ev.g,
            "per_object_max_gap_ratio": max(o["coarse_gap_ratio"] for o in big),
            "per_object_max_dE76": max(o["coarse_mean_dE76"] for o in big),
            "per_object_min_gap_ratio": min(o["coarse_gap_ratio"] for o in big),
            "per_object_min_dE76": min(o["coarse_mean_dE76"] for o in big),
            "n_large": sum(o["class"] == "large" for o in objs.values()),
            "n_objects": len(objs),
            "large_names": [o["name"] for o in objs.values() if o["class"] == "large"],
        }
    return refs


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--rerender")
    ap.add_argument("--target")
    ap.add_argument("--blockout")
    ap.add_argument("--tag")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    a = ap.parse_args(argv)

    t0 = time.time()
    case = common.case_dir(a.case)
    size = common.render_size(case)
    out_dir = common.stage_dir(case, "residual", a.iter)
    gbuf = common.stage_dir(case, "gbuffers", a.iter, create=False)
    p_re = Path(a.rerender) if a.rerender else out_dir / "rerender.png"
    p_tg = Path(a.target) if a.target else common.stage_dir(case, "target", a.iter, create=False) / "target.png"
    p_bl = Path(a.blockout) if a.blockout else gbuf / "beauty.png"
    p_id = gbuf / "id.png"
    missing = [str(p) for p in (p_re, p_tg, p_bl, p_id) if not p.exists()]
    if missing:
        print(f"[residual] missing inputs: {missing}", file=sys.stderr)
        return 2
    if a.tag and not a.tag.replace("_", "").replace("-", "").isalnum():
        print(f"[residual] bad tag {a.tag!r}", file=sys.stderr)
        return 2

    rerender, target, blockout = load_rgb(p_re, size), load_rgb(p_tg, size), load_rgb(p_bl, size)
    ids = load_ids(p_id, size)
    names = object_names(case, a.iter)
    device = pick_device(a.device)
    m = Metrics(device)
    t_model = time.time()

    tg_c = coarse(target)
    lab_tg, lab_tg_c = lab(target), lab(tg_c)
    ev_re = Eval(m, rerender, target, tg_c, lab_tg, lab_tg_c)
    ev_bl = Eval(m, blockout, target, tg_c, lab_tg, lab_tg_c)
    g_re, g_bl = ev_re.g, ev_bl.g
    objs = per_object(ids, names, ev_re, ev_bl)
    t_eval = time.time()

    T = lambda n: out_dir / tagged(n, a.tag)  # noqa: E731
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = {"rerender_vs_target": g_re, "blockout_vs_target": g_bl}
    np.save(T("error_map.npy"), ev_re.sp.astype(np.float32))
    err_rgb = colorize(ev_re.sp)
    Image.fromarray(err_rgb).save(T("error_map.png"))

    h16, dinfo = detail_height(target, rerender, ids, objs)
    Image.fromarray(h16).save(T("detail_height.png"))
    common.write_json(T("detail.json"), {"created_utc": now, "rerender": rel(case, p_re),
                                         "target": rel(case, p_tg), **dinfo})

    n_large = sum(o["class"] == "large" for o in objs.values())
    sky = objs.get(str(SKY_ID))
    common.write_json(T("per_object.json"), {
        "created_utc": now, "rule": CLASS_RULE,
        "thresholds": {"MIN_PIXELS": MIN_PIXELS, "GAP_RATIO_MAX": GAP_RATIO_MAX, "DELTA_E_MAX": DELTA_E_MAX,
                       "GAP_DENOM_FLOOR": GAP_DENOM_FLOOR, "COARSE_SIGMA": COARSE_SIGMA,
                       "MASK_ERODE_PX": MASK_ERODE_PX},
        "spatial_metric": "LPIPS-Alex spatial=True, [-1,1] inputs",
        "colour_metric": "CIELAB dE76 (skimage rgb2lab, sRGB D65); dL* = L*(rerender) - L*(target)",
        "open_sky_row": "row '0' = pixels with id 0 (no object): sky / view through openings",
        "n_large": n_large, "n_small": len(objs) - n_large,
        "large_names": [o["name"] for o in objs.values() if o["class"] == "large"],
        "objects": objs,
    })

    title = (f"{case.name}  iter {a.iter}" + (f"  [{a.tag}]" if a.tag else "")
             + f"   LPIPS-Alex re-render vs target = {g_re['lpips_alex']:.4f}"
             + f"  (blockout baseline {g_bl['lpips_alex']:.4f})")
    contact_sheet(blockout, target, rerender, err_rgb, ids, objs, metrics, title, T("contact_sheet.png"))

    if a.calibrate:
        tc = time.time()
        calib = calibrate(m, target, blockout, ids, names, ev_bl, tg_c, lab_tg, lab_tg_c)
        common.write_json(T("calibration.json"), {"created_utc": now, "rule": CLASS_RULE, "references": calib,
                                                  "seconds": round(time.time() - tc, 2)})

    common.write_json(T("lpips.json"), {
        "created_utc": now,
        "headline": {"metric": "lpips_alex(rerender, target)", "value": g_re["lpips_alex"]},
        "rerender_vs_target": g_re,
        "blockout_vs_target": g_bl,
        "lpips_input_scaling": "[-1, 1] (x*2-1 from 0..1 sRGB), full image, lpips linear v0.1",
        "images": {"rerender": rel(case, p_re), "target": rel(case, p_tg), "blockout": rel(case, p_bl),
                   "ids": rel(case, p_id)},
        "tag": a.tag,
        "device": str(device),
        "seconds": {"model_init_and_load": round(t_model - t0, 2), "evaluate": round(t_eval - t_model, 2),
                    "total": round(time.time() - t0, 2)},
    })
    print(f"[residual] device={device} LPIPS-Alex re={g_re['lpips_alex']:.4f} bl={g_bl['lpips_alex']:.4f} "
          f"VGG re={g_re['lpips_vgg']:.4f}  large={n_large}/{len(objs)} -> {out_dir}"
          + (f" (tag {a.tag})" if a.tag else ""))
    for k, o in sorted(objs.items(), key=lambda kv: int(kv[0])):
        print(f"   {k:>3} {o['name'][:22]:<22} {o['class']:<5} cgap={o['coarse_gap_ratio']:.2f} "
              f"cdE={o['coarse_mean_dE76']:.1f} | gap={o['gap_ratio']:.2f} dE={o['mean_dE76']:.1f} "
              f"dL={o['mean_signed_dL']:+.1f} px={o['pixels']}")
    if sky is not None:
        print(f"[residual] open sky (id 0): {sky['pixels']} px, class {sky['class']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
