"""Target stage: a photoreal image of the case's fixed view (Nano Banana 2 edit on fal), or your own.

    python -m b2r.target --case CASE --iter K [--levels all|L3|L2,L3] [--seed N] [--no-reference]
                         [--extra-prompt TEXT]
    python -m b2r.target --case CASE --iter K --image PHOTO.png        bring your own target (no ladder)
    python -m b2r.target --case CASE --iter K --set-level L3 --reason "why"   override the rule's pick
    python -m b2r.target --case CASE --iter K --rescore [--out-dir DIR]       re-score sweep/, no fal calls

Nano Banana 2 has no numeric conditioning scale, so "conditioning strength" is a ladder L1..L5 of how
much geometry the model is given and how strictly the prompt binds it:
    L1 depth only, loose   L2 grey render, loose   L3 grey render, strict
    L4 render + depth + normals, strict            L5 + object-id colours, strictest
Every level is scored by how well the target's luminance edges line up with the object-id boundaries
(`edge_scores`). The WEAKEST level that passes an explicit rule, calibrated against the grey render
(upper row) and each image flipped horizontally (chance row), becomes target.png. If none passes the
stage exits 2 and writes no target.

Prompts come only from case.json "look" and objects.json descriptions. Iteration 0 runs the whole
ladder and records the chosen level in case.json settings.target_level; later iterations run that
level and pass the previous iteration's target as a look reference (unless --no-reference).

Size: the request asks for case.json size.aspect_ratio / size.resolution. If the generator returns a
different size and size_verified is false, the real size is written to case.json (and to a user-level
cache), the image is kept under sweep/ as evidence and the stage exits 3 so the g-buffers are re-rendered.
If the size was already verified, a different size is an error. Nothing is ever cropped or padded.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import common
from .target_prompts import (ALL_LEVELS, HIDDEN_OCC, PARTLY_OCC, SMALL_FRAC, build_level, category,
                             id_colours, object_lines, visibility)

EXIT_SIZE_CHANGED = 3
EXIT_NO_PASS = 2
IMAGE_LEVEL = "IMAGE"  # the row name of a bring-your-own target in choice.json

# ---------------------------------------------------------------- edge metric
EDGE_SIGMA = 2.0  # Gaussian blur (px) before Canny: ignores fine texture/noise
CANNY_LO, CANNY_HI = 5, 12  # on 8-bit luminance, L2 gradient
RADIUS = 3  # recall tolerance in px

# Pass rule. Constants fixed before the experiment's sweep was run; the bound is expressed relative to the
# calibration rows so it adapts to the scene.
REL_RECALL = 0.90  # recall@3 >= REL_RECALL * best level's recall@3
DIST_FRAC = 0.35  # mean_dist <= upper_md + DIST_FRAC * (chance_md - upper_md)
MARGIN_OVER_CHANCE = 0.20  # recall@3 >= own flipped recall@3 + margin
OBJ_MIN_PX = 400  # non-structure objects with at least this many boundary px are checked...
OBJ_FLOOR = 0.40  # ...and each must have recall@3 >= OBJ_FLOOR

ASPECT_TOL = 0.01  # --image: relative aspect difference still treated as "the case aspect ratio"

KNOWN_BLIND_SPOT = (
    "The edge rule only measures recall of ID-mask boundaries. It cannot detect added or moved objects "
    "(or recoloured or enlarged ones) that leave the ID outlines intact, e.g. an object pulled in front of "
    "another, or a framed object drawn larger while its other edges still match. Check sweep_contact.png and "
    "the sweep/*_overlay.png images by eye and record a correction with --set-level LN --reason TEXT."
)


def luminance_edges(rgb: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), EDGE_SIGMA)
    return cv2.Canny(g, CANNY_LO, CANNY_HI, L2gradient=True) > 0


def id_boundaries(id_map: np.ndarray) -> np.ndarray:
    """Pixels whose id differs from a 4-neighbour; the outermost 2 px are excluded (Canny never fires there)."""
    idm = np.asarray(id_map)
    b = np.zeros(idm.shape, bool)
    d = idm[:, 1:] != idm[:, :-1]
    b[:, 1:] |= d
    b[:, :-1] |= d
    d = idm[1:, :] != idm[:-1, :]
    b[1:, :] |= d
    b[:-1, :] |= d
    b[:2, :] = b[-2:, :] = False
    b[:, :2] = b[:, -2:] = False
    return b


def edge_scores(id_map: np.ndarray, image_rgb: np.ndarray) -> dict:
    """Pure edge-alignment metric. id_map: (H, W) integer ids; image_rgb: (H, W, 3) uint8.
    Returns recall3 (share of id-boundary px within RADIUS px of a luminance edge), mean_dist_px
    (mean distance from an id-boundary px to the nearest edge), edge_density and per_object
    {id: {boundary_px, recall3}}."""
    import cv2

    idm = np.asarray(id_map).astype(np.int64)
    rgb = np.asarray(image_rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"image_rgb must be (H, W, 3); got {rgb.shape}")
    rgb = rgb[..., :3]
    if rgb.shape[:2] != idm.shape:
        raise ValueError(f"image {rgb.shape[:2]} and id map {idm.shape} differ in size")
    bnd = id_boundaries(idm)
    edges = luminance_edges(rgb)
    dist = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 5)
    d = dist[bnd]
    per_obj = {}
    for oid in np.unique(idm):
        m = bnd & (idm == oid)
        n = int(m.sum())
        if n == 0:
            continue
        per_obj[int(oid)] = {"boundary_px": n, "recall3": round(float((dist[m] <= RADIUS).mean()), 4)}
    return {
        "recall3": round(float((d <= RADIUS).mean()), 4) if d.size else 0.0,
        "mean_dist_px": round(float(d.mean()), 3) if d.size else float("nan"),
        "edge_density": round(float(edges.mean()), 4),
        "per_object": per_obj,
    }


def evaluate(results: dict, upper: dict, levels: list, cats: dict, names: dict) -> tuple[str, dict, dict]:
    """Apply the pass rule in place (results[lv]['pass'], ['fail_reasons']). Returns (rule text, chance row, bounds)."""
    chance_md = float(np.mean([results[l]["flipped_scores"]["mean_dist_px"] for l in levels]))
    chance_rc = float(np.mean([results[l]["flipped_scores"]["recall3"] for l in levels]))
    best = max(results[l]["scores"]["recall3"] for l in levels)
    md_bound = upper["mean_dist_px"] + DIST_FRAC * (chance_md - upper["mean_dist_px"])
    rule = (
        f"PASS iff (a) recall@{RADIUS}px >= {REL_RECALL} x best row recall ({best:.4f}) = {REL_RECALL * best:.4f}; "
        f"(b) mean distance <= upper_md + {DIST_FRAC} x (chance_md - upper_md) = "
        f"{upper['mean_dist_px']:.3f} + {DIST_FRAC} x ({chance_md:.3f} - {upper['mean_dist_px']:.3f}) = {md_bound:.3f} px; "
        f"(c) recall@{RADIUS}px >= the same image flipped horizontally + {MARGIN_OVER_CHANCE}; "
        f"(d) every object whose category is not 'structure' with >= {OBJ_MIN_PX} ID-boundary px has "
        f"recall@{RADIUS}px >= {OBJ_FLOOR}. upper = grey beauty.png, chance = mean over rows of the horizontally "
        f"flipped image. Chosen = weakest passing level."
    )
    for lv in levels:
        s, fs = results[lv]["scores"], results[lv]["flipped_scores"]
        fails = []
        if s["recall3"] < REL_RECALL * best:
            fails.append(f"a: recall below {REL_RECALL} x best")
        if s["mean_dist_px"] > md_bound:
            fails.append(f"b: mean dist {s['mean_dist_px']} > {md_bound:.3f}")
        if s["recall3"] < fs["recall3"] + MARGIN_OVER_CHANCE:
            fails.append("c: not clearly above own flipped control")
        weak = [f"{names.get(o, o)}={v['recall3']}" for o, v in s["per_object"].items()
                if o != 0 and cats.get(o, "") != "structure" and v["boundary_px"] >= OBJ_MIN_PX and v["recall3"] < OBJ_FLOOR]
        if weak:
            fails.append("d: objects below floor: " + ", ".join(weak))
        results[lv]["pass"] = not fails
        results[lv]["fail_reasons"] = fails
    chance = {"recall3": round(chance_rc, 4), "mean_dist_px": round(chance_md, 3)}
    return rule, chance, {"best_recall3": best, "mean_dist_bound_px": round(md_bound, 3)}


# ---------------------------------------------------------------- size knowledge
def _size_key(aspect: str, resolution: str) -> str:
    return f"{aspect}@{resolution}"


def user_sizes_path() -> Path:
    return common.cache_dir() / "nb2_sizes_learned.json"


def _lookup(table, aspect: str, resolution: str):
    """Tolerant reader: {"16:9@1K": {width,height}|[w,h]} or {"1K": {"16:9": ...}} or {"16:9": {"1K": ...}}."""
    if not isinstance(table, dict):
        return None
    cands = [table.get(_size_key(aspect, resolution))]
    for a, b in ((resolution, aspect), (aspect, resolution)):
        sub = table.get(a)
        if isinstance(sub, dict):
            cands.append(sub.get(b))
    for sub in table.values():  # e.g. {"sizes": {...}}
        if isinstance(sub, dict) and _size_key(aspect, resolution) in sub:
            cands.append(sub[_size_key(aspect, resolution)])
    for c in cands:
        if isinstance(c, dict) and "width" in c and "height" in c:
            return int(c["width"]), int(c["height"])
        if isinstance(c, (list, tuple)) and len(c) == 2:
            return int(c[0]), int(c[1])
    return None


def known_size(aspect: str, resolution: str) -> tuple[tuple[int, int], str] | None:
    """(size, source) from the user-level cache, then the repo's nb2_sizes.json (owned elsewhere)."""
    for p in (user_sizes_path(), Path(__file__).resolve().parent / "nb2_sizes.json"):
        if p.exists():
            try:
                hit = _lookup(common.read_json(p), aspect, resolution)
            except Exception:  # noqa: BLE001 - a malformed cache must not stop the stage; it is just not used
                hit = None
            if hit:
                return hit, str(p.name)
    return None


def remember_size(aspect: str, resolution: str, w: int, h: int, endpoint: str) -> None:
    p = user_sizes_path()
    table = {}
    if p.exists():
        try:
            table = common.read_json(p)
        except Exception:  # noqa: BLE001
            table = {}
    table[_size_key(aspect, resolution)] = {"width": int(w), "height": int(h), "endpoint": endpoint,
                                            "learned": time.strftime("%Y-%m-%dT%H:%M:%S")}
    common.write_json(p, table)


def set_case_size(case, w: int, h: int, how: str) -> dict:
    cfg = common.load_case(case)
    old = dict(cfg["size"])
    cfg["size"].update(width=int(w), height=int(h), size_verified=True)
    cfg["size"]["size_learned"] = {"from": how, "previous": [old.get("width"), old.get("height")],
                                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    common.save_case(case, cfg)
    return cfg


# ---------------------------------------------------------------- io helpers
def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlay(rgb: np.ndarray, bnd: np.ndarray) -> np.ndarray:
    o = rgb.copy()
    o[bnd] = (255, 0, 0)
    return o


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def contact_sheet(rows: list, path: Path, W: int, H: int) -> None:
    """rows: [(label, rgb, overlay_rgb)]"""
    tw, th, lab = W // 2, H // 2, 44
    sheet = Image.new("RGB", (2 * tw, len(rows) * (th + lab)), "white")
    d = ImageDraw.Draw(sheet)
    f = _font(20)
    for i, (label, a, b) in enumerate(rows):
        y = i * (th + lab)
        d.text((8, y + 10), label, fill="black", font=f)
        sheet.paste(Image.fromarray(a).resize((tw, th), Image.LANCZOS), (0, y + lab))
        sheet.paste(Image.fromarray(b).resize((tw, th), Image.NEAREST), (tw, y + lab))
    sheet.save(path)


def rel(case, p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(common.case_dir(case)))
    except ValueError:
        return str(p)


def fail(msg: str, code: int = 1) -> int:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    return code


# ---------------------------------------------------------------- the stage
class Inputs:
    """G-buffers and object tables of one iteration."""

    def __init__(self, case, k: int):
        g = common.stage_dir(case, "gbuffers", k, create=False)
        b = common.stage_dir(case, "blockout", k, create=False)
        need = [g / f for f in ("beauty.png", "depth.png", "normal_cam.png", "id.png", "objects_visible.json",
                                "camera.json")] + [b / "objects.json"]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            raise SystemExit(f"target: missing inputs (run blockout / gbuffers for iter {k} first): {missing}")
        self.g = g
        self.idm = np.array(Image.open(g / "id.png")).astype(np.int32)
        if self.idm.ndim == 3:
            self.idm = self.idm[..., 0]
        self.beauty = load_rgb(g / "beauty.png")
        self.objects = common.read_json(b / "objects.json")
        self.visible = common.read_json(g / "objects_visible.json")
        self.cam = common.read_json(g / "camera.json")
        self.depth16 = np.array(Image.open(g / "depth.png")).astype(np.float64)
        if self.depth16.ndim == 3:
            self.depth16 = self.depth16[..., 0]
        self.names = {o["id"]: o["name"] for o in self.objects}
        self.cats = {o["id"]: category(o) for o in self.objects}
        self.H, self.W = self.idm.shape
        for name, arr in (("beauty.png", self.beauty), ("depth.png", self.depth16)):
            if arr.shape[:2] != (self.H, self.W):
                raise SystemExit(f"target: {name} is {arr.shape[:2]} but id.png is {(self.H, self.W)}")
        self.bnd = id_boundaries(self.idm)

    def size(self) -> tuple[int, int]:
        return self.W, self.H


def score_rows(inp: Inputs, images: dict, results: dict, out_sweep: Path | None) -> dict:
    """images: level -> rgb. Fills results[lv]['scores'/'flipped_scores'], writes overlays. Returns upper row."""
    upper = edge_scores(inp.idm, inp.beauty)
    for lv, rgb in images.items():
        results.setdefault(lv, {})
        results[lv]["scores"] = edge_scores(inp.idm, rgb)
        results[lv]["flipped_scores"] = {k: v for k, v in edge_scores(inp.idm, rgb[:, ::-1].copy()).items()
                                         if k != "per_object"}
        if out_sweep is not None:
            Image.fromarray(overlay(rgb, inp.bnd)).save(out_sweep / f"{lv}_overlay.png")
    return upper


def table_text(inp: Inputs, upper: dict, results: dict, levels: list, chance: dict) -> str:
    hdr = f"{'row':<14}{'recall@3':>9}{'mean_d':>8}{'flip_rc':>8}{'flip_md':>8}{'edges':>7}  pass"
    lines = [hdr, f"{'beauty(upper)':<14}{upper['recall3']:>9.3f}{upper['mean_dist_px']:>8.2f}{'':>8}{'':>8}"
                  f"{upper['edge_density']:>7.3f}"]
    for lv in levels:
        s, fs = results[lv]["scores"], results[lv]["flipped_scores"]
        lines.append(f"{lv:<14}{s['recall3']:>9.3f}{s['mean_dist_px']:>8.2f}{fs['recall3']:>8.3f}{fs['mean_dist_px']:>8.2f}"
                     f"{s['edge_density']:>7.3f}  {'PASS' if results[lv]['pass'] else 'fail: ' + '; '.join(results[lv]['fail_reasons'])}")
    lines.append(f"{'chance(mean)':<14}{chance['recall3']:>9.3f}{chance['mean_dist_px']:>8.2f}")
    ids = sorted({o for l in levels for o in results[l]["scores"]["per_object"]})
    lines.append("\nper-object recall@3 (boundary px):")
    lines.append(f"{'id name':<22}{'npx':>6}{'beauty':>8}" + "".join(f"{l:>7}" for l in levels))
    for o in ids:
        if o == 0:
            continue
        lines.append(f"{(str(o) + ' ' + str(inp.names.get(o, '?')))[:21]:<22}"
                     f"{upper['per_object'].get(o, {}).get('boundary_px', 0):>6}"
                     f"{upper['per_object'].get(o, {}).get('recall3', float('nan')):>8.2f}"
                     + "".join(f"{results[l]['scores']['per_object'].get(o, {}).get('recall3', float('nan')):>7.2f}"
                               for l in levels))
    return "\n".join(lines)


def write_outputs(case, k: int, out: Path, inp: Inputs, images: dict, results: dict, levels: list,
                  extra_fields: dict) -> tuple[dict, str | None]:
    """Score, apply the rule, write scores.txt, sweep_contact.png, choice.json. Returns (choice, chosen)."""
    sweep = out / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    upper = score_rows(inp, images, results, sweep)
    rule, chance, bounds = evaluate(results, upper, levels, inp.cats, inp.names)
    chosen = next((l for l in levels if results[l]["pass"]), None)
    table = table_text(inp, upper, results, levels, chance)
    print(table)
    print("\nRULE:", rule)
    (out / "scores.txt").write_text(table + "\n\nRULE: " + rule + "\n")
    rows = [(f"grey beauty.png (upper ref)  recall@3={upper['recall3']:.3f}  mean_d={upper['mean_dist_px']:.2f}px",
             inp.beauty, overlay(inp.beauty, inp.bnd))]
    for lv in levels:
        s, fs = results[lv]["scores"], results[lv]["flipped_scores"]
        rows.append((f"{lv} [{'PASS' if results[lv]['pass'] else 'fail'}{' CHOSEN' if lv == chosen else ''}]  "
                     f"recall@3={s['recall3']:.3f}  mean_d={s['mean_dist_px']:.2f}px  "
                     f"(flipped {fs['recall3']:.3f} / {fs['mean_dist_px']:.2f})", images[lv], overlay(images[lv], inp.bnd)))
    contact_sheet(rows, out / "sweep_contact.png", inp.W, inp.H)
    vis_info = visibility(inp.objects, inp.visible, inp.idm, inp.depth16, inp.cam)
    choice = {
        "case": common.case_dir(case).name, "iter": k,
        "metric": {"edges": f"Canny(L2) on luminance, Gaussian sigma {EDGE_SIGMA}px, thresholds {CANNY_LO}/{CANNY_HI}",
                   "id_boundary": "pixels whose id differs from a 4-neighbour (2px image border excluded)",
                   "recall_radius_px": RADIUS, "function": "b2r.target.edge_scores(id_map, image_rgb)"},
        "calibration": {"upper_beauty": upper, "chance_flipped_mean": chance, **bounds},
        "rule": rule, "levels": results, "chosen_level": chosen, "chosen_by_rule": chosen,
        **extra_fields,
        "object_visibility": {inp.names.get(o, str(o)): r for o, r in vis_info.items()},
        "visibility_thresholds": {"small_image_frac": SMALL_FRAC, "almost_hidden_occluded_frac": HIDDEN_OCC,
                                  "partly_hidden_occluded_frac": PARTLY_OCC,
                                  "occluded_frac_definition": "px of the convex hull of the projected 3D bbox covered "
                                  "by nearer non-structure objects (depth.png) / hull px; visible_frac = visible id px / hull px"},
        "known_blind_spot": KNOWN_BLIND_SPOT,
    }
    if chosen is None:
        choice["why"] = "no row passed the rule; no target chosen"
    else:
        weaker = [l for l in levels if l != IMAGE_LEVEL and ALL_LEVELS.index(l) < ALL_LEVELS.index(chosen)] \
            if chosen != IMAGE_LEVEL else []
        choice["why"] = (f"{chosen} is the weakest row run that passes the rule"
                         + ("; weaker levels failed: " + "; ".join(f"{l}: {', '.join(results[l]['fail_reasons'])}"
                                                                  for l in weaker) if weaker else ""))
    common.write_json(out / "choice.json", choice)
    return choice, chosen


def publish(out: Path, choice: dict, level: str) -> None:
    shutil.copyfile(out / "sweep" / f"{level}.png", out / "target.png")
    prompt = choice["levels"][level].get("prompt")
    (out / "prompt.txt").write_text((prompt or choice["levels"][level].get("prompt_note", "")) + "\n")


def cmd_set_level(case, k: int, level: str, reason: str) -> int:
    out = common.stage_dir(case, "target", k, create=False)
    cp = out / "choice.json"
    if not cp.exists():
        return fail(f"--set-level: {cp} does not exist; run the target stage for iter {k} first")
    choice = common.read_json(cp)
    level = level.strip().upper()
    if level not in choice.get("levels", {}):
        return fail(f"--set-level: {level} was not run in iter {k}; rows available: {sorted(choice.get('levels', {}))}")
    if not (out / "sweep" / f"{level}.png").exists():
        return fail(f"--set-level: {out / 'sweep' / (level + '.png')} is missing")
    if not reason.strip():
        return fail("--set-level needs --reason TEXT (what was seen that the rule could not see)")
    prev = choice.get("chosen_level")
    if "chosen_by_rule" not in choice:
        choice["chosen_by_rule"] = prev
    row = choice["levels"][level]
    ov = {"from": prev, "to": level, "reason": reason.strip(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "by": "agent or person via --set-level", "to_passes_rule": bool(row.get("pass"))}
    if not row.get("pass"):
        ov["note"] = "the chosen row FAILS the numeric rule: " + "; ".join(row.get("fail_reasons", []))
    choice.setdefault("overrides", []).append(ov)
    choice["override"] = ov
    choice["chosen_level"] = level
    choice["why"] = (f"{choice['chosen_by_rule']} chosen by the numeric rule; overridden to {level} "
                     f"(see override): {reason.strip()}")
    common.write_json(cp, choice)
    publish(out, choice, level)
    if level != IMAGE_LEVEL:
        cfg = common.load_case(case)
        cfg.setdefault("settings", {})["target_level"] = level
        common.save_case(case, cfg)
    print(f"override recorded: {prev} -> {level}; target.png, prompt.txt"
          f"{', case.json settings.target_level' if level != IMAGE_LEVEL else ''} updated")
    return 0


def cmd_image(case, k: int, inp: Inputs, image: Path, size: tuple[int, int]) -> int:
    W, H = size
    if inp.size() != (W, H):
        return fail(f"g-buffers are {inp.size()} but case size is {(W, H)}: re-run gbuffers for iter {k}")
    if not image.exists():
        return fail(f"--image {image} does not exist")
    im = Image.open(image).convert("RGB")
    w, h = im.size
    note = None
    if (w, h) != (W, H):
        dev = abs((w / h) / (W / H) - 1)
        if dev > ASPECT_TOL:
            return fail(f"--image is {w}x{h} (aspect {w / h:.4f}); the case is {W}x{H} (aspect {W / H:.4f}). "
                        f"Aspect ratios differ by {dev * 100:.2f}% > {ASPECT_TOL * 100:.0f}%; refusing to crop or pad. "
                        f"Supply an image with the case aspect ratio.")
        im = im.resize((W, H), Image.LANCZOS)
        note = f"resized {w}x{h} -> {W}x{H} (pure resize, aspect differs by {dev * 100:.3f}%)"
        print("NOTE:", note)
    out = common.stage_dir(case, "target", k)
    (out / "sweep").mkdir(exist_ok=True)
    im.save(out / "sweep" / f"{IMAGE_LEVEL}.png")
    results = {IMAGE_LEVEL: {"source": "bring-your-own image (--image)", "file": image.name,
                             "sha256": sha256(image), "original_size": [w, h], "resize_note": note,
                             "prompt": None, "prompt_note": f"(no generation: target supplied with --image {image.name}, "
                                                            f"sha256 {sha256(image)[:16]})"}}
    for f in ("target.png", "prompt.txt"):
        (out / f).unlink(missing_ok=True)
    choice, chosen = write_outputs(case, k, out, inp, {IMAGE_LEVEL: load_rgb(out / "sweep" / f"{IMAGE_LEVEL}.png")},
                                   results, [IMAGE_LEVEL],
                                   {"ladder_run": False, "endpoint": None,
                                    "ladder_note": "no conditioning ladder was run: the target was supplied with --image",
                                    "reference": None, "extra_prompt": None})
    if chosen is None:
        print(f"\nFAIL: the supplied image does not pass the edge-alignment rule (see {out / 'scores.txt'}). "
              f"If it is right anyway, accept it with --set-level {IMAGE_LEVEL} --reason TEXT.", file=sys.stderr)
        return EXIT_NO_PASS
    publish(out, choice, IMAGE_LEVEL)
    print(f"\nTARGET (own image): {out / 'target.png'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="b2r.target", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--levels", default=None, help="all | L3 | L2,L3 (default: all at iter 0, settings.target_level later)")
    ap.add_argument("--reuse-sweep", action="store_true", help="keep sweep images already generated at the case size instead of paying for them again")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--image", type=Path, default=None, help="bring-your-own target PNG (no ladder)")
    ap.add_argument("--no-reference", action="store_true", help="iter > 0: do not pass the previous target as look reference")
    ap.add_argument("--extra-prompt", default="")
    ap.add_argument("--set-level", default=None, help="override the rule's pick for this iteration (with --reason)")
    ap.add_argument("--reason", default="")
    ap.add_argument("--rescore", action="store_true", help="debug: re-score existing sweep/L*.png, no fal calls")
    ap.add_argument("--out-dir", type=Path, default=None, help="with --rescore: write here (default target/iterK/rescore)")
    a = ap.parse_args(sys.argv[1:] if argv is None else list(argv))

    case, k = a.case, a.iter
    cfg = common.load_case(case)
    if a.set_level:
        return cmd_set_level(case, k, a.set_level, a.reason)

    size = cfg.get("size") or {}
    W, H = int(size["width"]), int(size["height"])
    aspect = size.get("aspect_ratio") or common.nearest_aspect(W, H)
    resolution = size.get("resolution") or common.DEFAULT_RESOLUTION
    inp = Inputs(case, k)

    if a.image is not None:
        if a.levels or a.rescore:
            return fail("--image runs no ladder; do not combine it with --levels or --rescore")
        return cmd_image(case, k, inp, a.image.expanduser().resolve(), (W, H))

    look = str(cfg.get("look") or "").strip()
    if not look:
        return fail("case.json \"look\" is empty. Write it first: one or two sentences saying what the photo should "
                    "look like (style, materials, time of day, camera feel), e.g. with `b2r.py new CASE --look TEXT` "
                    "or by editing case.json. The target prompt takes all of its style from \"look\".")

    # levels
    if a.levels is None:
        if k == 0:
            levels = list(ALL_LEVELS)
        else:
            lv = (cfg.get("settings") or {}).get("target_level")
            if not lv:
                return fail(f"iter {k}: case.json settings.target_level is not set (iteration 0 sets it); "
                            f"pass --levels explicitly")
            levels = [str(lv).upper()]
    elif a.levels.strip().lower() == "all":
        levels = list(ALL_LEVELS)
    else:
        levels = [s.strip().upper() for s in a.levels.split(",") if s.strip()]
    bad = [l for l in levels if l not in ALL_LEVELS]
    if bad:
        return fail(f"unknown levels {bad}; expected some of {ALL_LEVELS}")
    levels = [l for l in ALL_LEVELS if l in levels]  # weakest first

    ref = None
    if k > 0 and not a.no_reference:
        ref = common.stage_dir(case, "target", k - 1, create=False) / "target.png"
        if not ref.exists():
            return fail(f"look reference {ref} missing (iteration {k - 1} has no target); use --no-reference")

    vis_info = visibility(inp.objects, inp.visible, inp.idm, inp.depth16, inp.cam)
    objs = object_lines(inp.objects, inp.visible, vis_info, inp.W, inp.H)

    if a.rescore:
        out = (a.out_dir.expanduser().resolve() if a.out_dir else common.stage_dir(case, "target", k) / "rescore")
        src = common.stage_dir(case, "target", k, create=False) / "sweep"
        levels = [l for l in levels if (src / f"{l}.png").exists()]
        if not levels:
            return fail(f"--rescore: no sweep images in {src}")
        (out / "sweep").mkdir(parents=True, exist_ok=True)
        images, results = {}, {}
        for lv in levels:
            rgb = load_rgb(src / f"{lv}.png")
            if rgb.shape[:2] != (inp.H, inp.W):
                return fail(f"--rescore: {src / (lv + '.png')} is {rgb.shape[1]}x{rgb.shape[0]}, g-buffers {inp.W}x{inp.H}")
            images[lv] = rgb
            results[lv] = {"source": str(src / f"{lv}.png"), "rescored": True}
        write_outputs(case, k, out, inp, images, results, levels,
                      {"ladder_run": True, "rescore_of": str(src), "endpoint": None})
        print(f"\nrescored {levels} -> {out} (nothing in the case was changed)")
        return 0

    # ---------------------------------------------------------- size knowledge before spending anything
    from . import falio

    verified = bool(size.get("size_verified"))
    if not verified:
        hit = known_size(aspect, resolution)
        if hit:
            (kw, kh), src = hit
            print(f"known generator size for {aspect} at {resolution}: {kw}x{kh} (from {src})")
            cfg = set_case_size(case, kw, kh, f"known size table {src}")
            W, H, verified = kw, kh, True
            if inp.size() != (W, H):
                print(f"case size set to {W}x{H}; g-buffers are {inp.W}x{inp.H}: re-render g-buffers (exit 3, no fal call)")
                return EXIT_SIZE_CHANGED
    if verified and inp.size() != (W, H):
        return fail(f"g-buffers are {inp.W}x{inp.H} but the verified case size is {W}x{H}: re-run gbuffers for iter {k}")
    if not verified and inp.size() != (W, H):
        print(f"WARNING: g-buffers {inp.W}x{inp.H} differ from the provisional case size {W}x{H}; this call only "
              f"establishes the generator's size", flush=True)

    out = common.stage_dir(case, "target", k)
    sweep = out / "sweep"
    sweep.mkdir(exist_ok=True)
    idvis = inp.g / "id_vis.png"
    if not idvis.exists():
        idvis = out / "id_colours.png"
        Image.fromarray(id_colours(inp.idm)).save(idvis)
    paths = {"beauty": inp.g / "beauty.png", "depth": inp.g / "depth.png", "normal": inp.g / "normal_cam.png",
             "idvis": idvis}
    for f in ("target.png", "prompt.txt"):
        (out / f).unlink(missing_ok=True)

    results, images, not_generated = {}, {}, {}
    for lv in levels:
        prompt, imgs, extra = build_level(lv, paths, objs, look, ref)
        if a.extra_prompt:
            prompt += "\n\n" + a.extra_prompt
        dest = sweep / f"{lv}.png"
        meta = {"prompt": prompt, "inputs": [{"role": r, "path": rel(case, p)} for r, p in imgs],
                "endpoint_extra": extra, "seed": a.seed, "aspect_ratio": aspect, "resolution": resolution}
        if a.reuse_sweep and dest.exists() and Image.open(dest).size == (W, H):
            print(f"[{lv}] reusing {dest} (--reuse-sweep)", flush=True)
            images[lv] = load_rgb(dest)
            results[lv] = {**meta, "reused_from_earlier_run": True}
            continue
        args = {"prompt": prompt, "image_urls": [falio.upload(p) for _, p in imgs], "aspect_ratio": aspect,
                "resolution": resolution, "num_images": 1, "seed": a.seed, "output_format": "png", **extra}
        print(f"[{lv}] calling {falio.NANO_BANANA_2_EDIT} with {[r for r, _ in imgs]} ({aspect}, {resolution})", flush=True)
        try:
            res = falio.call(case, falio.NANO_BANANA_2_EDIT, args, note=f"target iter{k} {lv}", units=1)
        except falio.BudgetError:
            raise
        except RuntimeError as e:
            # The generator sometimes returns no image for one level (it answers with text, or declines).
            # That level is recorded as not generated and the ladder goes on; nothing stands in for it.
            not_generated[lv] = str(e)[:400]
            print(f"[{lv}] NOT GENERATED after retries; recorded and skipped: {str(e)[:160]}", flush=True)
            continue
        falio.download(res["images"][0]["url"], dest)
        got = Image.open(dest).size
        if got != (W, H):
            evidence = sweep / f"{lv}_returned_{got[0]}x{got[1]}.png"
            dest.replace(evidence)
            probe = {"aspect_ratio": aspect, "resolution": resolution, "requested_case_size": [W, H],
                     "returned_size": list(got), "evidence": rel(case, evidence), "level": lv,
                     "endpoint": falio.NANO_BANANA_2_EDIT, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if verified:
                probe["result"] = "error: size was already verified"
                common.write_json(out / "size_probe.json", probe)
                return fail(f"{falio.NANO_BANANA_2_EDIT} returned {got[0]}x{got[1]} for {aspect} at {resolution}, "
                            f"but the case size {W}x{H} was verified earlier. The generator changed behaviour; "
                            f"evidence kept at {evidence}. Not cropping or padding. Set size.size_verified to false "
                            f"in case.json to relearn, then re-run gbuffers.")
            set_case_size(case, got[0], got[1], f"{falio.NANO_BANANA_2_EDIT} returned it for {aspect} at {resolution}")
            remember_size(aspect, resolution, got[0], got[1], falio.NANO_BANANA_2_EDIT)
            probe["result"] = "learned: case.json size updated, size_verified true; g-buffers must be re-rendered"
            common.write_json(out / "size_probe.json", probe)
            print(f"[{lv}] generator returned {got[0]}x{got[1]} for {aspect} at {resolution} (case had {W}x{H}): "
                  f"case.json size updated; re-render g-buffers and run target again (exit 3)")
            return EXIT_SIZE_CHANGED
        if not verified:
            set_case_size(case, W, H, f"{falio.NANO_BANANA_2_EDIT} returned it for {aspect} at {resolution}")
            remember_size(aspect, resolution, W, H, falio.NANO_BANANA_2_EDIT)
            verified = True
            if inp.size() != (W, H):
                print(f"case size {W}x{H} verified; g-buffers are {inp.W}x{inp.H}: re-render g-buffers (exit 3)")
                return EXIT_SIZE_CHANGED
        images[lv] = load_rgb(dest)
        results[lv] = meta

    levels = [lv for lv in levels if lv in results]
    if not levels:
        return fail("no level produced an image: " + "; ".join(f"{l}: {m[:120]}" for l, m in not_generated.items()))
    choice, chosen = write_outputs(case, k, out, inp, images, results, levels,
                                   {"ladder_run": True, "endpoint": falio.NANO_BANANA_2_EDIT,
                                    "levels_not_generated": not_generated,
                                    "reference": rel(case, ref) if ref else None, "extra_prompt": a.extra_prompt,
                                    "look": look, "size": {"width": W, "height": H, "aspect_ratio": aspect,
                                                           "resolution": resolution}})
    if chosen is None:
        print("\nFAIL: no level passed the edge-alignment rule. Inspect sweep_contact.png; if one level is right "
              "anyway, accept it with --set-level LN --reason TEXT.", file=sys.stderr)
        return EXIT_NO_PASS
    publish(out, choice, chosen)
    if k == 0:
        cfg = common.load_case(case)
        cfg.setdefault("settings", {})["target_level"] = chosen
        common.save_case(case, cfg)
    print(f"\nCHOSEN: {chosen} -> {out / 'target.png'}")
    print("Look at sweep_contact.png before going on: " + KNOWN_BLIND_SPOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
