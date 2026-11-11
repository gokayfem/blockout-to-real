"""Decompose stage: the target photo -> registered intrinsic maps + a lighting estimate.

    python -m b2r.decompose --case CASE --iter K [--target PNG] [--refetch] [--out-dir DIR]
                            [--roughness-source auto|patina|marigold-iid|constant:VALUE]

Estimators on fal (see falio): Marigold V2 (albedo, normals), PATINA (basecolor, normal, roughness,
metalness, height), MoGe-3 (metric point map with the case's field of view). Raw outputs are cached under
decompose/iterK/raw/<target sha256[:12]>/ and reused unless --refetch. Every candidate is checked against
the g-buffers and the photo; the choices and the numbers behind them go to decompose.json.

Roughness: "auto" uses PATINA's roughness unless it fails the plausibility check (it follows the
lighting on whole-scene photos); then local Marigold-IID Appearance v1.1 (diffusers) is used. If
diffusers is missing the stage stops and names the two ways forward. "constant:V" uses V everywhere and
records it as a stated assumption.

--out-dir DIR writes everything to DIR and only READS the fal cache of decompose/iterK (never fetches).

Outputs (decompose/iterK/): albedo.png, roughness.png, metallic.png, normal_cam.png, height.png,
height_detail.png, depth_moge.exr, depth_delta.exr/.png, shading.exr, shading_flags.png, lighting.json,
geometry_delta.json, decompose.json, decompose_contact.png.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from . import common

common.setup_model_env()
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from . import decompose_fetch as fetch  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

# structural-surface fallback (no object has category "structure")
FLAT_DEG = 10.0  # a surface is "flat" if FLAT_SHARE of its pixels have a normal within FLAT_DEG of its mean normal
FLAT_SHARE = 0.9
FLAT_MAX_SURFACES = 4
FLAT_MIN_IMAGE_FRAC = 0.01

# albedo per-object dark replacement (linear luminance)
NEAR_BLACK, TARGET_DARK = 0.03, 0.06  # ~sRGB 48/255 and ~69/255

# roughness / metallic plausibility
LEAK_CORR = 0.5  # |corr(roughness, log shading)| above this inside a surface = lighting leaked into roughness
LEAK_MIN_PX = 5000
METAL_AMBIG = (0.15, 0.85)  # an object whose mean metallic lies here is "half metal": not a PBR material
METAL_AMBIG_SHARE = 0.25  # reject the map if this share of the visible surface area is ambiguous


# ----------------------------------------------------------------------------- io
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


class Ctx:
    W = H = 0
    size_log: dict = {}


def read_img(path, name, flags=cv2.IMREAD_UNCHANGED):
    W, H = Ctx.W, Ctx.H
    a = cv2.imread(str(path), flags)
    if a is None:
        raise RuntimeError(f"could not read {path}")
    h, w = a.shape[:2]
    entry = {"file": Path(path).name, "size": [w, h], "resized": False}
    if (w, h) != (W, H):
        if abs((w / h) / (W / H) - 1) > 0.005:
            raise RuntimeError(f"{name}: {w}x{h} has a different aspect ratio than {W}x{H}; refusing to resample")
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA if w > W else cv2.INTER_CUBIC)
        entry["resized"] = True
    Ctx.size_log[name] = entry
    if a.ndim == 3 and a.shape[2] >= 3:
        a = a[..., 2::-1]  # BGR(A) -> RGB, alpha dropped
    return a


def to_unit(a):
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    return a.astype(np.float32) / scale


def write_exr(path, a):
    a = a.astype(np.float32)
    if a.ndim == 3:
        a = a[..., ::-1]
    if not cv2.imwrite(str(path), a, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
        raise RuntimeError(f"could not write {path}")


def write_png(path, a, bits=8):
    a = np.clip(a, 0, 1)
    a = (a * (65535 if bits == 16 else 255) + 0.5).astype(np.uint16 if bits == 16 else np.uint8)
    if a.ndim == 3:
        a = a[..., ::-1]
    if not cv2.imwrite(str(path), a):
        raise RuntimeError(f"could not write {path}")


def load_gbuffers(case, k: int):
    gdir = common.stage_dir(case, "gbuffers", k, create=False)

    def exr(n):
        a = cv2.imread(str(gdir / n), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise SystemExit(f"decompose: cannot read {gdir / n} (run gbuffers for iter {k})")
        return a
    depth = exr("depth.exr")
    depth = depth[..., 0] if depth.ndim == 3 else depth
    pos = exr("position.exr")[..., 2::-1]
    nrm = exr("normal.exr")[..., 2::-1]
    ncam = to_unit(exr("normal_cam.png")[..., 2::-1]) * 2 - 1
    ncam /= np.linalg.norm(ncam, axis=-1, keepdims=True) + 1e-8
    ids = exr("id.png")
    ids = (ids[..., 0] if ids.ndim == 3 else ids).astype(np.int32)
    vp = gdir / "valid.png"
    if vp.exists():
        valid = cv2.imread(str(vp), cv2.IMREAD_UNCHANGED)
        valid = (valid if valid.ndim == 2 else valid[..., 0]) > 0
        valid_src = "valid.png"
    else:
        valid = (ids > 0) & (depth > 0)
        valid_src = "id > 0 and depth > 0 (no valid.png)"
    if ids.shape != (Ctx.H, Ctx.W):
        raise SystemExit(f"decompose: g-buffers are {ids.shape[1]}x{ids.shape[0]} but case size is {Ctx.W}x{Ctx.H}; "
                         f"re-run gbuffers")
    cam = common.read_json(gdir / "camera.json")
    vis = common.read_json(gdir / "objects_visible.json")
    names = {int(i): v["name"] for i, v in vis.items()}
    cats = {}
    op = common.stage_dir(case, "blockout", k, create=False) / "objects.json"
    if op.exists():
        for o in common.read_json(op):
            cats[int(o["id"])] = str(o.get("category") or "").strip().lower()
            names.setdefault(int(o["id"]), o["name"])
    for i in np.unique(ids):
        names.setdefault(int(i), f"id{int(i)}")
    return dict(gdir=gdir, depth=depth.astype(np.float32), pos=pos.astype(np.float32), nrm=nrm.astype(np.float32),
                ncam=ncam, ids=ids, valid=valid, valid_src=valid_src, cam=cam, names=names, cats=cats)


# ----------------------------------------------------------------------- normals
def signed_perms():
    mats = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), np.float32)
            for r in range(3):
                m[r, p[r]] = s[r]
            mats.append(m)
    return mats


def ang_err(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def describe_perm(m):
    ax = "xyz"
    out = []
    for r in range(3):
        c = int(np.argmax(np.abs(m[r])))
        out.append(("+" if m[r, c] > 0 else "-") + ax[c])
    return "out(X,Y,Z) = (" + ", ".join(out) + ") of source"


def normal_check(name, n_src, g, mask):
    n_src = n_src / (np.linalg.norm(n_src, axis=-1, keepdims=True) + 1e-8)
    ref = g["ncam"][mask]
    src = n_src[mask]
    sub = slice(None, None, 7)
    best = None
    for m in signed_perms():
        e = np.median(ang_err(src[sub] @ m.T, ref[sub]))
        if best is None or e < best[0]:
            best = (e, m)
    m = best[1]
    e0 = ang_err(src, ref)
    e1 = ang_err(src @ m.T, ref)
    conv = n_src @ m.T
    blur = cv2.GaussianBlur(conv, (0, 0), 3)
    blur /= np.linalg.norm(blur, axis=-1, keepdims=True) + 1e-8
    detail = float(np.mean(ang_err(conv[mask], blur[mask])))
    return conv, {
        "source": name,
        "identity": {"mean_deg": float(e0.mean()), "median_deg": float(np.median(e0))},
        "best_transform": {"matrix": m.astype(int).tolist(), "text": describe_perm(m),
                           "is_identity": bool(np.allclose(m, np.eye(3))),
                           "mean_deg": float(e1.mean()), "median_deg": float(np.median(e1)),
                           "p90_deg": float(np.percentile(e1, 90))},
        "detail_deg": detail,
    }


def pointmap_normals(P, ok):
    """Camera-space (GL) normals from a per-pixel point map via central differences."""
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[:-2] - P[2:]  # row up = +Y in image
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    good = ok.copy()
    good[:, [0, -1]] = False
    good[[0, -1]] = False
    good &= np.roll(ok, 1, 0) & np.roll(ok, -1, 0) & np.roll(ok, 1, 1) & np.roll(ok, -1, 1)
    n[~good] = 0
    return n, good


# --------------------------------------------------------------------------- moge
def moge_pointmap(glb_path, ply_path, cam):
    import plyfile
    import trimesh
    W, H = Ctx.W, Ctx.H
    mesh = trimesh.load(str(glb_path), force="mesh", process=False)
    V = np.asarray(mesh.vertices, np.float64)
    uv = np.asarray(mesh.visual.uv, np.float64)
    col = np.rint(uv[:, 0] * W - 0.5).astype(int)
    row = np.rint((1 - uv[:, 1]) * H - 0.5).astype(int)
    inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    P = np.zeros((H, W, 3), np.float32)
    ok = np.zeros((H, W), bool)
    P[row[inb], col[inb]] = V[inb]
    ok[row[inb], col[inb]] = True
    z = -V[:, 2]
    zs = np.where(np.abs(z) > 1e-9, z, np.nan)
    u = cam["cx"] + cam["fx"] * V[:, 0] / zs
    v = cam["cy"] - cam["fy"] * V[:, 1] / zs
    reproj = np.hypot(u - (col + 0.5), v - (row + 0.5))
    reproj = reproj[np.isfinite(reproj)]
    info = {"vertices": int(len(V)), "pixels_covered": int(ok.sum()), "pixels_total": W * H,
            "uv_vs_intrinsics_reproj_px": {"median": float(np.median(reproj)),
                                           "p99": float(np.percentile(reproj, 99)),
                                           "max": float(reproj.max())} if reproj.size else None,
            "coords": "GLB vertices are camera space, OpenGL axes (+X right, +Y up, -Z forward); "
                      "pixel from GLB UV: col = u*W-0.5, row = (1-v)*H-0.5"}
    Np = None
    try:
        ply = plyfile.PlyData.read(str(ply_path))["vertex"].data
        xyz = np.stack([ply["x"], ply["y"], ply["z"]], -1)
        if len(xyz) == len(V) and np.allclose(xyz, V, atol=1e-5):
            Np = np.zeros((H, W, 3), np.float32)
            Np[row[inb], col[inb]] = np.stack([ply["nx"], ply["ny"], ply["nz"]], -1)[inb]
            info["ply_same_order_as_glb"] = True
        else:
            info["ply_same_order_as_glb"] = False
    except Exception as e:  # noqa: BLE001 - optional extra normal source; the reason is recorded
        info["ply_error"] = repr(e)
    return P, ok, Np, info


# ------------------------------------------------------------------- local Marigold-IID
IID_MODEL = "prs-eth/marigold-iid-appearance-v1-1"


def _iid_cached(sha: str, roots: list) -> tuple[dict, dict, str] | None:
    tag = sha[:12]
    for root in roots:
        for npz, meta in ((root / tag / "marigold_iid_material.npz", root / tag / "marigold_iid.json"),
                          (root / f"marigold_iid_material_{tag}.npz", root / f"marigold_iid_{tag}.json")):
            if npz.exists() and meta.exists():
                m = json.loads(meta.read_text())
                if m.get("target_sha256") not in (None, sha):
                    continue
                d = np.load(npz)
                return {k: d[k] for k in d.files}, m, str(npz)
    return None


def diffusers_available() -> bool:
    try:
        import diffusers  # noqa: F401
        from diffusers import MarigoldIntrinsicsPipeline  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def need_diffusers_message(why: str) -> str:
    return (f"decompose: {why}, and the local Marigold-IID Appearance v1.1 model needs diffusers, which is not "
            f"installed. Two ways forward:\n"
            f"  1. pip install diffusers transformers accelerate   (then re-run; weights go to the b2r cache dir)\n"
            f"  2. --roughness-source constant:0.6   (a uniform roughness, recorded in decompose.json as a stated "
            f"assumption)\nNot substituting anything silently.")


def marigold_iid(target_path: Path, sha: str, read_roots: list, write_root: Path):
    """Marigold-IID Appearance v1.1 (local; MPS / CUDA / CPU). Returns (maps, meta, cache path)."""
    hit = _iid_cached(sha, read_roots + [write_root])
    if hit:
        return hit
    import torch
    from diffusers import MarigoldIntrinsicsPipeline
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = MarigoldIntrinsicsPipeline.from_pretrained(
        IID_MODEL, torch_dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev)
    img = Image.open(target_path).convert("RGB")
    gen = torch.Generator(device="cpu").manual_seed(0)
    out = pipe(img, num_inference_steps=4, ensemble_size=1, processing_resolution=768,
               match_input_resolution=True, generator=gen)
    pred = np.asarray(out.prediction, np.float32)  # (n_targets, H, W, 3) in [0,1]
    names = list(pipe.target_properties["target_names"])
    res = {n: pred[i] for i, n in enumerate(names)}
    meta = {"model": IID_MODEL, "device": dev, "steps": 4, "processing_resolution": 768, "target_names": names,
            "target": target_path.name, "target_sha256": sha,
            "target_properties": json.loads(json.dumps(pipe.target_properties, default=str))}
    d = write_root / sha[:12]
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "marigold_iid_material.npz", **res)
    (d / "marigold_iid.json").write_text(json.dumps(meta, indent=2))
    for n, a in res.items():
        write_png(d / f"marigold_iid_{n}.png", a)
    return res, meta, str(d / "marigold_iid_material.npz")


# --------------------------------------------------------------------------- checks
def surface_masks(g, erode=5):
    k = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
    out = {}
    for i in sorted(int(x) for x in np.unique(g["ids"]) if x > 0):
        m = ((g["ids"] == i) & g["valid"]).astype(np.uint8)
        if m.sum() == 0:
            continue
        e = cv2.erode(m, k).astype(bool) if erode else m.astype(bool)
        out[i] = e if e.sum() >= 200 else m.astype(bool)  # thin objects: keep un-eroded mask
    return out


def structural_surfaces(g, smasks) -> tuple[list, str]:
    """Objects with category "structure"; otherwise the few largest objects whose g-buffer normals are
    nearly constant. Returns (ids, how they were chosen)."""
    ids = [i for i in smasks if g["cats"].get(i) == "structure"]
    if ids:
        return ids, "objects.json category 'structure'"
    flat = []
    min_px = FLAT_MIN_IMAGE_FRAC * Ctx.W * Ctx.H
    for i, m in smasks.items():
        if m.sum() < min_px:
            continue
        n = g["nrm"][m]
        n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
        mean = n.mean(0)
        if np.linalg.norm(mean) < 1e-6:
            continue
        mean /= np.linalg.norm(mean)
        share = float((ang_err(n, mean[None]) <= FLAT_DEG).mean())
        if share >= FLAT_SHARE:
            flat.append((int(m.sum()), i))
    flat.sort(reverse=True)
    ids = [i for _, i in flat[:FLAT_MAX_SURFACES]]
    how = (f"no object has category 'structure': the up to {FLAT_MAX_SURFACES} largest objects (>= "
           f"{FLAT_MIN_IMAGE_FRAC * 100:.0f}% of the image) with >= {FLAT_SHARE * 100:.0f}% of their g-buffer "
           f"normals within {FLAT_DEG:.0f} deg of their mean normal")
    return ids, how + ("" if ids else "; none qualified")


def albedo_check(name, alb_lin, tgt_lin, g, smasks, struct):
    Yt = tgt_lin @ LUMA
    Ya = alb_lin @ LUMA
    per = {}
    for i in struct:
        m = smasks.get(i)
        if m is None or m.sum() < 2000:
            continue
        yt, ya = Yt[m], Ya[m]
        med = np.median(yt)
        sh = yt < 0.6 * med
        lit = yt > med
        rec = {"pixels": int(m.sum()),
               "cv_target": float(yt.std() / max(yt.mean(), 1e-6)), "cv_albedo": float(ya.std() / max(ya.mean(), 1e-6)),
               "shadow_pixels": int(sh.sum())}
        if sh.sum() >= 500 and lit.any():
            rt = yt[sh].mean() / yt[lit].mean()
            ra = ya[sh].mean() / max(ya[lit].mean(), 1e-6)
            rec["shadow_contrast_target"] = float(rt)
            rec["shadow_contrast_albedo"] = float(ra)
            rec["shadow_survival"] = float(np.log(max(ra, 1e-6)) / np.log(rt))  # 1 = shadow kept, 0 = removed
        bright = yt > 1.5 * med
        if bright.sum() >= 500 and (~bright).any():
            rec["sunpatch_pixels"] = int(bright.sum())
            rec["sunpatch_survival"] = float(np.log(max(ya[bright].mean(), 1e-6) / max(ya[~bright].mean(), 1e-6))
                                             / np.log(yt[bright].mean() / max(yt[~bright].mean(), 1e-6)))
        per[g["names"][i]] = rec
    if not per:
        return {"candidate": name, "surfaces": {}, "mean_cv_ratio_albedo_over_target": None,
                "mean_shadow_sun_survival": None, "skipped": "no structural surface with >= 2000 eroded pixels"}
    ratio = float(np.mean([r["cv_albedo"] / max(r["cv_target"], 1e-6) for r in per.values()]))
    surv = [r["shadow_survival"] for r in per.values() if "shadow_survival" in r]
    surv += [r["sunpatch_survival"] for r in per.values() if "sunpatch_survival" in r]
    return {"candidate": name, "surfaces": per, "mean_cv_ratio_albedo_over_target": ratio,
            "mean_shadow_sun_survival": float(np.mean(surv)) if surv else None}


def per_object_stats(a, g, smasks):
    out = {}
    for i, m in smasks.items():
        v = a[m]
        out[g["names"][i]] = {"mean": float(v.mean()), "std": float(v.std()), "p10": float(np.percentile(v, 10)),
                              "p90": float(np.percentile(v, 90)), "pixels": int(m.sum())}
    return out


def _corr(a, b):
    if a.size < 3 or a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def roughness_check(name, rough, metal, g, smasks, shadeY, struct):
    """Generic plausibility: (1) inside structural surfaces, roughness must not track the shading;
    (2) the same inside every large object (pixel-weighted); (3) the map must not be constant.
    Metallic: per-object means must be near 0 or near 1 for most of the surface (no 'half metal')."""
    pr = per_object_stats(rough, g, smasks)
    logs = np.log(np.maximum(shadeY, 1e-4))
    corr_s = {}
    for i in struct:
        m = smasks.get(i)
        if m is not None and m.sum() > LEAK_MIN_PX:
            c = _corr(rough[m], logs[m])
            if c is not None:
                corr_s[g["names"][i]] = c
    corr_o, wts = {}, {}
    for i, m in smasks.items():
        if m.sum() > LEAK_MIN_PX:
            c = _corr(rough[m], logs[m])
            if c is not None:
                corr_o[g["names"][i]] = c
                wts[g["names"][i]] = int(m.sum())
    problems, skipped = [], []
    if corr_s:
        ms = float(np.mean(np.abs(list(corr_s.values()))))
        if ms > LEAK_CORR:
            problems.append(f"roughness tracks shading inside structural surfaces (mean |r| = {ms:.2f} > {LEAK_CORR}): "
                            f"lighting leaked into roughness")
    else:
        skipped.append(f"structural-surface leakage check: no structural surface with > {LEAK_MIN_PX} px")
    if corr_o:
        mo = float(sum(abs(corr_o[n]) * wts[n] for n in corr_o) / sum(wts.values()))
        if mo > LEAK_CORR:
            problems.append(f"roughness tracks shading inside objects (pixel-weighted mean |r| = {mo:.2f} > {LEAK_CORR})")
    else:
        skipped.append(f"per-object leakage check: no object with > {LEAK_MIN_PX} px")
    if float(rough.max() - rough.min()) < 1e-6:
        problems.append("roughness map is constant: carries no information")
    mstat, mproblems, pm = None, [], None
    if metal is not None:
        pm = per_object_stats(metal, g, smasks)
        v = g["valid"] & (g["ids"] > 0)
        mstat = {"mean_all": float(metal[v].mean()) if v.any() else None,
                 "fraction_gt_0.5": float((metal[v] > 0.5).mean()) if v.any() else None,
                 "constant": bool(metal.max() - metal.min() < 1e-6)}
        tot = sum(s["pixels"] for s in pm.values()) or 1
        amb = {n: round(s["mean"], 3) for n, s in pm.items() if METAL_AMBIG[0] < s["mean"] < METAL_AMBIG[1]}
        share = sum(pm[n]["pixels"] for n in amb) / tot
        mstat["ambiguous_share_of_surface"] = round(float(share), 4)
        if share > METAL_AMBIG_SHARE:
            mproblems.append(f"{share * 100:.0f}% of the visible surface belongs to objects whose mean metallic is "
                             f"between {METAL_AMBIG[0]} and {METAL_AMBIG[1]} ('half metal'): {amb}")
        if mstat["constant"]:
            mstat["note"] = (f"constant map ({float(metal.flat[0]):.3f}): plausible only if every visible material is "
                             f"{'a dielectric' if metal.max() < 0.5 else 'a metal'}; carries no per-pixel information")
    return {"source": name, "per_object_roughness": pr, "per_object_metallic": pm,
            "roughness_vs_log_shading_corr_structural": corr_s, "roughness_vs_log_shading_corr_per_object": corr_o,
            "metallic_summary": mstat, "roughness_problems": problems, "roughness_usable": not problems,
            "checks_skipped": skipped,
            "metallic_problems": mproblems, "metallic_usable": metal is not None and not mproblems}


def fit_dominant_light(shY, nw, mask):
    idx = np.flatnonzero(mask.ravel())
    if len(idx) < 500:
        return {"skipped": f"only {len(idx)} usable surface pixels (< 500)"}
    rng = np.random.default_rng(0)
    idx = rng.choice(idx, min(60000, len(idx)), replace=False)
    y = shY.ravel()[idx].astype(np.float64)
    N = nw.reshape(-1, 3)[idx].astype(np.float64)

    def solve(l):
        f = np.maximum(0, N @ l)
        A = np.stack([np.ones_like(f), f], 1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        if c[1] < 0:  # constrain k >= 0
            c = np.array([y.mean(), 0.0])
        r = y - A @ c
        return float(r @ r), c

    k = 4000
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    th = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    errs = [solve(d)[0] for d in dirs]
    l = dirs[int(np.argmin(errs))]
    try:
        from scipy.optimize import minimize

        def f(ang):
            a, b = ang
            return solve(np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]))[0]
        a0 = [np.arccos(l[2]), np.arctan2(l[1], l[0])]
        r = minimize(f, a0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10})
        a, b = r.x
        l = np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)])
    except ImportError:
        pass
    sse, c = solve(l)
    sst = float(((y - y.mean()) ** 2).sum())
    return {"direction_world_towards_light": l.tolist(), "ambient": float(c[0]), "k_directional": float(c[1]),
            "ambient_over_directional": float(c[0] / c[1]) if c[1] > 0 else None,
            "r2": (1 - sse / sst) if sst > 0 else None, "samples": int(len(idx)),
            "elevation_deg": float(np.degrees(np.arcsin(np.clip(l[2], -1, 1)))),
            "azimuth_deg_from_+X_ccw": float(np.degrees(np.arctan2(l[1], l[0])))}


def components(mask, min_area):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(lab == j, stats[j], cent[j]) for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] >= min_area]


def colorize_delta(d, ok, lim=0.25):
    t = np.clip(d / lim, -1, 1)
    out = np.ones(d.shape + (3,), np.float32)
    pos, neg = np.clip(t, 0, 1), np.clip(-t, 0, 1)  # red: photo farther; blue: photo closer (blockout lacks geometry)
    out[..., 0] = 1 - neg
    out[..., 1] = 1 - pos - neg
    out[..., 2] = 1 - pos
    out = np.clip(out, 0, 1)
    out[~ok] = 0.35
    return out


def label_tile(img, text):
    w, h = Ctx.W // 2, Ctx.H // 2
    t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    t = (np.clip(t, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    cv2.rectangle(t, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(t, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def parse_roughness_source(s: str):
    s = s.strip().lower().replace("_", "-")
    if s in ("auto", "patina", "marigold-iid"):
        return s, None
    if s.startswith("constant:"):
        try:
            v = float(s.split(":", 1)[1])
        except ValueError:
            v = -1
        if 0.0 <= v <= 1.0:
            return "constant", v
    raise SystemExit(f"--roughness-source must be auto, patina, marigold-iid or constant:VALUE with 0 <= VALUE <= 1; "
                     f"got {s!r}")


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="b2r.decompose", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--target", type=Path, default=None, help="default: target/iterK/target.png")
    ap.add_argument("--refetch", action="store_true", help="re-run the fal calls even if raw/ holds this target")
    ap.add_argument("--roughness-source", default="auto", help="auto | patina | marigold-iid | constant:VALUE")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write outputs here instead of decompose/iterK (the fal cache of decompose/iterK is then "
                         "only READ, never fetched or written)")
    args = ap.parse_args(sys.argv[1:] if argv is None else list(argv))
    rmode, rconst = parse_roughness_source(args.roughness_source)
    # numpy 2.x on macOS Accelerate raises spurious FP warnings from matmul on finite data; silence only those
    import warnings
    warnings.filterwarnings("ignore", message=r".* encountered in matmul", category=RuntimeWarning)

    case, k = args.case, args.iter
    cfg = common.load_case(case)
    Ctx.W, Ctx.H = int(cfg["size"]["width"]), int(cfg["size"]["height"])
    Ctx.size_log = {}
    W, H = Ctx.W, Ctx.H
    iter_dir = common.stage_dir(case, "decompose", k, create=args.out_dir is None)
    if args.out_dir is not None:
        out = args.out_dir.expanduser().resolve()
        if out == iter_dir.resolve():
            out = iter_dir
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = iter_dir
    side = out != iter_dir
    if side and args.refetch:
        ap.error("--refetch with --out-dir would write decompose/iterK/raw; not allowed")
    fal_raw = iter_dir / "raw"   # fal result cache (read-only when side)
    raw = out / "raw"            # local model caches for this output dir
    target = (args.target.expanduser().resolve() if args.target
              else common.stage_dir(case, "target", k, create=False) / "target.png")
    if not target.exists():
        raise SystemExit(f"decompose: target {target} does not exist (run the target stage for iter {k})")
    g = load_gbuffers(case, k)
    tsha = fetch.sha256(target)
    report = {"case": common.case_dir(case).name, "iter": k, "target": str(target), "target_sha256": tsha,
              "gbuffers": str(g["gdir"]), "valid_mask": g["valid_src"], "out_dir": str(out),
              "fal_raw_dir": str(fal_raw), "skipped": [], "assumptions": []}

    tgt = to_unit(read_img(target, "target")[..., :3])
    if Ctx.size_log["target"]["resized"]:
        raise SystemExit(f"decompose: target is {Ctx.size_log['target']['size']} but the case size is {W}x{H}; "
                         f"the target stage guarantees the case size, so this target is not from it")
    fetched = fetch.fetch_all(case, target, fal_raw, W, H, g["cam"]["fov_x_deg"], refetch=args.refetch,
                              allow_fetch=not side)
    report["fal_results_cached"] = {n: v["cache"] for n, v in fetched.items()}
    F = {n: v["files"] for n, v in fetched.items()}

    tgt_lin = srgb_to_lin(tgt)
    ids, valid = g["ids"], g["valid"]
    surf = valid & (ids > 0)
    if surf.sum() == 0:
        raise SystemExit("decompose: no surface pixels in the g-buffers (id > 0 and valid); nothing to decompose")
    smasks = surface_masks(g)
    smasks_full = surface_masks(g, erode=0)
    struct, struct_how = structural_surfaces(g, smasks)
    report["structural_surfaces"] = {"ids": struct, "names": [g["names"][i] for i in struct], "chosen_by": struct_how}
    if not struct:
        report["skipped"].append("structural surfaces: none (" + struct_how + "); the albedo flatness check, the "
                                 "structural leakage check and the lighting summary per surface are skipped")

    # ---------------------------------------------------------------- 1. normals
    n_sources = {
        "marigold_v2_normals_png": to_unit(read_img(F["marigold_normals"]["image"], "marigold_normals")[..., :3]) * 2 - 1,
        "patina_normal_png": to_unit(read_img(F["patina"]["normal"], "patina_normal")[..., :3]) * 2 - 1,
        "moge3_normal_png": to_unit(read_img(F["moge3"]["normal_map"], "moge3_normal_map")[..., :3]) * 2 - 1,
    }
    P, pok, Nply, moge_info = moge_pointmap(F["moge3"]["model_mesh"], F["moge3"]["point_cloud"], g["cam"])
    n_pm, _ = pointmap_normals(P, pok)
    n_sources["moge3_pointmap_derived"] = n_pm
    if Nply is not None:
        n_sources["moge3_ply_vertex_normals"] = Nply
    nchecks, nconv = {}, {}
    for name, n in n_sources.items():
        n = np.nan_to_num(n.astype(np.float32))
        m = surf & (np.linalg.norm(n, axis=-1) > 0.5)
        if m.sum() < 100:
            report["skipped"].append(f"normal source {name}: only {int(m.sum())} usable pixels")
            continue
        nconv[name], nchecks[name] = normal_check(name, n, g, m)
        nchecks[name]["pixels"] = int(m.sum())
    if not nchecks:
        raise SystemExit("decompose: no normal source covers the surfaces")
    gb_blur = cv2.GaussianBlur(g["ncam"], (0, 0), 3)
    gb_blur /= np.linalg.norm(gb_blur, axis=-1, keepdims=True) + 1e-8
    gb_detail = float(np.mean(ang_err(g["ncam"][surf], gb_blur[surf])))
    detail_min = 0.5 * gb_detail
    cands = [n for n in nchecks if nchecks[n]["detail_deg"] >= detail_min] or list(nchecks)
    best_med = min(nchecks[n]["best_transform"]["median_deg"] for n in cands)
    near = [n for n in cands if nchecks[n]["best_transform"]["median_deg"] <= best_med + 0.5]
    pick_n = min(near, key=lambda n: nchecks[n]["best_transform"]["p90_deg"])
    ncan = nconv[pick_n].copy()
    holes = np.linalg.norm(ncan, axis=-1) < 0.5
    fill_from = []
    for alt in sorted(nchecks, key=lambda n: nchecks[n]["best_transform"]["median_deg"]):
        if alt == pick_n or alt == "patina_normal_png" or not holes.any():
            continue
        cand = nconv[alt]
        ok = holes & (np.linalg.norm(cand, axis=-1) > 0.5)
        if ok.any():
            ncan[ok] = cand[ok]
            fill_from.append((alt, int(ok.sum())))
            holes &= ~ok
    ncan = ncan / (np.linalg.norm(ncan, axis=-1, keepdims=True) + 1e-8)
    write_png(out / "normal_cam.png", ncan * 0.5 + 0.5)
    report["normals"] = {"checks": nchecks, "gbuffer_detail_deg": gb_detail, "detail_threshold_deg": detail_min,
                         "chosen": pick_n, "holes_filled_from": fill_from, "holes_left": int(holes.sum()),
                         "why": "lowest median angular error vs g-buffer normal_cam after the best signed axis "
                                "permutation, among sources not over-smoothed (detail >= 0.5x g-buffer's); ties within "
                                "0.5 deg broken by lowest p90 error (noise must not count as detail)",
                         "near_tie_candidates": near,
                         "output_convention": "camera space, X right, Y up, Z towards camera, encoded n*0.5+0.5"}

    # ---------------------------------------------------------------- 2. albedo
    alb = {"marigold_v2_albedo": to_unit(read_img(F["marigold_albedo"]["image"], "marigold_albedo")[..., :3]),
           "patina_basecolor": to_unit(read_img(F["patina"]["basecolor"], "patina_basecolor")[..., :3])}
    achecks = {n: albedo_check(n, srgb_to_lin(a), tgt_lin, g, smasks, struct) for n, a in alb.items()}
    if all(c["mean_cv_ratio_albedo_over_target"] is not None for c in achecks.values()):
        score = {n: c["mean_cv_ratio_albedo_over_target"] + (c["mean_shadow_sun_survival"] or 0) for n, c in achecks.items()}
        pick_a = min(score, key=score.get)
        a_why = ("score = mean(CV_albedo/CV_target over structural surfaces) + mean shadow/sun-patch survival "
                 "(log-contrast ratio, 1 = lighting kept in albedo, 0 = removed); lower is better")
    else:
        score, pick_a = None, "marigold_v2_albedo"
        a_why = ("flatness check skipped (no structural surface large enough); default marigold_v2_albedo, the "
                 "candidate that de-lit structural surfaces best in the reference experiment")
        report["skipped"].append("albedo flatness check: " + a_why)
    other_a = [n for n in alb if n != pick_a][0]
    albedo = alb[pick_a].copy()
    # Per-object sanity: replace the chosen albedo with the other candidate ONLY when it is near-black but the
    # photo shows the object is not dark. A dark glossy object with bright reflections must count as dark, so the
    # target is summarised by a LOW percentile, not the mean.
    alb_lin0 = srgb_to_lin(albedo)
    Ya0, Yt0, Yo0 = alb_lin0 @ LUMA, tgt_lin @ LUMA, srgb_to_lin(alb[other_a]) @ LUMA
    patched, decisions = {}, {}
    for i, m in smasks_full.items():
        name = g["names"][i]
        ay, ty = float(np.median(Ya0[m])), float(np.percentile(Yt0[m], 25))
        d = {"id": i, "pixels": int(m.sum()), "albedo_median_lin_Y": ay, "target_p25_lin_Y": ty,
             "target_median_lin_Y": float(np.median(Yt0[m]))}
        if m.sum() < 500:
            d.update(replace=False, reason="fewer than 500 visible pixels")
        elif ay >= NEAR_BLACK:
            d.update(replace=False, reason=f"albedo not near-black (median {ay:.3f} >= {NEAR_BLACK})")
        elif ty < TARGET_DARK:
            d.update(replace=False, reason=f"genuinely dark object: target p25 {ty:.3f} < {TARGET_DARK}, keep dark albedo")
        elif ty / max(ay, 1e-4) <= 4:
            d.update(replace=False, reason=f"target/albedo ratio {ty / max(ay, 1e-4):.1f} <= 4")
        else:
            d.update(replace=True, replaced_with=other_a,
                     reason=f"near-black albedo ({ay:.3f}) but object is light in the photo (p25 {ty:.3f})")
        if d["replace"]:
            albedo[m] = alb[other_a][m]
            d["albedo_median_lin_Y_after"] = float(np.median(Yo0[m]))
            patched[name] = d
        else:
            d["albedo_median_lin_Y_after"] = ay
        decisions[name] = d
    write_png(out / "albedo.png", albedo)
    alb_lin = srgb_to_lin(albedo)
    report["albedo"] = {"checks": achecks, "score_lower_is_better": score, "chosen": pick_a, "why": a_why,
                        "assumed_encoding": "8-bit sRGB for both candidates",
                        "per_object_replacements": patched,
                        "replacement_rule": f"replace iff median(albedo Y) < {NEAR_BLACK} and p25(target Y) >= "
                                            f"{TARGET_DARK} and p25(target Y)/median(albedo Y) > 4 (linear luminance, "
                                            f"per object)",
                        "per_object_replacement_decisions": decisions,
                        "no_surface_note": "id 0 (no geometry) has no surface; albedo there is meaningless"}

    # ---------------------------------------------------------------- shading
    sat = (tgt >= 250 / 255).any(-1)
    clampd = (alb_lin @ LUMA) < 0.02
    shading = np.clip(tgt_lin / np.maximum(alb_lin, 0.02), 0, 16)
    nosurf = ~surf
    shading[nosurf] = tgt_lin[nosurf]
    write_exr(out / "shading.exr", shading)
    cv2.imwrite(str(out / "shading_flags.png"), (sat * 1 + clampd * 2 + nosurf * 4).astype(np.uint8))
    shY = shading @ LUMA
    report["shading"] = {"definition": "linear(target)/max(linear(albedo),0.02) per channel, clipped to [0,16]; sRGB "
                                       "EOTF used for linearisation; no-surface pixels hold linear(target)",
                         "flags_png": "shading_flags.png bit0=target saturated (>=250 in any channel), bit1=albedo "
                                      "luminance < 0.02 (clamped), bit2=no surface (id 0 / invalid)",
                         "saturated_pixels": int(sat.sum()),
                         "albedo_clamped_pixels_on_surfaces": int((clampd & surf).sum()),
                         "median_luminance_on_surfaces": float(np.median(shY[surf]))}

    # ---------------------------------------------------------------- 3. roughness / metallic
    rough_p = to_unit(read_img(F["patina"]["roughness"], "patina_roughness"))
    metal_p = to_unit(read_img(F["patina"]["metalness"], "patina_metalness"))
    rough_p = rough_p[..., 0] if rough_p.ndim == 3 else rough_p
    metal_p = metal_p[..., 0] if metal_p.ndim == 3 else metal_p
    rchecks = {"patina": roughness_check("patina", rough_p, metal_p, g, smasks, shY, struct)}
    rchecks["patina_inverted_info_only"] = roughness_check("patina_inverted (1-r), diagnostic only", 1 - rough_p, None,
                                                           g, smasks, shY, struct)
    p_ok, pm_ok = rchecks["patina"]["roughness_usable"], rchecks["patina"]["metallic_usable"]
    if rmode == "auto":
        rsrc = "patina" if p_ok else "marigold-iid"
    else:
        rsrc = rmode
    need_iid = rsrc == "marigold-iid" or (not pm_ok and rsrc != "constant")
    roughness, metallic, msrc = rough_p, metal_p, "patina"
    notes = []
    if rsrc == "patina":
        notes.append("roughness: PATINA passed the plausibility check" if p_ok else
                     "roughness: PATINA forced by --roughness-source despite failing: "
                     + "; ".join(rchecks["patina"]["roughness_problems"]))
    if rsrc == "constant":
        roughness = np.full((H, W), rconst, np.float32)
        a_txt = (f"roughness = {rconst} everywhere (--roughness-source constant:{rconst}); PATINA roughness was "
                 f"{'usable' if p_ok else 'rejected: ' + '; '.join(rchecks['patina']['roughness_problems'])}")
        notes.append(a_txt)
        report["assumptions"].append(a_txt)
        if not pm_ok:
            metallic, msrc = np.zeros((H, W), np.float32), "constant:0"
            a2 = ("metallic = 0 everywhere (all dielectric): PATINA metalness failed ("
                  + "; ".join(rchecks["patina"]["metallic_problems"]) + ") and the constant roughness mode runs no "
                  "local model")
            notes.append(a2)
            report["assumptions"].append(a2)
    if need_iid:
        read_roots = [fal_raw] + ([raw] if side else [])
        write_root = raw if side else fal_raw
        if _iid_cached(tsha, read_roots + [write_root]) is None and not diffusers_available():
            why = ("PATINA roughness failed the plausibility check ("
                   + "; ".join(rchecks["patina"]["roughness_problems"]) + ")") if rsrc == "marigold-iid" and rmode == "auto" \
                else ("--roughness-source marigold-iid was requested" if rsrc == "marigold-iid"
                      else "PATINA metalness failed (" + "; ".join(rchecks["patina"]["metallic_problems"]) + ")")
            print(need_diffusers_message(why), file=sys.stderr)
            return 1
        res, meta, iid_path = marigold_iid(target, tsha, read_roots, write_root)
        mat = res.get("material")
        if mat is None:
            raise RuntimeError(f"Marigold-IID returned targets {list(res)}; no 'material'")
        rough_m, metal_m = mat[..., 0], mat[..., 1]
        if rough_m.shape != (H, W):
            raise RuntimeError(f"Marigold-IID output {rough_m.shape} != {(H, W)}")
        Ctx.size_log["marigold_iid_material"] = {"size": [W, H], "resized": False, "cache": iid_path,
                                                 "note": "processed at 768 px long side, upsampled by the pipeline "
                                                         "(match_input_resolution)"}
        rchecks["marigold_iid"] = roughness_check("marigold_iid_appearance_v1_1", rough_m, metal_m, g, smasks, shY, struct)
        rchecks["marigold_iid"]["meta"] = meta
        if rsrc == "marigold-iid":
            roughness = rough_m
            notes.append(("roughness: PATINA is UNUSABLE (" + "; ".join(rchecks["patina"]["roughness_problems"]) + ")"
                          if not p_ok else "roughness: --roughness-source marigold-iid")
                         + "; used local Marigold-IID Appearance v1.1 roughness")
            if not rchecks["marigold_iid"]["roughness_usable"]:
                notes.append("NOTE the Marigold-IID roughness also fails: "
                             + "; ".join(rchecks["marigold_iid"]["roughness_problems"]))
        if not pm_ok and rchecks["marigold_iid"]["metallic_usable"]:
            metallic, msrc = metal_m, "marigold-iid"
    if msrc == "patina":
        ms = rchecks["patina"]["metallic_summary"] or {}
        notes.append("metallic: PATINA metalness" + (f" ({ms['note']})" if ms.get("note") else "")
                     + ("" if pm_ok else " DESPITE failing: " + "; ".join(rchecks["patina"]["metallic_problems"])
                        + " (Marigold-IID metallicity failed too)"))
        if need_iid and not rchecks["marigold_iid"]["metallic_usable"]:
            notes.append("Marigold-IID metallicity rejected: " + "; ".join(rchecks["marigold_iid"]["metallic_problems"]))
    elif msrc == "marigold-iid":
        notes.append("metallic: Marigold-IID (PATINA metalness failed: "
                     + "; ".join(rchecks["patina"]["metallic_problems"]) + ")")
    write_png(out / "roughness.png", roughness)
    write_png(out / "metallic.png", metallic)
    report["roughness_metallic"] = {"checks": rchecks, "roughness_source": rsrc if rsrc != "constant" else f"constant:{rconst}",
                                    "metallic_source": msrc, "why": notes, "encoding": "8-bit, linear 0..1 (value/255)",
                                    "requested": args.roughness_source}

    # ---------------------------------------------------------------- height
    hraw = cv2.imread(F["patina"]["height"], cv2.IMREAD_UNCHANGED)
    if hraw is None:
        raise RuntimeError(f"could not read {F['patina']['height']}")
    hraw = hraw[..., 0] if hraw.ndim == 3 else hraw
    if hraw.shape != (H, W):
        hraw = read_img(F["patina"]["height"], "patina_height")
        hraw = hraw[..., 0] if hraw.ndim == 3 else hraw
    bits = 16 if hraw.dtype == np.uint16 else 8
    hgt = to_unit(hraw)
    cv2.imwrite(str(out / "height.png"), hraw)  # unchanged PATINA height
    lo = cv2.GaussianBlur(hgt, (0, 0), 8)
    hi = hgt - lo
    hcorr = _corr(lo[surf], np.log(np.maximum(shY[surf], 1e-4)))
    write_png(out / "height_detail.png", np.clip(hi * 4 + 0.5, 0, 1), bits=16)
    report["height"] = {"source": "patina height", "bits": bits, "file": "height.png (PATINA output, unmodified)",
                        "std_total": float(hgt[surf].std()), "std_highpass_sigma8": float(hi[surf].std()),
                        "lowpass_vs_log_shading_corr": hcorr,
                        "height_detail_png": "height_detail.png = 0.5 + 4*(height - gaussian(height, 8px)), 16-bit; "
                                             "use for bump if height.png's low frequencies are lighting/shape"}

    # ---------------------------------------------------------------- 4. depth
    dm = np.where(pok, -P[..., 2], 0).astype(np.float32)
    dg = g["depth"]
    both = surf & pok & (dm > 0) & (dg > 0)
    if both.sum() < 100:
        raise SystemExit(f"decompose: MoGe-3 and the g-buffers overlap on only {int(both.sum())} surface pixels; "
                         f"cannot align depth")
    ratio = dg[both] / dm[both]
    s = float(np.median(ratio))
    for _ in range(5):  # IRLS refine on |s*dm - dg| (Huber-ish)
        r = s * dm[both] - dg[both]
        w = 1 / np.maximum(np.abs(r), 0.05)
        s = float((w * dm[both] * dg[both]).sum() / (w * dm[both] ** 2).sum())
    dal = (s * dm).astype(np.float32)
    rel_before = np.abs(dm[both] - dg[both]) / dg[both]
    rel_after = np.abs(dal[both] - dg[both]) / dg[both]
    delta = np.where(both, dal - dg, 0).astype(np.float32)
    write_exr(out / "depth_moge.exr", np.where(pok, dal, 0))
    write_exr(out / "depth_delta.exr", delta)
    write_png(out / "depth_delta.png", colorize_delta(delta, both))
    gd = {}
    for i, m in smasks_full.items():
        mm = m & both
        if mm.sum() == 0:
            continue
        d = delta[mm]
        gd[str(i)] = {"name": g["names"][i], "visible_pixels": int(mm.sum()),
                      "mean_abs_m": float(np.abs(d).mean()), "median_abs_m": float(np.median(np.abs(d))),
                      "p90_abs_m": float(np.percentile(np.abs(d), 90)), "signed_mean_m": float(d.mean()),
                      "median_rel_err": float(np.median(np.abs(d) / dg[mm]))}
    common.write_json(out / "geometry_delta.json", gd)
    ranked = sorted(gd.values(), key=lambda r: -r["median_abs_m"])
    report["depth"] = {"moge": moge_info, "scale_median_ratio_then_irls": s,
                       "scale_plain_median_ratio": float(np.median(ratio)),
                       "median_rel_err_before_alignment": float(np.median(rel_before)),
                       "median_rel_err_after_alignment": float(np.median(rel_after)),
                       "pixels_compared": int(both.sum()),
                       "depth_moge_exr": "planar depth (-Z of the GL camera-space point), scaled by s; 0 = no MoGe point",
                       "delta_sign": "depth_delta = aligned MoGe - g-buffer; negative (blue in png) = photo surface "
                                     "closer than blockout (blockout lacks geometry), positive (red) = photo farther; "
                                     "png clipped at +-0.25 m, grey = not compared",
                       "most_changed_objects": [(r["name"], round(r["median_abs_m"], 3), round(r["signed_mean_m"], 3))
                                                for r in ranked[:6]]}

    # ---------------------------------------------------------------- lighting
    em = []
    nogeo = ids == 0
    nogeo_frac = float(nogeo.mean())
    ring_k = np.ones((11, 11), np.uint8)
    for comp, st, c in components(nogeo, 200):
        x, y, bw, bh = (int(v) for v in st[:4])
        touches = x == 0 or y == 0 or x + bw == W or y + bh == H
        ring = cv2.dilate(comp.astype(np.uint8), ring_k).astype(bool) & ~comp & valid & (ids > 0)
        rid_vals, cnt = np.unique(ids[ring], return_counts=True)
        em.append({"kind": "open_background (id 0, touches the image border)" if touches else
                   "opening (id 0, enclosed by geometry)",
                   "bbox_xywh": [x, y, bw, bh], "centroid_px": [float(c[0]), float(c[1])], "area_px": int(st[4]),
                   "image_frac": round(float(st[4]) / (W * H), 5),
                   "mean_linear_rgb": tgt_lin[comp].mean(0).tolist(), "saturated_fraction": float(sat[comp].mean()),
                   "on_object_id": 0, "position_world": None,
                   "border_position_world": g["pos"][ring].mean(0).tolist() if ring.any() else None,
                   "border_object": g["names"].get(int(rid_vals[np.argmax(cnt)])) if ring.any() else None})
    hot = (tgt.min(-1) >= 245 / 255) & (ids > 0)
    for comp, st, c in components(hot, 50):
        oid_vals, cnt = np.unique(ids[comp], return_counts=True)
        oid = int(oid_vals[np.argmax(cnt)])
        pv = comp & valid
        em.append({"kind": "near_saturated_surface", "bbox_xywh": [int(x) for x in st[:4]],
                   "centroid_px": [float(c[0]), float(c[1])], "area_px": int(st[4]),
                   "mean_linear_rgb": tgt_lin[comp].mean(0).tolist(), "on_object_id": oid,
                   "on_object": g["names"].get(oid), "position_world": g["pos"][pv].mean(0).tolist() if pv.any() else None})
    surf_light = {}
    for i in struct:
        m = smasks.get(i)
        if m is not None and m.any():
            surf_light[g["names"][i]] = {"mean_linear_rgb_shading": shading[m].mean(0).tolist(),
                                         "median_luminance": float(np.median(shY[m])), "pixels": int(m.sum())}
    fitmask = surf & ~sat & ~clampd
    dom = fit_dominant_light(shY, g["nrm"], fitmask)
    if "skipped" in dom:
        report["skipped"].append("dominant light fit: " + dom["skipped"])
    aY = alb_lin @ LUMA
    sat_a = (alb_lin.max(-1) - alb_lin.min(-1)) / np.maximum(alb_lin.max(-1), 1e-4)
    white = surf & (aY > 0.5) & (sat_a < 0.1) & ~sat
    if white.sum() > 1000:
        rgb = shading[white].mean(0)
        col = {"mean_rgb": rgb.tolist(), "chromaticity_rgb_sum1": (rgb / rgb.sum()).tolist(),
               "normalised_max1": (rgb / rgb.max()).tolist(), "pixels": int(white.sum()),
               "near_white_definition": "albedo luminance > 0.5 and albedo saturation < 0.1"}
    else:
        col = {"skipped": f"only {int(white.sum())} near-white surface pixels (need > 1000)"}
        report["skipped"].append("light colour estimate: " + col["skipped"])
    lighting = {"space": "image-space estimate from shading.exr; world coords = Blender world (Z up)",
                "no_geometry_image_frac": round(nogeo_frac, 5),
                "enclosure": ("open: id 0 (no geometry) touches the image border over a large area"
                              if any(e["on_object_id"] == 0 and e["kind"].startswith("open_background")
                                     and e["image_frac"] > 0.05 for e in em) else "closed or mostly closed"),
                "emitters": em, "structural_surfaces": surf_light, "structural_surfaces_chosen_by": struct_how,
                "dominant_light_fit": {**dom, "model": "shading_luminance ~ ambient + k*max(0, n.l), n = g-buffer world "
                                                       "normal, pixels: surfaces, not saturated, not albedo-clamped"},
                "colour_estimate": col}
    if not surf_light:
        lighting["structural_surfaces_note"] = "skipped: no structural surfaces (" + struct_how + ")"
    common.write_json(out / "lighting.json", lighting)
    report["lighting_headline"] = {"emitters": [(e["kind"], e["area_px"], e.get("on_object") or e.get("border_object"))
                                                for e in em],
                                   "dominant_direction": dom.get("direction_world_towards_light"), "r2": dom.get("r2"),
                                   "ambient_over_directional": dom.get("ambient_over_directional")}

    # ---------------------------------------------------------------- contact sheet
    shv = shading / max(float(np.percentile(shading[surf], 99)), 1e-4)
    tiles = [label_tile(tgt, "target"), label_tile(albedo, f"albedo ({pick_a})"),
             label_tile(np.repeat(roughness[..., None], 3, -1), f"roughness ({report['roughness_metallic']['roughness_source']})"),
             label_tile(np.repeat(metallic[..., None], 3, -1),
                        f"metallic ({msrc}){' = 0 everywhere' if metallic.max() == 0 else ''}"),
             label_tile(ncan * 0.5 + 0.5, f"normal_cam ({pick_n})"),
             label_tile(np.repeat(hgt[..., None], 3, -1), "height (patina)"),
             label_tile(lin_to_srgb(shv), "shading (lin, /p99, sRGB)"),
             label_tile(colorize_delta(delta, both), f"depth delta, s={s:.3f}, blue=photo closer, +-0.25m")]
    cv2.imwrite(str(out / "decompose_contact.png"), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))

    from . import falio
    report["output_sizes"] = Ctx.size_log
    report["fal_cost"] = {"case_total_usd": falio.spent(case), "by_endpoint": falio.summary(case)}
    common.write_json(out / "decompose.json", report)
    print(json.dumps({"normals": pick_n, "albedo": pick_a, "albedo_replaced": list(patched),
                      "roughness": report["roughness_metallic"]["roughness_source"], "metallic": msrc,
                      "depth_scale": s, "structural": report["structural_surfaces"]["names"],
                      "skipped": report["skipped"], "assumptions": report["assumptions"],
                      "lighting": report["lighting_headline"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
