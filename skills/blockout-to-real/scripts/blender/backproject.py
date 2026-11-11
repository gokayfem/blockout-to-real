"""backproject: project the decomposed per-pixel maps into each object's own UV textures,
fill what the camera cannot see, build ordinary Principled BSDF materials, and verify the
projector with an unlit round-trip render.

    blender -b --factory-startup --python scripts/blender/backproject.py -- \
        --case CASE --iter K [--maps-dir PATH] [--blend PATH] [--gbuffers PATH] \
        [--normal-strength 1.0] [--bump-strength 0.02] [--bump-distance 0.005] [--samples 16] \
        [--novel-orbit-deg 30 180] [--novel-dolly 1.0 0.6]

Reads   blockout/iterK/{blockout.blend, objects.json}, gbuffers/iterK/{camera.json, position.exr,
        normal.exr, id.png, valid.png (optional: id > 0 when absent)},
        decompose/iterK/{albedo, roughness, metallic, normal_cam, height}.png
Writes  backproject/iterK/textures/<obj>_{basecolor,roughness,metallic,normal,height,coverage}.png
        backproject/iterK/scene.blend (Principled materials, images packed; lights, camera and
        geometry untouched), coverage.json, roundtrip.png, roundtrip.json, roundtrip_err_x4.png,
        roundtrip_novel_<i>.png, backproject_log.txt

UV map: `b2r_uv` (made at ingest). If a mesh has no `b2r_uv`, its active UV map is used and
the fallback is logged and recorded in coverage.json.

Texels the camera never saw are filled from the same object's seen texels (k nearest in 3D,
blended towards the object's robust mean with distance). An object with no seen texel at all
gets the robust mean of the most similar SEEN objects: same category first (all seen
objects if none shares it), then the ones closest in size (sorted bounding-box extents,
log-ratio within SIZE_TOL of the best match), then, if more than DONOR_MAX remain, the
nearest by bounding-box centre. The donors and the rule are recorded per object.

Novel views orbit the camera about the point it looks at (render_view.orbit_camera: median
scene depth along the view axis), so no scene coordinates are needed.

Two halves. Under Blender (bpy) the script exports triangle data, builds materials and renders.
The numeric part (texel rasterisation, visibility, sampling, fill, tangent-space normals,
metrics) runs in the user's Python (numpy + scipy + opencv): Blender re-invokes this file with
--core / --metrics through common.python_exe().

Camera conventions: Blender camera axes (+X right, +Y up, -Z forward); pixel centre of column i
/ row j is (i+0.5, j+0.5); u = cx + fx*Xc/(-Zc), v = cy - fy*Yc/(-Zc); row 0 = top.
Texture convention: PNG row 0 = top = UV v=1; texel (col c, row r) centre is
UV ((c+0.5)/R, 1-(r+0.5)/R).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # .../scripts
sys.path.insert(1, str(Path(__file__).resolve().parent))         # .../scripts/blender
from b2r import common  # noqa: E402

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

import numpy as np  # noqa: E402
import warnings  # noqa: E402

# numpy 2.x + Accelerate on Apple silicon raises spurious FP warnings in matmul
warnings.filterwarnings("ignore", message=".*encountered in matmul")

# --------------------------------------------------------------------------- parameters
COS_GRAZING_MIN = 0.15        # texel seen only if cos(view, geometric normal) > this (~81.4 deg)
COS_GRAZING_FILL = 0.04       # grazing-but-consistent texels (0.04 < cos <= 0.15) give fill values
TOL_PLANE_M = (0.005, 0.002)  # |dot(P_exr - P_texel, Ng)| < a + b * depth
TOL_DIST_PX = 3.0             # |P_exr - P_texel| < 0.01 m + TOL_DIST_PX * depth/fx / max(cos,0.15)
FILL_FALLOFF_M = 0.25         # unseen fill: blend = 1 - exp(-d_nn / FILL_FALLOFF_M) towards robust mean
FILL_KNN = 8
NORMAL_HP_SIGMA_PX = 6.0      # gaussian sigma of the low-pass removed from the normal residual
NORMAL_DEADZONE = 2.0 / 255   # soft threshold on |high-pass residual| (8-bit quantisation)
RES_MIN, RES_MAX, RES_PROBE = 512, 4096, 512
DENSITY_PERCENTILE = 99.0     # "densest visible part" = this percentile of required texels/m
SIZE_TOL = 0.10               # unseen-object donors: mean |log extent ratio| within best + this
DONOR_MAX = 3                 # ... and at most this many, nearest by bbox centre
SIZE_EPS_FRAC = 0.01          # extents get + this fraction of the scene diagonal (thin slabs)


def read_valid(path_valid: Path, ids):
    """valid.png when the g-buffers have one, else id > 0. Returns (mask, source)."""
    import cv2
    if path_valid.is_file():
        v = cv2.imread(str(path_valid), cv2.IMREAD_UNCHANGED)
        if v is None or v.shape[:2] != ids.shape[:2]:
            raise ValueError(f"{path_valid} unreadable or wrong size")
        return (v if v.ndim == 2 else v[..., 0]) > 0, str(path_valid)
    return ids > 0, "id.png > 0 (no valid.png)"


# =========================================================================== CORE (venv)
def core_main(work: Path) -> None:
    import cv2
    from scipy import ndimage
    from scipy.spatial import cKDTree

    job = json.loads((work / "job.json").read_text())
    log_lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print("[backproject-core]", s, flush=True)
        log_lines.append(s)

    t0 = time.time()
    W, H = int(job["width"]), int(job["height"])
    gb = Path(job["gbuffers"])
    maps = Path(job["maps_dir"])
    out_tex = Path(job["tex_dir"])
    out_tex.mkdir(parents=True, exist_ok=True)
    cam = json.loads((gb / "camera.json").read_text())
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    w2c = np.array(cam["world_to_cam"], np.float64)
    c2w = np.array(cam["cam_to_world"], np.float64)
    cam_pos = c2w[:3, 3]

    def rd(path, flags=cv2.IMREAD_UNCHANGED):
        a = cv2.imread(str(path), flags)
        if a is None:
            raise FileNotFoundError(path)
        if a.shape[:2] != (H, W):
            raise ValueError(f"{path} has shape {a.shape}, expected {(H, W)}")
        return a

    def to_unit(a):
        return a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)

    def gray(a):
        return a if a.ndim == 2 else cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2GRAY)

    pos = rd(gb / "position.exr")[..., ::-1].astype(np.float64)        # BGR -> XYZ
    gnrm = rd(gb / "normal.exr")[..., ::-1].astype(np.float64)
    ids = rd(gb / "id.png").astype(np.int64)
    if ids.ndim == 3:
        ids = ids[..., 0]
    valid, valid_src = read_valid(gb / "valid.png", ids)
    log(f"valid mask: {valid_src}")

    alb_raw = rd(maps / "albedo.png")
    albedo = to_unit(alb_raw[..., :3][..., ::-1])                       # sRGB-encoded RGB 0..1
    rough = to_unit(gray(rd(maps / "roughness.png")))
    metal = to_unit(gray(rd(maps / "metallic.png")))
    h_raw = rd(maps / "height.png")
    height_16 = h_raw.dtype == np.uint16
    height = to_unit(gray(h_raw))
    ncam_raw = rd(maps / "normal_cam.png")
    ncam = to_unit(ncam_raw[..., :3][..., ::-1]).astype(np.float64) * 2 - 1
    # camera-space (X right, Y up, Z to camera) == Blender camera axes -> world
    nest = ncam @ c2w[:3, :3].T
    nest /= np.maximum(np.linalg.norm(nest, axis=-1, keepdims=True), 1e-9)

    # normal detail: residual vs the true geometry normal, high-passed within each object
    resid = (nest - gnrm) * valid[..., None]
    resid_hf = np.zeros_like(resid)
    for oid in np.unique(ids[valid]):
        m = ((ids == oid) & valid).astype(np.float64)
        num = cv2.GaussianBlur(resid * m[..., None], (0, 0), NORMAL_HP_SIGMA_PX)
        den = cv2.GaussianBlur(m, (0, 0), NORMAL_HP_SIGMA_PX)
        low = num / np.maximum(den, 1e-9)[..., None]
        resid_hf += (resid - low) * m[..., None]
    # soft dead-zone at the 8-bit quantisation level of normal_cam.png (2/255 per unit):
    # detail below what an 8-bit map can encode is treated as noise, not surface relief.
    mag = np.linalg.norm(resid_hf, axis=-1, keepdims=True)
    resid_hf *= np.clip(1 - NORMAL_DEADZONE / np.maximum(mag, 1e-12), 0, 1)
    # stacked sample image: albedo(3) rough metal height resid_hf(3)
    stack = np.concatenate([albedo, rough[..., None], metal[..., None], height[..., None],
                            resid_hf.astype(np.float32)], -1).astype(np.float32)
    NC = stack.shape[-1]
    strength = float(job["normal_strength"])
    log(f"maps loaded from {maps}; normal residual |r| mean {np.linalg.norm(resid[valid],axis=1).mean():.4f}, "
        f"hf mean {np.linalg.norm(resid_hf[valid],axis=1).mean():.4f}")

    # ----------------------------------------------------------------- helpers
    def rasterise(tri_xy, R):
        """tri_xy (T,3,2) in texel coords (x right, y down). Returns tri map + barycentrics."""
        tmap = np.full((R, R), -1, np.int32)
        bary = np.zeros((R, R, 3), np.float32)
        for t in range(len(tri_xy)):
            p = tri_xy[t].astype(np.float64)
            x0 = max(int(math.floor(p[:, 0].min() - 0.5)), 0)
            x1 = min(int(math.ceil(p[:, 0].max() - 0.5)), R - 1)
            y0 = max(int(math.floor(p[:, 1].min() - 0.5)), 0)
            y1 = min(int(math.ceil(p[:, 1].max() - 0.5)), R - 1)
            if x1 < x0 or y1 < y0:
                continue
            xs = np.arange(x0, x1 + 1) + 0.5
            ys = np.arange(y0, y1 + 1) + 0.5
            X, Y = np.meshgrid(xs, ys)
            (ax, ay), (bx, by), (qx, qy) = p
            den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
            if abs(den) < 1e-12:
                continue
            l0 = ((by - qy) * (X - qx) + (qx - bx) * (Y - qy)) / den
            l1 = ((qy - ay) * (X - qx) + (ax - qx) * (Y - qy)) / den
            l2 = 1 - l0 - l1
            eps = -1e-7
            sub = tmap[y0:y1 + 1, x0:x1 + 1]
            ins = (l0 >= eps) & (l1 >= eps) & (l2 >= eps) & (sub < 0)
            if not ins.any():
                continue
            sub[ins] = t
            bsub = bary[y0:y1 + 1, x0:x1 + 1]
            bsub[ins] = np.stack([l0[ins], l1[ins], l2[ins]], -1)
        return tmap, bary

    def texel_geometry(o, R):
        uv = o["tri_uv"]
        xy = np.stack([uv[..., 0] * R, (1 - uv[..., 1]) * R], -1)
        tmap, bary = rasterise(xy, R)
        rr, cc = np.nonzero(tmap >= 0)
        t = tmap[rr, cc]
        b = bary[rr, cc].astype(np.float64)
        P = np.einsum("nk,nkj->nj", b, o["tri_pos"][t])
        Ns = np.einsum("nk,nkj->nj", b, o["tri_nrm"][t])
        Ns /= np.maximum(np.linalg.norm(Ns, axis=1, keepdims=True), 1e-12)
        return tmap, rr, cc, t, P, Ns

    def project(P):
        pc = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = -pc[:, 2]
        zs = np.where(z > 1e-6, z, 1e-6)
        u = cx + fx * pc[:, 0] / zs
        v = cy - fy * pc[:, 1] / zs
        return u, v, z

    def visibility(oid, P, Ng):
        u, v, z = project(P)
        i = np.floor(u).astype(np.int64)
        j = np.floor(v).astype(np.int64)
        inside = (z > cam["clip_start"]) & (i >= 0) & (i < W) & (j >= 0) & (j < H)
        ic, jc = np.clip(i, 0, W - 1), np.clip(j, 0, H - 1)
        view = cam_pos[None] - P
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
        cosv = np.sum(view * Ng, 1)
        same = inside & valid[jc, ic] & (ids[jc, ic] == oid)
        d = pos[jc, ic] - P
        dplane = np.abs(np.sum(d * Ng, 1))
        dist = np.linalg.norm(d, axis=1)
        tol_plane = TOL_PLANE_M[0] + TOL_PLANE_M[1] * z
        tol_dist = 0.01 + TOL_DIST_PX * z / fx / np.maximum(cosv, COS_GRAZING_MIN)
        seen = same & (dplane < tol_plane) & (dist < tol_dist) & (cosv > COS_GRAZING_MIN)
        diag = {"in_frame": int(inside.sum()), "same_id": int(same.sum()),
                "rej_plane": int((same & ~(dplane < tol_plane)).sum()),
                "rej_dist": int((same & ~(dist < tol_dist)).sum()),
                "rej_grazing": int((same & ~(cosv > COS_GRAZING_MIN)).sum()),
                "seen": int(seen.sum())}
        # fraction of the object's visible pixels hit by at least one seen texel
        hit = np.zeros((H, W), bool)
        hit[jc[seen], ic[seen]] = True
        objpix = (ids == oid) & valid
        diag["visible_pixels_hit_by_seen_texel"] = float(hit[objpix].mean()) if objpix.any() else 0.0
        # second tier: consistent but grazing (COS_GRAZING_FILL < cos <= COS_GRAZING_MIN).
        # NOT counted as seen (coverage 0); their samples are used only as fill values.
        graze = same & (dplane < tol_plane) & (dist < tol_dist) & \
            (cosv > COS_GRAZING_FILL) & ~seen
        diag["grazing_fill_texels"] = int(graze.sum())
        visibility.last = diag
        visibility.graze = graze
        return seen, u, v, z, cosv, tol_dist

    def sample(oid, P, u, v, tol_dist):
        """Bilinear sample of `stack`, restricted to same-id, position-consistent pixels."""
        x, y = u - 0.5, v - 0.5
        i0, j0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        ax_, ay_ = x - i0, y - j0
        acc = np.zeros((len(P), NC), np.float64)
        wsum = np.zeros(len(P), np.float64)
        for di, dj, wgt in ((0, 0, (1 - ax_) * (1 - ay_)), (1, 0, ax_ * (1 - ay_)),
                            (0, 1, (1 - ax_) * ay_), (1, 1, ax_ * ay_)):
            ii, jj = i0 + di, j0 + dj
            inb = (ii >= 0) & (ii < W) & (jj >= 0) & (jj < H)
            ii, jj = np.clip(ii, 0, W - 1), np.clip(jj, 0, H - 1)
            ok = inb & valid[jj, ii] & (ids[jj, ii] == oid) & \
                (np.linalg.norm(pos[jj, ii] - P, axis=1) < 2 * tol_dist)
            w_ = wgt * ok
            acc += w_[:, None] * stack[jj, ii]
            wsum += w_
        # fallback: nearest pixel (same id by the visibility test)
        ni = np.clip(np.floor(u).astype(np.int64), 0, W - 1)
        nj = np.clip(np.floor(v).astype(np.int64), 0, H - 1)
        out = np.where((wsum > 1e-6)[:, None], acc / np.maximum(wsum, 1e-12)[:, None],
                       stack[nj, ni])
        return out, int(np.sum(wsum <= 1e-6))

    def robust_mean(vals):
        lo, hi = np.percentile(vals, 10, axis=0), np.percentile(vals, 90, axis=0)
        res = np.empty(vals.shape[1])
        for c in range(vals.shape[1]):
            col = vals[:, c]
            sel = col[(col >= lo[c]) & (col <= hi[c])]
            res[c] = sel.mean() if len(sel) else col.mean()
        return res

    def pow2_clip(x):
        r = RES_MIN
        while r < x and r < RES_MAX:
            r *= 2
        return r

    def srgb8(a):
        return np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)

    # ----------------------------------------------------------------- per object
    objs = []
    for e in job["objects"]:
        d = np.load(work / e["file"])
        o = dict(e)
        o.update({k: d[k] for k in d.files})
        objs.append(o)

    report = {}
    stats = {}          # name -> seen texel values (for donors)
    pending_unseen = []
    rng = np.random.default_rng(0)

    def finish_object(o, R, tmap, rr, cc, t, P, Ns, seen, vals_seen, fill_info):
        """Fill unseen covered texels, build normal map, dilate gutters, write PNGs."""
        name = o["name"]
        n_cov = len(P)
        vals = np.zeros((n_cov, NC), np.float64)
        unseen = ~seen
        mean = fill_info.pop("mean")
        graze, gvals = fill_info.pop("_graze", (np.zeros(n_cov, bool), None))
        if seen.any():
            vals[seen] = vals_seen
        if graze.any():
            vals[graze] = gvals
        src = seen | graze
        rest = unseen & ~graze
        fill_info["filled_from_grazing_samples_fraction_of_texture"] = float(graze.sum() / (R * R))
        if rest.any():
            if src.any():
                Ps = P[src]
                vs = vals[src]
                if len(Ps) > 1_000_000:
                    sel = rng.choice(len(Ps), 1_000_000, replace=False)
                    Ps, vs = Ps[sel], vs[sel]
                tree = cKDTree(Ps)
                k = min(FILL_KNN, len(Ps))
                dd, ii = tree.query(P[rest], k=k, workers=-1)
                if k == 1:
                    dd, ii = dd[:, None], ii[:, None]
                wk = 1.0 / (dd + 1e-3)
                knn = np.einsum("nk,nkc->nc", wk, vs[ii]) / wk.sum(1, keepdims=True)
                blend = 1 - np.exp(-dd[:, 0] / FILL_FALLOFF_M)
                filled = (1 - blend)[:, None] * knn + blend[:, None] * mean[None]
                fill_info["fill_nn_dist_m"] = {"median": float(np.median(dd[:, 0])),
                                               "p95": float(np.percentile(dd[:, 0], 95)),
                                               "max": float(dd[:, 0].max())}
            else:
                filled = np.repeat(mean[None], rest.sum(), 0)
            vals[rest] = filled
        vals[unseen, 6:9] = 0.0           # unseen -> no normal detail (flat tangent normal)

        # tangent-space normal
        Ng = o["tri_ng"][t]
        Tt = o["tri_tan"][t]
        Bt = o["tri_bit"][t]
        sign = o["tri_sign"][t]
        T_ = Tt - Ns * np.sum(Tt * Ns, 1, keepdims=True)
        T_ /= np.maximum(np.linalg.norm(T_, axis=1, keepdims=True), 1e-12)
        B_ = sign[:, None] * np.cross(Ns, T_)
        nw = Ns + strength * vals[:, 6:9]
        nw /= np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-12)
        M = np.stack([T_, B_, Ns], -1)                   # columns T, B, N
        ct = np.linalg.solve(M, nw[..., None])[..., 0]
        ct /= np.maximum(np.linalg.norm(ct, axis=1, keepdims=True), 1e-12)
        ct[unseen] = (0.0, 0.0, 1.0)
        dev = np.abs(ct - np.array([0.0, 0.0, 1.0])).max(1)
        fill_info["normal_tangent_max_abs_dev_from_flat"] = float(dev.max()) if len(dev) else 0.0
        fill_info["normal_tangent_mean_abs_dev_from_flat"] = float(dev.mean()) if len(dev) else 0.0

        # full texture images with UV-gutter dilation (nearest covered texel in UV)
        cov = tmap >= 0
        full = np.zeros((R, R, NC + 3), np.float32)
        full[rr, cc, :NC] = vals
        full[rr, cc, NC:] = ct
        _, (ir, ic) = ndimage.distance_transform_edt(~cov, return_indices=True)
        full = full[ir, ic]
        seen_img = np.zeros((R, R), np.uint8)
        seen_img[rr[seen], cc[seen]] = 255
        nenc = srgb8(full[..., NC:] * 0.5 + 0.5)
        tex = {
            "basecolor": srgb8(full[..., 0:3])[..., ::-1],
            "roughness": srgb8(full[..., 3]),
            "metallic": srgb8(full[..., 4]),
            "normal": nenc[..., ::-1],
            "height": (np.clip(np.rint(full[..., 5] * 65535), 0, 65535).astype(np.uint16)
                       if height_16 else srgb8(full[..., 5])),
            "coverage": seen_img,
        }
        enc_dev = np.abs(nenc[cov].astype(np.int64) - np.array([128, 128, 255])).max() \
            if cov.any() else 0
        fill_info["normal_png_max_level_dev_from_128_128_255"] = int(enc_dev)
        files = {}
        for k_, img in tex.items():
            pth = out_tex / f"{name}_{k_}.png"
            if not cv2.imwrite(str(pth), img):
                raise RuntimeError(f"cannot write {pth}")
            files[k_] = str(pth)
        n_tex = R * R
        report[name] = {
            "id": o["id"], "resolution": R,
            "resolution_reason": fill_info.pop("res_reason"),
            "visible_pixels": o["visible_pixels"],
            "uv_covered_fraction": float(n_cov / n_tex),
            "seen_fraction_of_texture": float(seen.sum() / n_tex),
            "seen_fraction_of_covered": float(seen.sum() / max(n_cov, 1)),
            "filled_fraction_of_texture": float(1 - seen.sum() / n_tex),
            "filled_3d_fraction_of_texture": float(unseen.sum() / n_tex),
            "gutter_dilated_fraction_of_texture": float(1 - n_cov / n_tex),
            "fill_method": fill_info.pop("method"),
            "robust_mean_rgb_srgb8": [round(float(x) * 255, 1) for x in mean[:3]],
            "robust_mean_rough_metal_height": [round(float(x), 4) for x in mean[3:6]],
            **fill_info,
            "files": files,
        }
        log(f"{name}: R={R} covered {n_cov/n_tex:.3f} seen {seen.sum()/n_tex:.3f} "
            f"of texture ({seen.sum()/max(n_cov,1):.3f} of covered); "
            f"normal dev {report[name]['normal_tangent_max_abs_dev_from_flat']:.4g}")

    for o in objs:
        name, oid = o["name"], int(o["id"])
        ts = time.time()
        o["visible_pixels"] = int(((ids == oid) & valid).sum())
        # area scale: metres per UV unit, per triangle
        e1 = o["tri_pos"][:, 1] - o["tri_pos"][:, 0]
        e2 = o["tri_pos"][:, 2] - o["tri_pos"][:, 0]
        a_w = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        f1 = o["tri_uv"][:, 1] - o["tri_uv"][:, 0]
        f2 = o["tri_uv"][:, 2] - o["tri_uv"][:, 0]
        a_uv = 0.5 * np.abs(f1[:, 0] * f2[:, 1] - f1[:, 1] * f2[:, 0])
        s_tri = np.sqrt(a_w / np.maximum(a_uv, 1e-12))
        # probe pass at RES_PROBE for the resolution choice
        reason = {}
        R = RES_MIN
        if o["visible_pixels"] > 0:
            tmap, rr, cc, t, P, Ns = texel_geometry(o, RES_PROBE)
            seen, u, v, z, cosv, tol = visibility(oid, P, o["tri_ng"][t])
            if seen.any():
                need = s_tri[t[seen]] * fx / z[seen]      # texels per UV unit for 1 texel/px
                req = float(np.percentile(need, DENSITY_PERCENTILE))
                R = pow2_clip(req)
                reason = {"required_res_p%g" % DENSITY_PERCENTILE: round(req, 1),
                          "required_res_max": round(float(need.max()), 1),
                          "clipped_at_max": bool(req > RES_MAX)}
        if not reason:
            reason = {"note": "no visible pixels; minimum resolution"}
        tmap, rr, cc, t, P, Ns = texel_geometry(o, R)
        Ng = o["tri_ng"][t]
        seen, u, v, z, cosv, tol = visibility(oid, P, Ng)
        info = {"res_reason": reason, "rasterise_project_seconds": round(time.time() - ts, 2),
                "visibility_diag": dict(visibility.last)}
        graze = visibility.graze
        if graze.any():
            gv, _ = sample(oid, P[graze], u[graze], v[graze], tol[graze])
            info["_graze"] = (graze, gv)
        if seen.any():
            vs, nfb = sample(oid, P[seen], u[seen], v[seen], tol[seen])
            info["bilinear_fallback_nearest_texels"] = nfb
            # achieved density at the chosen resolution (texels per screen pixel, frontal)
            dens = R / (s_tri[t[seen]] * fx / z[seen])
            info["texels_per_pixel_at_densest_p1"] = round(float(np.percentile(dens, 1)), 3)
            mean = robust_mean(vs)
            mean[6:9] = 0
            stats[name] = (o["category"], vs[:, :6])
            info["mean"] = mean
            info["method"] = ("grazing-angle samples where consistent (0.04<cos<=0.15), else "
                              "knn-%d (inverse-distance) nearest seen/grazing texels in 3D, blended to "
                              "object robust mean with 1-exp(-d/%.2fm); flat normal; UV-gutter "
                              "nearest-texel dilation" % (FILL_KNN, FILL_FALLOFF_M))
            finish_object(o, R, tmap, rr, cc, t, P, Ns, seen, vs, info)
        else:
            pending_unseen.append((o, R, tmap, rr, cc, t, P, Ns, seen, info))

    # bounding boxes for the donor rule (world, from the exported triangles)
    bb = {o["name"]: (o["tri_pos"].reshape(-1, 3).min(0), o["tri_pos"].reshape(-1, 3).max(0)) for o in objs}
    allp = np.concatenate([np.stack(v) for v in bb.values()], 0)
    scene_diag = float(np.linalg.norm(allp.max(0) - allp.min(0))) or 1.0
    size_eps = SIZE_EPS_FRAC * scene_diag

    def donors_for(name, cat):
        """Same category first, then closest in size, then nearest in position."""
        pool = [n for n, (c, _) in stats.items() if cat and c == cat]
        rule = f"seen objects of category '{cat}'"
        if not pool:
            pool = list(stats)
            rule = "all seen objects (none shares the category)"
        ext = np.sort(bb[name][1] - bb[name][0]) + size_eps
        ctr = (bb[name][0] + bb[name][1]) / 2

        def dsize(n):
            e = np.sort(bb[n][1] - bb[n][0]) + size_eps
            return float(np.mean(np.abs(np.log(e / ext))))

        def dpos(n):
            return float(np.linalg.norm((bb[n][0] + bb[n][1]) / 2 - ctr))

        ds = {n: dsize(n) for n in pool}
        best = min(ds.values())
        close = [n for n in pool if ds[n] <= best + SIZE_TOL]
        close.sort(key=dpos)
        chosen = close[:DONOR_MAX]
        rule += (f"; closest in size (mean |log ratio| of sorted bbox extents <= best {best:.3f} + {SIZE_TOL}); "
                 f"at most {DONOR_MAX}, nearest bbox centre first")
        scores = {n: {"size_dissimilarity": round(ds[n], 4), "centre_distance_m": round(dpos(n), 3)}
                  for n in chosen}
        return chosen, rule, scores

    for (o, R, tmap, rr, cc, t, P, Ns, seen, info) in pending_unseen:
        name, cat = o["name"], o["category"]
        if not stats:
            raise RuntimeError("no object has a single seen texel; nothing to fill from")
        donors, rule, scores = donors_for(name, cat)
        info["donor_scores"] = scores
        pooled = np.concatenate([stats[n][1] for n in donors], 0)
        mean = np.zeros(NC)
        mean[:6] = robust_mean(pooled)
        info["mean"] = mean
        info["donors"] = donors
        info["method"] = (f"no visible pixels: constant robust mean (10-90% trimmed) of seen "
                          f"texels pooled from {donors} ({rule}); flat normal")
        finish_object(o, R, tmap, rr, cc, t, P, Ns, seen, None, info)

    summary = {
        "maps_dir": str(maps), "gbuffers": str(gb), "valid_mask": valid_src,
        "parameters": {
            "unseen_object_rule": f"same category first (else all seen objects); keep those whose mean "
                                  f"|log ratio| of sorted bbox extents (+{SIZE_EPS_FRAC} x scene diagonal) "
                                  f"is within {SIZE_TOL} of the best; at most {DONOR_MAX}, nearest bbox "
                                  f"centre first; constant 10-90% trimmed mean of their pooled seen texels",
            "cos_grazing_min": COS_GRAZING_MIN,
            "cos_grazing_fill": COS_GRAZING_FILL,
            "tol_plane_m": f"{TOL_PLANE_M[0]} + {TOL_PLANE_M[1]} * depth",
            "tol_dist_m": f"0.01 + {TOL_DIST_PX} * depth/fx / max(cos, {COS_GRAZING_MIN})",
            "fill_falloff_m": FILL_FALLOFF_M, "fill_knn": FILL_KNN,
            "normal_highpass_sigma_px": NORMAL_HP_SIGMA_PX,
            "normal_strength": strength,
            "normal_deadzone": NORMAL_DEADZONE,
            "normal_detail": "high-pass of (estimated - g-buffer geometric normal) in world space, "
                             "per-object masked gaussian; added to texel shading normal",
            "tangent_frame": "per-triangle analytic dP/du tangent, Gram-Schmidt against the "
                             "interpolated corner normal, sign from UV handedness (== MikkTSpace "
                             "for these all-flat-shaded meshes)",
            "resolution_rule": f"pow2 >= p{DENSITY_PERCENTILE} of s_tri*fx/depth over seen texels "
                               f"(probe at {RES_PROBE}), clipped to [{RES_MIN},{RES_MAX}]",
            "sampling": "bilinear, taps restricted to same-id pixels whose position is within "
                        "2*tol_dist of the texel; nearest-pixel fallback",
        },
        "core_seconds": round(time.time() - t0, 1),
        "objects": report,
    }
    (work / "core_result.json").write_text(json.dumps(summary, indent=2))
    log(f"core done in {time.time() - t0:.1f}s")


# =========================================================================== METRICS (venv)
def metrics_main(work: Path) -> None:
    import cv2
    job = json.loads((work / "metrics_job.json").read_text())
    rt = cv2.imread(job["roundtrip"], cv2.IMREAD_COLOR).astype(np.float64)
    al = cv2.imread(job["albedo"], cv2.IMREAD_COLOR).astype(np.float64)
    ids = cv2.imread(job["id"], cv2.IMREAD_UNCHANGED).astype(np.int64)
    if ids.ndim == 3:
        ids = ids[..., 0]
    if rt is None or al is None or rt.shape != al.shape or rt.shape[:2] != ids.shape:
        raise ValueError("round-trip render, albedo and id map must exist and have the same size")
    valid, _ = read_valid(Path(job["valid"]), ids)
    k = np.ones((5, 5), np.uint8)   # 2 px radius
    idf = ids.astype(np.float32)
    boundary = cv2.dilate(idf, k) != cv2.erode(idf, k)
    m = valid & ~boundary
    err = np.abs(rt - al)[m]                 # per channel
    mae = float(err.mean())
    mse = float(((rt - al)[m] ** 2).mean())
    psnr = float(10 * np.log10(255 ** 2 / max(mse, 1e-12)))
    pix_err = err.max(1)
    frac16 = float((pix_err > 16).mean())
    per_obj = {}
    for oid in np.unique(ids[m]):
        mm = m & (ids == oid)
        e = np.abs(rt - al)[mm]
        per_obj[str(int(oid))] = {"pixels": int(mm.sum()), "mae": round(float(e.mean()), 3),
                                  "frac_err_gt16": round(float((e.max(1) > 16).mean()), 4)}
    errimg = np.clip(np.abs(rt - al).max(2) * 4, 0, 255).astype(np.uint8)
    errimg[~m] = 0
    cv2.imwrite(job["errmap"], errimg)
    res = {"pixels_compared": int(m.sum()), "mask": "valid & >=2 px from any ID boundary",
           "mae_0_255": mae, "psnr_db": psnr,
           "frac_pixels_maxchannel_err_gt16": frac16, "per_object": per_obj}
    (work / "metrics_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "per_object"}))


# =========================================================================== BLENDER
UV_NAME = "b2r_uv"


def blender_main() -> None:
    import bpy
    import render_view as rv

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--maps-dir", default=None)
    ap.add_argument("--blend", default=None)
    ap.add_argument("--gbuffers", default=None)
    ap.add_argument("--normal-strength", type=float, default=1.0)
    ap.add_argument("--bump-strength", type=float, default=0.02)
    ap.add_argument("--bump-distance", type=float, default=0.005)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--novel-orbit-deg", type=float, nargs="*", default=[30.0, 180.0],
                    help="one novel round-trip view per value: orbit about the point the camera looks at")
    ap.add_argument("--novel-dolly", type=float, nargs="*", default=[1.0, 0.6],
                    help="camera-pivot distance factor per novel view (last value repeats)")
    args = ap.parse_args(common.script_args())
    t_start = time.time()
    case = args.case
    W, H = common.render_size(case)

    bdir = common.stage_dir(case, "blockout", args.iter, create=False)
    blend = Path(args.blend) if args.blend else bdir / "blockout.blend"
    gb = Path(args.gbuffers) if args.gbuffers else common.stage_dir(case, "gbuffers", args.iter, create=False)
    maps = Path(args.maps_dir) if args.maps_dir else common.stage_dir(case, "decompose", args.iter, create=False)
    blend, gb, maps = blend.resolve(), gb.resolve(), maps.resolve()
    for pth in (blend, gb / "camera.json", gb / "position.exr", gb / "normal.exr", gb / "id.png"):
        if not pth.exists():
            raise FileNotFoundError(pth)
    for n in ("albedo", "roughness", "metallic", "normal_cam", "height"):
        if not (maps / f"{n}.png").is_file():
            raise FileNotFoundError(f"decomposed map missing: {maps / (n + '.png')}")
    cam_json = common.read_json(gb / "camera.json")
    if (int(cam_json["width"]), int(cam_json["height"])) != (W, H):
        raise ValueError(f"camera.json is {cam_json['width']}x{cam_json['height']}, case size is {W}x{H}")
    out = common.stage_dir(case, "backproject", args.iter)
    tex_dir = out / "textures"
    if tex_dir.exists():
        shutil.rmtree(tex_dir)
    tex_dir.mkdir()
    for old in out.glob("roundtrip_novel*.png"):
        old.unlink()
    work = out / "_work"
    work.mkdir(exist_ok=True)
    log_lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print("[backproject]", s, flush=True)
        log_lines.append(s)

    log(f"case={common.case_dir(case)} iter={args.iter} size={W}x{H}")
    log(f"blend={blend} gbuffers={gb} maps={maps}")

    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH" and not o.hide_render]
    objects_json = blend.parent / "objects.json"
    if objects_json.is_file():
        cats = {o["name"]: o.get("category", "") for o in common.read_json(objects_json)}
    else:
        cats = {}
        log(f"WARNING: no {objects_json}; every object has an empty category")
    ids_seen = {}
    for o in meshes:
        if o.pass_index < 1:
            raise RuntimeError(f"{o.name}: pass_index {o.pass_index}; every mesh needs a unique id >= 1")
        if o.pass_index in ids_seen:
            raise RuntimeError(f"{o.name} and {ids_seen[o.pass_index]} share pass_index {o.pass_index}")
        ids_seen[o.pass_index] = o.name

    # ---------------------------------------------------------------- UV map per object
    uv_used, uv_fallback = {}, {}
    for o in meshes:
        me = o.data
        if UV_NAME in me.uv_layers:
            uv_used[o.name] = UV_NAME
        elif me.uv_layers.active is not None:
            uv_used[o.name] = me.uv_layers.active.name
            uv_fallback[o.name] = me.uv_layers.active.name
        else:
            raise RuntimeError(f"{o.name}: no UV map (expected '{UV_NAME}' from ingest)")
    if uv_fallback:
        log(f"WARNING: no '{UV_NAME}' UV map on {len(uv_fallback)} of {len(meshes)} meshes; using their "
            f"active UV map instead: {sorted(set(uv_fallback.values()))}")

    # ---------------------------------------------------------------- export triangles
    t0 = time.time()
    entries = []
    geom_sig = {}
    for o in meshes:
        me = o.data
        if me.users > 1:
            raise RuntimeError(f"{o.name}: mesh data shared by {me.users} users (make it single-user at ingest)")
        me.calc_loop_triangles()
        nl, nv, nt = len(me.loops), len(me.vertices), len(me.loop_triangles)
        uv = np.empty(nl * 2, np.float64)
        me.uv_layers[uv_used[o.name]].data.foreach_get("uv", uv)
        uv = uv.reshape(nl, 2)
        co = np.empty(nv * 3, np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(nv, 3)
        lv = np.empty(nl, np.int64)
        me.loops.foreach_get("vertex_index", lv)
        cn = np.empty(nl * 3, np.float64)
        me.corner_normals.foreach_get("vector", cn)
        cn = cn.reshape(nl, 3)
        tl = np.empty(nt * 3, np.int64)
        me.loop_triangles.foreach_get("loops", tl)
        tl = tl.reshape(nt, 3)
        Mw = np.array(o.matrix_world, np.float64)
        Nw = np.linalg.inv(Mw[:3, :3]).T
        cow = co @ Mw[:3, :3].T + Mw[:3, 3]
        tri_pos = cow[lv[tl]]                      # (T,3,3)
        tri_uv = uv[tl]                            # (T,3,2)
        tri_nrm = cn[tl] @ Nw.T
        tri_nrm /= np.maximum(np.linalg.norm(tri_nrm, axis=-1, keepdims=True), 1e-12)
        e1 = tri_pos[:, 1] - tri_pos[:, 0]
        e2 = tri_pos[:, 2] - tri_pos[:, 0]
        ng = np.cross(e1, e2)
        ng /= np.maximum(np.linalg.norm(ng, axis=1, keepdims=True), 1e-12)
        flip = np.sum(ng * tri_nrm.mean(1), 1) < 0  # orient like the shading normal
        ng[flip] *= -1
        d1 = tri_uv[:, 1] - tri_uv[:, 0]
        d2 = tri_uv[:, 2] - tri_uv[:, 0]
        det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        dets = np.where(np.abs(det) > 1e-14, det, 1e-14)
        tan = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / dets[:, None]      # dP/du
        bit = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / dets[:, None]      # dP/dv
        sign = np.where(np.sum(np.cross(ng, tan) * bit, 1) < 0, -1.0, 1.0)
        np.savez(work / f"mesh_{o.pass_index}.npz", tri_pos=tri_pos, tri_uv=tri_uv, tri_nrm=tri_nrm,
                 tri_ng=ng, tri_tan=tan, tri_bit=bit, tri_sign=sign)
        entries.append({"name": o.name, "id": int(o.pass_index), "category": cats.get(o.name, ""),
                        "file": f"mesh_{o.pass_index}.npz"})
        geom_sig[o.name] = (nv, nl, float(cow.sum()))
    log(f"exported {len(entries)} meshes in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- core (user's Python)
    job = {"gbuffers": str(gb), "maps_dir": str(maps), "tex_dir": str(tex_dir), "width": W, "height": H,
           "objects": entries, "normal_strength": args.normal_strength}
    (work / "job.json").write_text(json.dumps(job, indent=2))
    env = dict(os.environ, OPENCV_IO_ENABLE_OPENEXR="1")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    r = subprocess.run([common.python_exe(), str(Path(__file__).resolve()), "--core", "--work", str(work)], env=env)
    if r.returncode != 0:
        raise RuntimeError("core projection step failed (see output above)")
    core = json.loads((work / "core_result.json").read_text())
    log(f"core finished in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- materials
    def load_img(path, colorspace):
        img = bpy.data.images.load(str(path), check_existing=False)
        img.name = Path(path).stem
        img.colorspace_settings.name = colorspace
        img.pack()
        return img

    principled = {}
    images = {}
    for o in meshes:
        rep = core["objects"][o.name]
        f = rep["files"]
        imgs = {
            "basecolor": load_img(f["basecolor"], "sRGB"),
            "roughness": load_img(f["roughness"], "Non-Color"),
            "metallic": load_img(f["metallic"], "Non-Color"),
            "normal": load_img(f["normal"], "Non-Color"),
            "height": load_img(f["height"], "Non-Color"),
        }
        images[o.name] = imgs
        mat = bpy.data.materials.new(f"{o.name}_mat")
        try:
            mat.use_nodes = True  # deprecated / always on in Blender 5.x
        except Exception:
            pass
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        L = nt.links

        def node(kind, name, x, y, label=None):
            n = nt.nodes.new(kind)
            n.name = name
            n.label = label or name
            n.location = (x, y)
            return n

        uvname = uv_used[o.name]
        outn = node("ShaderNodeOutputMaterial", "Material Output", 600, 0)
        bsdf = node("ShaderNodeBsdfPrincipled", f"{o.name}_bsdf", 250, 0, "Principled BSDF")
        uvn = node("ShaderNodeUVMap", "UV", -1000, 0)
        uvn.uv_map = uvname
        texn = {}
        for k_, y in (("basecolor", 400), ("roughness", 120), ("metallic", -120),
                      ("normal", -380), ("height", -640)):
            tn = node("ShaderNodeTexImage", f"{o.name}_{k_}_tex", -700, y, f"{k_} texture")
            tn.image = imgs[k_]
            tn.interpolation = "Linear"
            L.new(uvn.outputs["UV"], tn.inputs["Vector"])
            texn[k_] = tn
        nmap = node("ShaderNodeNormalMap", f"{o.name}_normal_map", -350, -380, "Normal Map")
        nmap.space = "TANGENT"
        nmap.uv_map = uvname
        nmap.inputs["Strength"].default_value = 1.0
        bump = node("ShaderNodeBump", f"{o.name}_bump", -100, -560, "Bump (height)")
        bump.inputs["Strength"].default_value = args.bump_strength
        bump.inputs["Distance"].default_value = args.bump_distance
        L.new(texn["basecolor"].outputs["Color"], bsdf.inputs["Base Color"])
        L.new(texn["roughness"].outputs["Color"], bsdf.inputs["Roughness"])
        L.new(texn["metallic"].outputs["Color"], bsdf.inputs["Metallic"])
        L.new(texn["normal"].outputs["Color"], nmap.inputs["Color"])
        L.new(nmap.outputs["Normal"], bump.inputs["Normal"])
        L.new(texn["height"].outputs["Color"], bump.inputs["Height"])
        L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        L.new(bsdf.outputs["BSDF"], outn.inputs["Surface"])
        o.data.materials.clear()
        o.data.materials.append(mat)
        principled[o.name] = mat
    for m in list(bpy.data.materials):  # drop the old (now unused) blockout materials
        if m.users == 0:
            bpy.data.materials.remove(m)

    # geometry unchanged check
    for o in meshes:
        me = o.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        Mw = np.array(o.matrix_world)
        cow = co.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
        if geom_sig[o.name][:2] != (len(me.vertices), len(me.loops)) or \
                abs(geom_sig[o.name][2] - float(cow.sum())) > 1e-6 * max(1.0, abs(geom_sig[o.name][2])):
            raise RuntimeError(f"{o.name}: geometry changed during backprojection")

    scene_path = out / "scene.blend"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path), compress=True)
    log(f"saved {scene_path} ({scene_path.stat().st_size / 1e6:.1f} MB)")

    # ---------------------------------------------------------------- round-trip renders
    saved = {
        "res": (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage),
        "vt": scene.view_settings.view_transform, "look": scene.view_settings.look,
        "exp": scene.view_settings.exposure, "gamma": scene.view_settings.gamma,
        "dither": scene.render.dither_intensity, "cam": scene.camera,
        "lights": {o.name: o.hide_render for o in scene.objects if o.type == "LIGHT"},
    }
    device = rv.setup_device(scene)
    log("render device", device)
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.film_transparent = False
    scene.render.dither_intensity = 0.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    for o in scene.objects:
        if o.type == "LIGHT":
            o.hide_render = True
    rv.set_format(scene.render.image_settings, "PNG", "8", "RGB")

    emis = {}
    for o in meshes:
        m = bpy.data.materials.new(f"_tmp_unlit_{o.name}")
        try:
            m.use_nodes = True
        except Exception:
            pass
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = uv_used[o.name]
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = images[o.name]["basecolor"]
        tn.interpolation = "Linear"
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0
        on = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(uvn.outputs["UV"], tn.inputs["Vector"])
        nt.links.new(tn.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], on.inputs["Surface"])
        o.data.materials[0] = m
        emis[o.name] = m

    def render(path):
        scene.render.filepath = str(path)
        ts = time.time()
        bpy.ops.render.render(write_still=True)
        return time.time() - ts

    t_rt = render(out / "roundtrip.png")
    log(f"roundtrip render {t_rt:.1f}s ({args.samples} spp, box 1px, Standard, lights hidden)")

    # novel views: orbit about the point the camera looks at (no scene coordinates)
    cam0 = saved["cam"]
    novel = {}
    dollies = args.novel_dolly or [1.0]
    for i, yaw in enumerate(args.novel_orbit_deg):
        dol = dollies[min(i, len(dollies) - 1)]
        cd = cam0.data.copy()
        co_ = bpy.data.objects.new("_tmp_novel_cam", cd)
        scene.collection.objects.link(co_)
        co_.matrix_world = cam0.matrix_world.copy()
        bpy.context.view_layer.update()
        scene.camera = co_
        try:
            info = rv.orbit_camera(scene, co_, yaw, dol)
            fname = f"roundtrip_novel_{i}.png"
            info["seconds"] = round(render(out / fname), 1)
            novel[fname] = info
        except RuntimeError as e:  # no geometry in view: say so, write nothing for this view
            novel[f"roundtrip_novel_{i}.png"] = {"skipped": str(e), "yaw_deg": yaw, "dolly": dol}
        scene.camera = cam0
        bpy.data.objects.remove(co_)
        bpy.data.cameras.remove(cd)
    log(f"novel views: {json.dumps(novel)}")

    # restore principled materials + scene settings, verify, drop temps
    for o in meshes:
        o.data.materials[0] = principled[o.name]
    for m in emis.values():
        bpy.data.materials.remove(m)
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = saved["res"]
    scene.view_settings.view_transform = saved["vt"]
    scene.view_settings.look = saved["look"]
    scene.view_settings.exposure = saved["exp"]
    scene.view_settings.gamma = saved["gamma"]
    scene.render.dither_intensity = saved["dither"]
    for n, h in saved["lights"].items():
        scene.objects[n].hide_render = h
    restored_ok = all(o.data.materials[0].name == f"{o.name}_mat" for o in meshes)

    # verify the saved file independently
    mesh_names = {o.name for o in meshes}
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    chk = {"materials_ok": True, "all_images_packed": True, "issues": []}
    for o in [x for x in bpy.context.scene.objects if x.type == "MESH" and x.name in mesh_names]:
        mats = [m.name for m in o.data.materials]
        if mats != [f"{o.name}_mat"]:
            chk["materials_ok"] = False
            chk["issues"].append(f"{o.name}: {mats}")
            continue
        nodes = o.data.materials[0].node_tree.nodes
        if not any(n.type == "BSDF_PRINCIPLED" for n in nodes):
            chk["materials_ok"] = False
            chk["issues"].append(f"{o.name}: no principled")
    for img in bpy.data.images:
        if img.source == "FILE" and img.packed_file is None:
            chk["all_images_packed"] = False
            chk["issues"].append(f"image {img.name} not packed")
    chk["lights"] = [(o.name, o.data.type) for o in bpy.context.scene.objects if o.type == "LIGHT"]
    chk["n_images"] = len(bpy.data.images)

    # ---------------------------------------------------------------- metrics (user's Python)
    mj = {"roundtrip": str(out / "roundtrip.png"), "albedo": str(maps / "albedo.png"),
          "id": str(gb / "id.png"), "valid": str(gb / "valid.png"),
          "errmap": str(out / "roundtrip_err_x4.png")}
    (work / "metrics_job.json").write_text(json.dumps(mj))
    r = subprocess.run([common.python_exe(), str(Path(__file__).resolve()), "--metrics", "--work", str(work)],
                       env=env)
    if r.returncode != 0:
        raise RuntimeError("round-trip metrics failed")
    met = json.loads((work / "metrics_result.json").read_text())
    met.update({"render": {"samples": args.samples, "filter": "box 1.0 px", "size": [W, H],
                           "view_transform": "Standard", "lights_hidden": True,
                           "materials": "unlit emission of basecolor texture, strength 1"},
                "albedo_input": str(maps / "albedo.png"),
                "error_map": mj["errmap"] + " (max-channel abs error x4)",
                "novel_views": novel, "principled_restored_in_memory": restored_ok,
                "saved_scene_check": chk})
    common.write_json(out / "roundtrip.json", met)

    cov = {"case": str(common.case_dir(case)), "iter": args.iter, "maps_dir": str(maps),
           "uv_map": UV_NAME, "uv_fallback_active_map": uv_fallback, **core}
    common.write_json(out / "coverage.json", cov)
    shutil.rmtree(work, ignore_errors=True)
    log(f"total {time.time() - t_start:.1f}s")
    log(json.dumps({k: met[k] for k in ("mae_0_255", "psnr_db", "frac_pixels_maxchannel_err_gt16")}))
    (out / "backproject_log.txt").write_text("\n".join(log_lines) + "\n")
    if not (chk["materials_ok"] and chk["all_images_packed"]):
        print("[backproject] SAVED SCENE CHECK FAILED", chk["issues"])
        sys.exit(1)


# =========================================================================== entry
if __name__ == "__main__":
    try:
        if IN_BLENDER:
            blender_main()
        else:
            ap = argparse.ArgumentParser()
            g = ap.add_mutually_exclusive_group(required=True)
            g.add_argument("--core", action="store_true")
            g.add_argument("--metrics", action="store_true")
            ap.add_argument("--work", required=True)
            a = ap.parse_args(common.script_args())
            (core_main if a.core else metrics_main)(Path(a.work))
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
