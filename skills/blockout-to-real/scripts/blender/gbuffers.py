"""gbuffers: crisp per-pixel data maps + a grey beauty render of the case's blockout.

    blender -b --factory-startup --python scripts/blender/gbuffers.py -- \
        --case CASE [--iter K] [--blend PATH] [--samples N]

Input: CASE/blockout/iter<K>/blockout.blend (--blend overrides). Size: case.json "size".
Writes CASE/gbuffers/iter<K>/ (conventions in references/conventions.md):
    beauty.exr / beauty.png   Cycles beauty with the blend's own lights (AA + denoise; EXR scene-linear,
                              PNG 8-bit sRGB, Standard view transform, exposure 0, no look)
    depth.exr                 float32 1ch planar camera-space Z (m), 0 = no object
    depth.png                 16-bit inverse depth, nearest = white, no object = black
    position.exr              float32 RGB world XYZ of first hit
    normal.exr                float32 RGB world unit normal
    normal_cam.png            8-bit camera-space normal (X right, Y up, Z to camera)
    id.png / id_vis.png       16-bit pass_index / colour-coded ids
    valid.png                 8-bit hit mask
    camera.json, objects_visible.json, selfcheck.json

Data passes are rendered separately: 1 sample, box pixel filter of width 0.01 px, no denoising,
DOF and motion blur off, lights hidden, so every value is the pixel-centre hit (+-0.005 px).

Self-checks (any failure -> exit 1, selfcheck.json "ok": false):
  1 reprojection of position.exr through camera.json lands on the pixel centre (< 0.5 px)
  2 depth.exr equals the planar depth of position.exr
  3 normals are unit length and >= 99% face the camera
  4 every id in id.png is an object of the blend / objects.json; ids are integers (no blended edges)
  5 normal_cam.png convention: camera-space normals agree with normals computed from the position map
    (finite differences inside one object; median cosine >= 0.95)
  6 every map has the case size, and OpenCV (user's Python) reads every file
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from b2r import common  # noqa: E402

import bpy  # noqa: E402
import numpy as np  # noqa: E402
import OpenImageIO as oiio  # noqa: E402

VIEW_TRANSFORM = "Standard"


# --------------------------------------------------------------------------- io helpers
def write_img(path: Path, arr: np.ndarray, fmt) -> None:
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    spec = oiio.ImageSpec(w, h, c, fmt)
    if str(path).endswith(".exr"):
        spec.attribute("compression", "zip")
        if c == 1:
            spec.channelnames = ("Y",)
        elif c == 3:
            spec.channelnames = ("R", "G", "B")
    out = oiio.ImageOutput.create(str(path))
    if out is None or not out.open(str(path), spec):
        raise RuntimeError(f"cannot open {path} for writing: {oiio.geterror()}")
    if not out.write_image(arr):
        raise RuntimeError(f"write failed {path}: {out.geterror()}")
    out.close()


def read_multilayer(path: Path) -> dict[str, np.ndarray]:
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"cannot open {path}: {oiio.geterror()}")
    # Blender 5 writes multilayer EXR as multi-part: one part (subimage) per pass.
    chans = {}
    k = 0
    while inp.seek_subimage(k, 0):
        spec = inp.spec()
        px = inp.read_image(k, 0, 0, spec.nchannels, oiio.FLOAT)  # (h, w, c), row 0 = top
        if px is None:
            raise RuntimeError(f"read failed {path} part {k}: {inp.geterror()}")
        px = px.reshape(spec.height, spec.width, spec.nchannels)
        for i, name in enumerate(spec.channelnames):
            chans[name] = px[:, :, i]
        k += 1
    inp.close()
    return chans


def pick(ch: dict, pass_name: str, comps: str) -> np.ndarray:
    out = []
    for c in comps:
        keys = [k for k in ch if k.endswith(f".{pass_name}.{c}") or k == f"{pass_name}.{c}"]
        if len(keys) != 1:
            raise KeyError(f"pass {pass_name}.{c} not found uniquely; channels: {sorted(ch)}")
        out.append(ch[keys[0]])
    return np.stack(out, -1) if len(out) > 1 else out[0]


# --------------------------------------------------------------------------- setup
def setup_device(scene) -> str:
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        names = []
        for d in prefs.devices:
            d.use = d.type == backend
            if d.use:
                names.append(d.name)
        if names:
            scene.cycles.device = "GPU"
            return f"GPU ({backend}): " + ", ".join(names)
    scene.cycles.device = "CPU"
    print("[gbuffers] NOTE no GPU device found for Cycles; rendering on CPU")
    return "CPU"


def set_format(settings, fmt: str, depth: str, mode: str) -> None:
    if hasattr(settings, "media_type"):
        settings.media_type = "MULTI_LAYER_IMAGE" if fmt == "OPEN_EXR_MULTILAYER" else "IMAGE"
    settings.file_format = fmt
    settings.color_mode = mode
    settings.color_depth = depth
    if fmt.startswith("OPEN_EXR"):
        settings.exr_codec = "ZIP"


def camera_model(scene, cam) -> dict:
    W, H = scene.render.resolution_x, scene.render.resolution_y
    assert scene.render.resolution_percentage == 100
    cd = cam.data
    if cd.type != "PERSP":
        raise RuntimeError(f"camera type {cd.type} unsupported (need PERSP)")
    ax, ay = scene.render.pixel_aspect_x, scene.render.pixel_aspect_y
    if abs(ax - ay) > 1e-9:
        raise RuntimeError("non-square pixel aspect unsupported")
    fit = cd.sensor_fit
    if fit == "AUTO":
        fit = "HORIZONTAL" if W >= H else "VERTICAL"
    if fit == "HORIZONTAL":
        sensor = cd.sensor_width
        f_px = cd.lens / sensor * W
        maxdim = W
    else:
        sensor = cd.sensor_height if cd.sensor_fit == "VERTICAL" else cd.sensor_width
        f_px = cd.lens / sensor * H
        maxdim = H
    # Blender shift is in units of the fitted dimension; +shift_x moves the frame right,
    # so the principal point moves left in the image (v grows downwards).
    cx = W / 2.0 - cd.shift_x * maxdim
    cy = H / 2.0 + cd.shift_y * maxdim
    c2w = np.array(cam.matrix_world, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    return {
        "width": W, "height": H,
        "fx": f_px, "fy": f_px, "cx": cx, "cy": cy,
        "fov_x_deg": math.degrees(2 * math.atan(W / (2 * f_px))),
        "fov_y_deg": math.degrees(2 * math.atan(H / (2 * f_px))),
        "clip_start": cd.clip_start, "clip_end": cd.clip_end,
        "lens_mm": cd.lens, "sensor_width_mm": cd.sensor_width,
        "sensor_height_mm": cd.sensor_height, "sensor_fit": cd.sensor_fit,
        "shift_x": cd.shift_x, "shift_y": cd.shift_y,
        "camera_name": cam.name,
        "cam_location": list(c2w[:3, 3]),
        "cam_to_world": c2w.tolist(),
        "world_to_cam": w2c.tolist(),
        "convention": ("Blender camera axes: +X right, +Y up, -Z forward. Pixel (u,v), "
                       "v down, pixel centre of column i / row j is (i+0.5, j+0.5): "
                       "u = cx + fx*Xc/(-Zc), v = cy - fy*Yc/(-Zc). Row 0 = top."),
    }


def id_colours(ids) -> dict[int, tuple[int, int, int]]:
    out = {}
    for k, i in enumerate(sorted(ids)):
        h = (k * 0.6180339887) % 1.0
        s = 0.55 + 0.45 * ((k * 7) % 3) / 2
        v = 0.75 + 0.25 * ((k * 5) % 2)
        out[i] = tuple(int(round(255 * c)) for c in colorsys.hsv_to_rgb(h, s, v))
    return out


def position_normals(pc: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-space normals from central differences of camera-space positions, where the 4
    neighbours belong to the same object. n = dP/dv x dP/du (v down) faces the camera."""
    same = np.zeros(ids.shape, bool)
    c = ids[1:-1, 1:-1]
    same[1:-1, 1:-1] = ((c > 0) & (ids[1:-1, 2:] == c) & (ids[1:-1, :-2] == c)
                        & (ids[2:, 1:-1] == c) & (ids[:-2, 1:-1] == c))
    du = np.zeros_like(pc)
    dv = np.zeros_like(pc)
    du[1:-1, 1:-1] = pc[1:-1, 2:] - pc[1:-1, :-2]
    dv[1:-1, 1:-1] = pc[2:, 1:-1] - pc[:-2, 1:-1]
    n = np.cross(dv, du)
    ln = np.linalg.norm(n, axis=-1)
    ok = same & (ln > 1e-12)
    n[ok] /= ln[ok][:, None]
    return n, ok


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("--blend", default=None)
    ap.add_argument("--samples", type=int, default=512)
    args = ap.parse_args(common.script_args())

    t_start = time.time()
    RW, RH = common.render_size(args.case)
    blend = Path(args.blend).resolve() if args.blend else \
        common.stage_dir(args.case, "blockout", args.iter, create=False) / "blockout.blend"
    if not blend.is_file():
        raise FileNotFoundError(blend)
    out = common.stage_dir(args.case, "gbuffers", args.iter)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        raise RuntimeError("blend has no scene camera")
    scene.render.resolution_x = RW
    scene.render.resolution_y = RH
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.use_border = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    device = setup_device(scene)
    vl = bpy.context.view_layer
    for other in scene.view_layers:
        if other != vl:
            other.use = False

    meshes = [o for o in scene.objects if o.type == "MESH" and not o.hide_render]
    by_index = {o.pass_index: o.name for o in meshes}
    if len(by_index) != len(meshes) or 0 in by_index:
        raise RuntimeError("mesh pass_index values must be unique and >= 1")
    others = [o.name for o in scene.objects if o.type not in ("MESH", "LIGHT", "CAMERA", "EMPTY") and not o.hide_render]
    objects_json = blend.parent / "objects.json"
    spec_objs = {int(o["id"]): o["name"] for o in common.read_json(objects_json)} \
        if objects_json.is_file() else dict(by_index)

    # ---------------------------------------------------------------- beauty pass
    scene.view_settings.view_transform = VIEW_TRANSFORM
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.005
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    scene.cycles.filter_width = 1.5
    scene.render.film_transparent = False
    for attr in ("use_pass_z", "use_pass_normal", "use_pass_position", "use_pass_object_index"):
        setattr(vl, attr, False)
    set_format(scene.render.image_settings, "OPEN_EXR", "32", "RGB")
    scene.render.filepath = str(out / "beauty.exr")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    t_beauty = time.time() - t0
    rr = bpy.data.images["Render Result"]
    # The grey beauty is conditioning for the image model and the LPIPS baseline, so it must be
    # readable whatever the input scene's own lights are (a studio softbox at hundreds of watts
    # blows a grey scene out to white). Expose the PNG so the 95th luminance percentile sits at
    # 0.8 scene-linear; beauty.exr stays untouched. The exposure used is recorded.
    lin = read_multilayer(out / "beauty.exr")
    rgb = [lin[c] for c in lin if c.split(".")[-1] in ("R", "G", "B")][:3]
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    p95 = float(np.percentile(lum[np.isfinite(lum)], 95))
    beauty_exposure = 0.0
    if p95 > 1e-6:
        stops = math.log2(0.8 / p95)
        if abs(stops) > 0.25:
            beauty_exposure = max(-10.0, min(10.0, stops))
    scene.view_settings.exposure = beauty_exposure
    set_format(scene.render.image_settings, "PNG", "8", "RGB")
    rr.save_render(filepath=str(out / "beauty.png"), scene=scene)
    scene.view_settings.exposure = 0.0
    print(f"[gbuffers] beauty: {args.samples} spp, {t_beauty:.1f}s, PNG exposure {beauty_exposure:+.2f} stops")

    # ---------------------------------------------------------------- data pass
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0
    scene.cycles.seed = 0
    scene.render.use_motion_blur = False
    dof_was = cam.data.dof.use_dof
    cam.data.dof.use_dof = False
    for o in scene.objects:  # lights must not occlude geometry in the data pass
        if o.type == "LIGHT":
            o.hide_render = True
    vl.use_pass_z = True
    vl.use_pass_normal = True
    vl.use_pass_position = True
    vl.use_pass_object_index = True
    set_format(scene.render.image_settings, "OPEN_EXR_MULTILAYER", "32", "RGBA")
    ml_path = tmp / "data_multilayer.exr"
    scene.render.filepath = str(ml_path)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    t_data = time.time() - t0
    ch = read_multilayer(ml_path)
    pos = pick(ch, "Position", "XYZ").astype(np.float32)
    nrm = pick(ch, "Normal", "XYZ").astype(np.float32)
    idx_f = pick(ch, "Object Index", "X")
    zpass = pick(ch, "Depth", "Z") if any(k.endswith("Depth.Z") for k in ch) else None
    H, W = idx_f.shape
    if (W, H) != (RW, RH):
        raise RuntimeError(f"data pass is {W}x{H}, case size is {RW}x{RH}")

    ids = np.rint(idx_f).astype(np.int64)
    id_nonint = int(np.sum(np.abs(idx_f - ids) > 1e-6))
    valid = ids > 0
    if not valid.any():
        raise RuntimeError("no object is visible from the camera")

    # ---------------------------------------------------------------- camera + derived maps
    camj = camera_model(scene, cam)
    camj["dof_disabled_for_data_pass"] = bool(dof_was)
    w2c = np.array(camj["world_to_cam"])
    c2w = np.array(camj["cam_to_world"])
    R = w2c[:3, :3]
    pc = pos.astype(np.float64) @ R.T + w2c[:3, 3]
    depth = np.where(valid, -pc[..., 2], 0.0).astype(np.float32)
    pos = np.where(valid[..., None], pos, 0.0).astype(np.float32)
    nrm = np.where(valid[..., None], nrm, 0.0).astype(np.float32)
    ncam = nrm.astype(np.float64) @ R.T  # rotation only

    write_img(out / "position.exr", pos, oiio.FLOAT)
    write_img(out / "normal.exr", nrm, oiio.FLOAT)
    write_img(out / "depth.exr", depth, oiio.FLOAT)

    inv = np.zeros_like(depth, dtype=np.float64)
    inv[valid] = 1.0 / depth[valid]
    lo, hi = inv[valid].min(), inv[valid].max()
    dvis = np.zeros_like(inv)
    dvis[valid] = (inv[valid] - lo) / max(hi - lo, 1e-12)
    write_img(out / "depth.png", np.rint(dvis * 65535).astype(np.uint16), oiio.UINT16)

    nc_enc = np.where(valid[..., None], ncam * 0.5 + 0.5, 0.0)
    write_img(out / "normal_cam.png", np.clip(np.rint(nc_enc * 255), 0, 255).astype(np.uint8), oiio.UINT8)
    write_img(out / "id.png", ids.astype(np.uint16), oiio.UINT16)
    write_img(out / "valid.png", (valid * 255).astype(np.uint8), oiio.UINT8)
    cols = id_colours(set(spec_objs) | set(by_index))
    vis = np.zeros((H, W, 3), np.uint8)
    for i, c in cols.items():
        vis[ids == i] = c
    write_img(out / "id_vis.png", vis, oiio.UINT8)
    common.write_json(out / "camera.json", camj)

    # ---------------------------------------------------------------- objects_visible
    ov = {}
    for i in sorted(set(spec_objs) | set(by_index)):
        m = ids == i
        n = int(m.sum())
        entry = {"name": spec_objs.get(i, by_index.get(i)), "pixels": n, "fraction": n / (W * H), "bbox": None}
        if n:
            ys, xs = np.nonzero(m)
            entry["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        ov[str(i)] = entry
    common.write_json(out / "objects_visible.json", ov)

    # ---------------------------------------------------------------- self-checks
    failures = []
    sc = {"blend": blend.name, "blockout_iter": args.iter, "size": [W, H], "device": device,
          "view_transform": VIEW_TRANSFORM, "beauty_png_exposure_stops": round(beauty_exposure, 3),
          "beauty_samples": args.samples, "beauty_adaptive_threshold": 0.005,
          "beauty_denoiser": "OpenImageDenoise (albedo+normal)", "beauty_filter": "Blackman-Harris 1.5 px",
          "data_samples": 1, "data_filter": "box 0.01 px", "data_denoise": False,
          "render_seconds": {"beauty": round(t_beauty, 2), "data": round(t_data, 2)},
          "valid_pixels": int(valid.sum()), "valid_fraction": float(valid.mean()),
          "non_mesh_renderables_ignored": others}

    # 1 reprojection
    jj, ii = np.nonzero(valid)
    P = pc[jj, ii]
    u = camj["cx"] + camj["fx"] * P[:, 0] / (-P[:, 2])
    v = camj["cy"] - camj["fy"] * P[:, 1] / (-P[:, 2])
    err = np.hypot(u - (ii + 0.5), v - (jj + 0.5))
    sc["reproj_err_px"] = {"mean": float(err.mean()), "max": float(err.max()), "p99": float(np.percentile(err, 99))}
    if err.max() >= 0.5:
        failures.append(f"reprojection max error {err.max():.4f} px >= 0.5")
    cj = common.read_json(out / "camera.json")
    w2c_j = np.array(cj["world_to_cam"])
    ph = np.concatenate([pos[jj, ii].astype(np.float64), np.ones((len(jj), 1))], 1) @ w2c_j.T
    u2 = cj["cx"] + cj["fx"] * ph[:, 0] / (-ph[:, 2])
    v2 = cj["cy"] - cj["fy"] * ph[:, 1] / (-ph[:, 2])
    sc["reproj_err_px_from_json_float32_position"] = float(np.hypot(u2 - (ii + 0.5), v2 - (jj + 0.5)).max())
    if sc["reproj_err_px_from_json_float32_position"] >= 0.5:
        failures.append("reprojection through camera.json >= 0.5 px")
    sc["cam_to_world_times_world_to_cam_err"] = float(np.abs(c2w @ w2c - np.eye(4)).max())

    # 2 depth vs position
    dd = np.abs(depth[valid].astype(np.float64) + ph[:, 2])
    sc["depth_vs_position_max_abs_diff_m"] = float(dd.max())
    tol = 1e-4 * max(1.0, float(depth[valid].max()) / 10.0)  # float32 positions: relative tolerance far away
    if dd.max() > tol:
        failures.append(f"depth vs position diff {dd.max()} > {tol}")
    sc["depth_range_m"] = [float(depth[valid].min()), float(depth[valid].max())]
    if zpass is not None:
        zp = zpass[valid].astype(np.float64)
        dv = depth[valid].astype(np.float64)
        rad = np.linalg.norm(P, axis=1)
        sc["cycles_depth_pass_vs"] = {"planar_max_abs_diff": float(np.abs(zp - dv).max()),
                                      "radial_max_abs_diff": float(np.abs(zp - rad).max()),
                                      "note": "info only; depth.exr is derived from position"}
    if np.any(depth[valid] <= 0):
        failures.append("non-positive depth on valid pixels")

    # 3 normals
    nl = np.linalg.norm(nrm[valid].astype(np.float64), axis=1)
    sc["normal_len"] = {"min": float(nl.min()), "max": float(nl.max()), "max_abs_dev": float(np.abs(nl - 1).max())}
    if np.abs(nl - 1).max() > 1e-3:
        failures.append(f"normals not unit, max dev {np.abs(nl - 1).max()}")
    view = c2w[:3, 3][None] - pos[jj, ii].astype(np.float64)
    facing = np.sum(view * nrm[jj, ii], 1) > 0
    sc["normal_facing_camera_fraction"] = float(facing.mean())
    if facing.mean() < 0.99:
        failures.append(f"only {facing.mean():.4f} of normals face camera")
    if (~facing).any():
        bad = np.unique(ids[jj[~facing], ii[~facing]], return_counts=True)
        sc["normal_backfacing_by_id"] = {str(int(a)): int(b) for a, b in zip(*bad)}

    # 4 ids
    present = sorted(int(x) for x in np.unique(ids) if x > 0)
    unknown = [i for i in present if i not in spec_objs]
    sc["ids_present"] = present
    sc["ids_unknown"] = unknown
    sc["id_nonint_pixels"] = id_nonint
    sc["objects_zero_pixels"] = [spec_objs[i] for i in sorted(spec_objs) if i not in present]
    missing_in_blend = [i for i in spec_objs if i not in by_index]
    sc["objects_json_ids_missing_in_blend"] = missing_in_blend
    if unknown:
        failures.append(f"unknown ids {unknown}")
    if missing_in_blend:
        failures.append(f"objects.json ids not in the blend: {missing_in_blend}")
    if id_nonint:
        failures.append(f"{id_nonint} pixels with non-integer id (blended edges)")

    # 5 normal_cam convention vs geometry
    pn, ok = position_normals(pc, ids)
    sc["normal_cam_vs_position_gradient"] = None
    if ok.sum() >= 100:
        cos = np.sum(pn[ok] * ncam[ok], axis=1)
        sc["normal_cam_vs_position_gradient"] = {
            "pixels": int(ok.sum()), "median_cos": float(np.median(cos)),
            "fraction_cos_gt_0.9": float((cos > 0.9).mean())}
        if np.median(cos) < 0.95:
            failures.append(f"normal_cam disagrees with position-derived normals (median cos {np.median(cos):.3f})")
    else:
        failures.append("too few interior pixels to check the normal_cam convention")
    up = valid & (nrm[..., 2] > 0.9)  # info: upward-facing world surfaces in normal_cam.png
    if up.any():
        enc = np.clip(np.rint(nc_enc * 255), 0, 255)[up].mean(0)
        sc["upward_surfaces_normal_cam_png_mean_rgb"] = [float(x) for x in enc]

    # 6 sizes and OpenCV readability from the user's Python
    bi = oiio.ImageInput.open(str(out / "beauty.png"))
    bs = (bi.spec().width, bi.spec().height)
    bi.close()
    sc["beauty_png_size"] = list(bs)
    if bs != (W, H):
        failures.append(f"beauty.png size {bs}")
    code = (
        "import os,sys,json;os.environ['OPENCV_IO_ENABLE_OPENEXR']='1';import cv2;r={}\n"
        "for f in sys.argv[1:]:\n"
        "  a=cv2.imread(f,cv2.IMREAD_UNCHANGED)\n"
        "  r[os.path.basename(f)]=None if a is None else [list(a.shape),str(a.dtype)]\n"
        "print(json.dumps(r))")
    files = [str(out / n) for n in ("beauty.exr", "depth.exr", "position.exr", "normal.exr", "depth.png",
                                    "normal_cam.png", "id.png", "id_vis.png", "valid.png", "beauty.png")]
    res = None
    try:
        res = subprocess.run([common.python_exe(), "-c", code, *files], capture_output=True, text=True, timeout=120)
        cvr = json.loads(res.stdout.strip().splitlines()[-1])
        sc["opencv_read"] = cvr
        for k, val in cvr.items():
            if val is None:
                failures.append(f"opencv cannot read {k}")
            elif tuple(val[0][:2]) != (H, W):
                failures.append(f"{k} shape {val[0]}")
    except Exception as exc:
        failures.append(f"opencv readability check failed to run ({common.python_exe()}): {exc}; "
                        f"{res.stderr[-500:] if res is not None else ''}")

    sc["total_seconds"] = round(time.time() - t_start, 2)
    sc["failures"] = failures
    sc["ok"] = not failures
    common.write_json(out / "selfcheck.json", sc)
    ml_path.unlink(missing_ok=True)
    try:
        tmp.rmdir()
    except OSError:
        pass
    print(json.dumps(sc, indent=1))
    if failures:
        print("[gbuffers] SELF-CHECK FAILED:", *failures, sep="\n  ")
        sys.exit(1)
    print("[gbuffers] all self-checks passed")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
