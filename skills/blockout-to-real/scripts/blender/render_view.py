"""Render any .blend with Cycles (GPU when available, denoised), optionally from a moved camera
or with scaled / tinted lights. Nothing is saved back to the .blend.

    blender -b --factory-startup --python scripts/blender/render_view.py -- \
        --blend PATH --out PATH.png [--exr PATH.exr] [--samples N] [--scale F] \
        [--cam-loc x,y,z [--cam-target x,y,z]] [--orbit-deg YAW] [--dolly F] \
        [--light-scale NAME=FACTOR ...] [--light-color NAME=r,g,b ...] \
        [--sun-rotation-z DEG] [--view-transform Standard] [--exposure EV]

* Size = the .blend's own render resolution (x its percentage) times --scale.
* --cam-loc / --cam-target move and aim the scene camera (world coordinates).
* --orbit-deg / --dolly need no coordinates: the pivot is the point the camera looks at,
  at the median scene depth along its view axis (ray casts over a grid of the view frustum).
  --orbit-deg turns the camera about the world Z axis through the pivot (positive =
  counter-clockwise seen from above), --dolly F scales the camera-pivot distance by F
  (F < 1 moves closer). The camera keeps looking at the pivot.
* --light-scale / --light-color address light objects by exact name or a shell pattern
  ('virtual_*'); the name "world" addresses the world background.
* --sun-rotation-z rotates every SUN light about the world Z axis.
* The PNG goes through the view transform (default Standard, look None, exposure 0: directly
  comparable to an sRGB photo); the EXR is scene-linear.

Other Blender scripts import the helpers (setup_device, configure_cycles, set_colour_management,
render_to, look_rotation, view_pivot, orbit_camera); main() only runs as a script.
"""
from __future__ import annotations

import argparse
import fnmatch
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # .../scripts
from b2r import common  # noqa: E402

import bpy  # noqa: E402
from mathutils import Euler, Matrix, Vector  # noqa: E402

GPU_BACKENDS = ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI")


# --------------------------------------------------------------------------- helpers
def setup_device(scene) -> str:
    """Cycles on the first GPU backend that has a device; CPU (said so) otherwise."""
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in GPU_BACKENDS:
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
            return f"GPU {backend}: " + ", ".join(names)
    scene.cycles.device = "CPU"
    return "CPU (no GPU device found)"


def set_format(settings, fmt: str, depth: str, mode: str) -> None:
    if hasattr(settings, "media_type"):
        settings.media_type = "MULTI_LAYER_IMAGE" if fmt == "OPEN_EXR_MULTILAYER" else "IMAGE"
    settings.file_format = fmt
    settings.color_mode = mode
    settings.color_depth = depth
    if fmt.startswith("OPEN_EXR"):
        settings.exr_codec = "ZIP"


def scene_size(scene) -> tuple[int, int]:
    """The .blend's own output size in pixels."""
    p = scene.render.resolution_percentage / 100.0
    return int(round(scene.render.resolution_x * p)), int(round(scene.render.resolution_y * p))


def configure_cycles(scene, samples: int, width: int, height: int, adaptive_threshold: float = 0.005,
                     denoise: bool = True) -> None:
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.film_transparent = False
    scene.cycles.samples = int(samples)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = adaptive_threshold
    scene.cycles.use_denoising = denoise
    if denoise:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    scene.cycles.filter_width = 1.5
    for other in scene.view_layers[1:]:
        other.use = False


def set_colour_management(scene, view_transform: str = "Standard", exposure: float = 0.0) -> None:
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = view_transform
    scene.view_settings.look = "None"
    scene.view_settings.exposure = exposure
    scene.view_settings.gamma = 1.0
    try:
        scene.sequencer_colorspace_settings.name = "sRGB"
    except (AttributeError, TypeError):
        pass


def render_to(scene, png: Path | None = None, exr: Path | None = None) -> float:
    """Render once; write a linear EXR and/or a view-transformed 8-bit PNG. Returns seconds."""
    if png is None and exr is None:
        raise ValueError("need png or exr")
    t0 = time.time()
    if exr is not None:
        Path(exr).parent.mkdir(parents=True, exist_ok=True)
        set_format(scene.render.image_settings, "OPEN_EXR", "32", "RGB")
        scene.render.filepath = str(exr)
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(write_still=False)
    dt = time.time() - t0
    if png is not None:
        Path(png).parent.mkdir(parents=True, exist_ok=True)
        set_format(scene.render.image_settings, "PNG", "8", "RGB")
        bpy.data.images["Render Result"].save_render(filepath=str(png), scene=scene)
    return dt


def look_rotation(forward: Vector, up_hint: Vector = Vector((0, 0, 1))) -> Euler:
    """Rotation whose local -Z points along `forward` (cameras, area / sun / spot lights)."""
    f = Vector(forward).normalized()
    z = -f
    up = up_hint if abs(up_hint.normalized().dot(f)) < 0.999 else Vector((0, 1, 0))
    x = up.cross(z).normalized()
    y = z.cross(x).normalized()
    return Matrix((x, y, z)).transposed().to_euler()


def camera_rays(scene, cam, nx: int = 32, ny: int = 24):
    """(origin, unit direction) world-space rays through an nx x ny grid of the view frustum."""
    mw = cam.matrix_world
    rot = mw.to_3x3()
    frame = [Vector(v) for v in cam.data.view_frame(scene=scene)]  # camera local, z < 0
    xs = [v.x for v in frame]
    ys = [v.y for v in frame]
    z = frame[0].z
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    rays = []
    for j in range(ny):
        for i in range(nx):
            px = x0 + (x1 - x0) * (i + 0.5) / nx
            py = y0 + (y1 - y0) * (j + 0.5) / ny
            if cam.data.type == "ORTHO":
                o = mw @ Vector((px, py, 0.0))
                d = rot @ Vector((0.0, 0.0, -1.0))
            else:
                o = mw.translation.copy()
                d = rot @ Vector((px, py, z))
            rays.append((o, d.normalized()))
    return rays


def view_pivot(scene, cam):
    """Point the camera looks at: camera + forward * median planar depth of the scene over a
    grid of view rays. Returns (pivot, depth, hit_fraction); raises if no ray hits geometry."""
    dg = bpy.context.evaluated_depsgraph_get()
    fwd = (cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
    far = cam.data.clip_end
    depths = []
    rays = camera_rays(scene, cam)
    for o, d in rays:
        hit, loc, _n, _i, _ob, _m = scene.ray_cast(dg, o, d, distance=far)
        if hit:
            depths.append((loc - cam.matrix_world.translation).dot(fwd))
    if not depths:
        raise RuntimeError("no camera ray hits geometry: cannot find the point the camera looks at")
    depths.sort()
    dep = depths[len(depths) // 2]
    return cam.matrix_world.translation + fwd * dep, dep, len(depths) / len(rays)


def orbit_camera(scene, cam, yaw_deg: float = 0.0, dolly: float = 1.0) -> dict:
    """Orbit `cam` by yaw_deg about the world Z axis through view_pivot(), then scale its
    distance to the pivot by `dolly`; the camera ends up looking at the pivot."""
    pivot, dep, frac = view_pivot(scene, cam)
    loc0 = cam.matrix_world.translation.copy()
    rz = Matrix.Rotation(math.radians(yaw_deg), 3, "Z")
    loc = pivot + rz @ (loc0 - pivot)
    loc = pivot + (loc - pivot) * float(dolly)
    # keep the camera in free space: if geometry lies between pivot and the new spot, stop short of it
    clamped = False
    span = (loc - pivot).length
    if span > 1e-6:
        dg = bpy.context.evaluated_depsgraph_get()
        hit, hloc, _n, _i, _ob, _m = scene.ray_cast(dg, pivot, (loc - pivot).normalized(), distance=span)
        if hit:
            loc = pivot + (hloc - pivot) * 0.9
            clamped = True
    cam.rotation_mode = "XYZ"
    cam.location = loc
    cam.rotation_euler = look_rotation(pivot - loc)
    bpy.context.view_layer.update()
    return {"pivot": [round(x, 4) for x in pivot], "pivot_depth_m": round(dep, 4),
            "ray_hit_fraction": round(frac, 3), "yaw_deg": yaw_deg, "dolly": dolly,
            "clamped_to_free_space": clamped,
            "location_before": [round(x, 4) for x in loc0], "location": [round(x, 4) for x in loc]}


def _vec(s: str) -> Vector:
    v = [float(x) for x in s.split(",")]
    if len(v) != 3:
        raise ValueError(f"expected x,y,z got {s!r}")
    return Vector(v)


def _kv(items, parse):
    out = []
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"expected NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out.append((k, parse(v)))
    return out


def world_background(scene):
    w = scene.world
    if w is None or w.node_tree is None:
        return None
    for n in w.node_tree.nodes:
        if n.bl_idname == "ShaderNodeBackground":
            return n
    return None


def _match_lights(scene, pattern: str):
    lights = [o for o in scene.objects if o.type == "LIGHT"]
    hit = [o for o in lights if o.name == pattern] or \
          [o for o in lights if fnmatch.fnmatchcase(o.name, pattern)]
    if not hit:
        raise KeyError(f"no light matches {pattern!r}; lights: {[o.name for o in lights]}")
    return hit


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True, help="PNG (view-transformed, 8-bit)")
    ap.add_argument("--exr", default=None, help="optional scene-linear EXR")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--cam-loc", default=None)
    ap.add_argument("--cam-target", default=None)
    ap.add_argument("--orbit-deg", type=float, default=0.0,
                    help="orbit about world Z through the point the camera looks at")
    ap.add_argument("--dolly", type=float, default=1.0,
                    help="scale the camera's distance to the point it looks at")
    ap.add_argument("--light-scale", nargs="*", default=[], metavar="NAME=FACTOR")
    ap.add_argument("--light-color", nargs="*", default=[], metavar="NAME=r,g,b")
    ap.add_argument("--sun-rotation-z", type=float, default=0.0)
    ap.add_argument("--view-transform", default="Standard")
    ap.add_argument("--exposure", type=float, default=0.0)
    args = ap.parse_args(common.script_args())

    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()), use_scripts=False)
    scene = bpy.context.scene
    if scene.camera is None:
        raise RuntimeError("blend has no scene camera")
    device = setup_device(scene)
    w0, h0 = scene_size(scene)
    configure_cycles(scene, args.samples, max(1, round(w0 * args.scale)), max(1, round(h0 * args.scale)))
    set_colour_management(scene, args.view_transform, args.exposure)

    cam = scene.camera
    if args.cam_loc:
        cam.location = _vec(args.cam_loc)
    if args.cam_target:
        cam.rotation_mode = "XYZ"
        cam.rotation_euler = look_rotation(_vec(args.cam_target) - Vector(cam.location))
    bpy.context.view_layer.update()
    orbit = None
    if args.orbit_deg or args.dolly != 1.0:
        orbit = orbit_camera(scene, cam, args.orbit_deg, args.dolly)

    changes = []
    for name, f in _kv(args.light_scale, float):
        if name == "world":
            bg = world_background(scene)
            if bg is None:
                raise KeyError("scene has no world Background node")
            bg.inputs["Strength"].default_value *= f
            changes.append(f"world strength x{f}")
            continue
        for o in _match_lights(scene, name):
            o.data.energy *= f
            changes.append(f"{o.name} energy x{f} -> {o.data.energy:.4g}")
    for name, rgb in _kv(args.light_color, _vec):
        if name == "world":
            bg = world_background(scene)
            if bg is None:
                raise KeyError("scene has no world Background node")
            c = bg.inputs["Color"].default_value
            bg.inputs["Color"].default_value = (c[0] * rgb[0], c[1] * rgb[1], c[2] * rgb[2], 1.0)
            changes.append(f"world colour x{tuple(rgb)}")
            continue
        for o in _match_lights(scene, name):
            o.data.color = tuple(rgb)
            changes.append(f"{o.name} colour = {tuple(rgb)}")
    if args.sun_rotation_z:
        rz = Matrix.Rotation(math.radians(args.sun_rotation_z), 4, "Z")
        suns = [o for o in scene.objects if o.type == "LIGHT" and o.data.type == "SUN"]
        for o in suns:
            o.matrix_world = rz @ o.matrix_world
            changes.append(f"{o.name} rotated {args.sun_rotation_z} deg about Z")
        if not suns:
            print("[render_view] WARNING: --sun-rotation-z given but the scene has no sun")

    dt = render_to(scene, png=Path(args.out), exr=Path(args.exr) if args.exr else None)
    print(f"[render_view] {device}; {scene.render.resolution_x}x{scene.render.resolution_y}, "
          f"{args.samples} spp adaptive+OIDN, view {args.view_transform}; {dt:.1f}s; "
          f"camera {tuple(round(x, 3) for x in cam.matrix_world.translation)}")
    if orbit:
        print("[render_view]  orbit", orbit)
    for c in changes:
        print("[render_view]  ", c)
    print(f"[render_view] wrote {args.out}" + (f" and {args.exr}" if args.exr else ""))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
