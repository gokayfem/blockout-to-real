"""Render the frame sequences of a case's time-lapse video (Cycles, one Blender session).

    blender -b --factory-startup --python scripts/blender/timelapse_frames.py -- \
        --case CASE [--iter K] [--blend PATH] [--samples 64] [--scale F] \
        [--orbit-deg 6] [--orbit-frames 37] [--relight-frames 13] \
        [--only reveal,lights,orbit,relight] [--out-dir DIR]

Sequences, written to OUT/<seq>/f_NNNN.png (8-bit PNG through the blend's own view settings):

  reveal/f_0000..f_00NN  state k: the first k mesh objects keep their materials, the rest get one
                         neutral grey material; fitted lights unchanged. Order: category `structure`,
                         then other objects by visible pixels in gbuffers/iter<K>/id.png (largest
                         first; objects missing from objects.json after them), `exterior` last.
  lights/f_0001..        each fitted light alone (all others off; the world background counts as one
                         light, `world`), by energy_share_of_image descending from
                         backproject/iter<K>/lights.json. Portals and zero-strength / zero-share
                         lights are skipped.
  orbit/f_0001..         yaw(i) = -A cos(pi i / (M-1)) about world Z through the pivot P (P stays at
                         the image centre, height fixed; the middle frame is the original camera when
                         M is odd). P = point on the optical axis at the median planar depth of visible
                         non-structure, non-exterior pixels (gbuffers depth.exr + id.png); fallback:
                         centre of all mesh bounding boxes projected on the axis. A is halved (down to
                         1 deg) while structure geometry lies between P and either end camera.
  relight/f_0001..       linear from the scene as fitted to a generic "warm evening": virtual lights
                         x0.12; world x0.45 tinted 1.0,0.62,0.45; every other light x2.0 with colour
                         1.0,0.5,0.25.

frames.json in OUT records what was rendered; with --only, entries of other sequences are kept.
Fixed seed, denoising, persistent data. Nothing is saved back to the .blend; nothing from the target
is composited into any frame.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # .../scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))  # .../scripts/blender
from b2r import common  # noqa: E402
import render_view as rv  # noqa: E402

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

SEQUENCES = ("reveal", "lights", "orbit", "relight")
SEED = 0
GREY = (0.5, 0.5, 0.5, 1.0)
EVENING = {"virtual_scale": 0.12, "world_scale": 0.45, "world_tint": (1.0, 0.62, 0.45),
           "other_scale": 2.0, "other_colour": (1.0, 0.5, 0.25)}


def log(msg: str) -> None:
    print(f"[timelapse] {msg}", flush=True)


# --------------------------------------------------------------------------- case inputs
def default_iteration(case: Path) -> int:
    summ = case / "final" / "summary.json"
    if summ.is_file():
        k = common.read_json(summ).get("final_iteration")
        if k is not None:
            return int(k)
    its = sorted(int(p.name[4:]) for p in (case / "backproject").glob("iter*")
                 if p.is_dir() and p.name[4:].isdigit())
    if not its:
        raise SystemExit(f"no backproject/iter* folder in {case}")
    return its[-1]


def default_blend(case: Path, k: int) -> Path:
    bp = case / "backproject" / f"iter{k}"
    for p in (case / "final" / "scene_final.blend", bp / "scene_lit_detail.blend", bp / "scene_lit.blend"):
        if p.is_file():
            return p
    raise SystemExit(f"no scene_final.blend / scene_lit_detail.blend / scene_lit.blend for iter{k} in {case}")


def read_map(path: Path):
    import OpenImageIO as oiio
    import numpy as np
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"cannot read {path}: {oiio.geterror()}")
    spec = inp.spec()
    fmt = oiio.FLOAT if path.suffix.lower() == ".exr" else oiio.UINT16
    arr = inp.read_image(fmt)
    inp.close()
    arr = np.asarray(arr).reshape(spec.height, spec.width, spec.nchannels)
    return arr[..., 0]


def object_categories(case: Path, k: int) -> dict[str, str]:
    p = case / "blockout" / f"iter{k}" / "objects.json"
    if not p.is_file():
        log(f"WARNING no {p}: every object treated as uncategorised")
        return {}
    return {o["name"]: str(o.get("category", "")) for o in common.read_json(p)}


def visible_pixels(case: Path, k: int, meshes) -> tuple[dict[str, int], object]:
    """name -> visible pixel count from id.png (and the id array, or None if missing)."""
    p = case / "gbuffers" / f"iter{k}" / "id.png"
    if not p.is_file():
        log(f"WARNING no {p}: reveal order by name only, pivot from bounding boxes")
        return {}, None
    import numpy as np
    ids = read_map(p).astype(np.int64)
    counts = np.bincount(ids.ravel())
    out = {}
    for o in meshes:
        i = int(o.pass_index)
        out[o.name] = int(counts[i]) if 0 < i < len(counts) else 0
    return out, ids


# --------------------------------------------------------------------------- scene state
def mesh_objects(scene):
    return [o for o in scene.objects if o.type == "MESH" and not o.hide_render]


def light_objects(scene):
    return [o for o in scene.objects if o.type == "LIGHT"]


def is_portal(o) -> bool:
    return o.data.type == "AREA" and bool(getattr(o.data, "is_portal", False)) or \
        bool(getattr(getattr(o.data, "cycles", None), "is_portal", False))


class LightState:
    """Remembers the fitted lights and world, and sets per-light (scale, colour) absolutely."""

    def __init__(self, scene):
        self.bg = rv.world_background(scene)
        self.base = {o.name: (o.data.energy, tuple(o.data.color)) for o in light_objects(scene)
                     if not is_portal(o)}
        self.world = None
        if self.bg is not None:
            self.world = (self.bg.inputs["Strength"].default_value, tuple(self.bg.inputs["Color"].default_value))

    def apply(self, state: dict) -> None:
        """state: name -> (scale, colour or None); world colour is a multiplier. Missing = as fitted."""
        for n, (e, c) in self.base.items():
            s, col = state.get(n, (1.0, None))
            o = bpy.data.objects[n]
            o.data.energy = e * s
            o.data.color = tuple(col) if col is not None else c
        if self.world is not None:
            s, mul = state.get("world", (1.0, None))
            st, c = self.world
            mul = mul or (1.0, 1.0, 1.0)
            self.bg.inputs["Strength"].default_value = st * s
            self.bg.inputs["Color"].default_value = (c[0] * mul[0], c[1] * mul[1], c[2] * mul[2], c[3])


# --------------------------------------------------------------------------- sequences
def seq_reveal(scene, out: Path, cats: dict, vis: dict) -> dict:
    meshes = mesh_objects(scene)
    names = [o.name for o in meshes]
    struct = [n for n in names if cats.get(n) == "structure"]
    ext = [n for n in names if cats.get(n) == "exterior"]
    known = [n for n in names if n in cats and n not in struct and n not in ext]
    unknown = [n for n in names if n not in cats]

    def by_pixels(lst):
        return sorted(lst, key=lambda n: (-vis.get(n, 0), n))

    order = by_pixels(struct) + by_pixels(known) + by_pixels(unknown) + by_pixels(ext)
    grey = bpy.data.materials.new("timelapse_neutral_grey")
    grey.use_nodes = True
    bsdf = next(n for n in grey.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = GREY
    bsdf.inputs["Roughness"].default_value = 0.8
    bsdf.inputs["Metallic"].default_value = 0.0
    orig = {}
    for n in order:
        o = bpy.data.objects[n]
        orig[n] = [s.material for s in o.material_slots]
        if not o.material_slots:
            o.data.materials.append(None)
            orig[n] = [None]
    times = []
    for k in range(len(order) + 1):
        real = set(order[:k])
        for n in order:
            o = bpy.data.objects[n]
            for slot, m in zip(o.material_slots, orig[n]):
                want = m if n in real else grey
                if slot.material is not want:
                    slot.material = want
        bpy.context.view_layer.update()
        times.append(rv.render_to(scene, png=out / "reveal" / f"f_{k:04d}.png"))
        log(f"reveal {k}/{len(order)} ({order[k - 1] if k else 'all grey'}) {times[-1]:.1f}s")
    for n in order:  # restore for later sequences in this session
        for slot, m in zip(bpy.data.objects[n].material_slots, orig[n]):
            slot.material = m
    return {"count": len(order) + 1, "order": order, "seconds_per_frame": round(statistics.mean(times), 2)}


def seq_lights(scene, out: Path, ls: LightState, lights_json: dict | None) -> dict:
    entries = []
    if lights_json:
        for L in lights_json.get("lights", []):
            entries.append({"name": L["name"], "virtual": bool(L.get("virtual", False)),
                            "energy_share": float(L.get("energy_share_of_image", L.get("energy_share", 0.0)) or 0.0),
                            "strength": float(L.get("strength", 0.0) or 0.0)})
    else:
        log("WARNING no lights.json: lights ordered by blend energy, share unknown")
        for n, (e, _c) in ls.base.items():
            entries.append({"name": n, "virtual": n.startswith("virtual"), "energy_share": None, "strength": e})
    order, skipped = [], []
    for e in entries:
        n = e["name"]
        exists = (n == "world" and ls.world is not None) or n in ls.base
        if not exists:
            skipped.append({"name": n, "why": "not a (non-portal) light in the blend"})
            continue
        cur = ls.world[0] if n == "world" else ls.base[n][0]
        if e["strength"] <= 0 or cur <= 0 or (e["energy_share"] is not None and e["energy_share"] <= 0):
            skipped.append({"name": n, "why": "zero strength or share"})
            continue
        order.append(e)
    order.sort(key=lambda e: -(e["energy_share"] if e["energy_share"] is not None else e["strength"]))
    off = {n: (0.0, None) for n in ls.base}
    off["world"] = (0.0, None)
    times = []
    for i, e in enumerate(order, 1):
        st = dict(off)
        st[e["name"]] = (1.0, None)
        ls.apply(st)
        times.append(rv.render_to(scene, png=out / "lights" / f"f_{i:04d}.png"))
        log(f"lights {i}/{len(order)} {e['name']} alone {times[-1]:.1f}s")
    ls.apply({})
    return {"count": len(order),
            "order": [{"name": e["name"], "virtual": e["virtual"], "energy_share": e["energy_share"]} for e in order],
            "skipped": skipped, "seconds_per_frame": round(statistics.mean(times), 2) if times else None}


def orbit_pivot(scene, cam, case: Path, k: int, cats: dict, ids):
    M0 = cam.matrix_world.copy()
    C = M0.translation.copy()
    fwd = (M0.to_3x3() @ Vector((0, 0, -1))).normalized()
    dep_p = case / "gbuffers" / f"iter{k}" / "depth.exr"
    if ids is not None and dep_p.is_file():
        import numpy as np
        depth = read_map(dep_p)
        if depth.shape == ids.shape:
            keep = [o.pass_index for o in mesh_objects(scene)
                    if cats.get(o.name) not in ("structure", "exterior") and o.pass_index > 0]
            m = np.isin(ids, keep) & (depth > 0) & np.isfinite(depth)
            if m.any():
                d = float(np.median(depth[m]))
                return C + fwd * d, d, f"median planar depth of {int(m.sum())} non-structure, non-exterior pixels"
        else:
            log(f"WARNING depth {depth.shape} and id {ids.shape} differ: bounding-box pivot")
    pts = [o.matrix_world @ Vector(c) for o in mesh_objects(scene) for c in o.bound_box]
    centre = sum(pts, Vector()) / len(pts)
    d = (centre - C).dot(fwd)
    return C + fwd * d, d, "centre of mesh bounding boxes projected on the optical axis (fallback)"


def orbit_matrix(M0, P, yaw_deg: float):
    return Matrix.Translation(P) @ Matrix.Rotation(math.radians(yaw_deg), 4, "Z") @ Matrix.Translation(-P) @ M0


def structure_blocks(scene, P, target, cats) -> str | None:
    dg = bpy.context.evaluated_depsgraph_get()
    v = target - P
    span = v.length
    origin, d = P.copy(), v.normalized()
    travelled = 0.0
    for _ in range(64):  # step through non-structure hits
        hit, loc, _n, _i, ob, _m = scene.ray_cast(dg, origin, d, distance=span - travelled)
        if not hit:
            return None
        name = ob.original.name if ob is not None else "?"
        if cats.get(name) == "structure":
            return name
        step = (loc - origin).length + 1e-4
        travelled += step
        origin = loc + d * 1e-4
        if travelled >= span:
            return None
    return None


def seq_orbit(scene, out: Path, case: Path, k: int, cats: dict, ids, amp0: float, m: int) -> dict:
    cam = scene.camera
    M0 = cam.matrix_world.copy()
    P, dist, method = orbit_pivot(scene, cam, case, k, cats, ids)
    amp, tries = float(amp0), []
    while True:
        blocks = [structure_blocks(scene, P, orbit_matrix(M0, P, s * amp).translation, cats) for s in (-1, 1)]
        tries.append({"amplitude_deg": amp, "blocked_by": blocks})
        if not any(blocks) or amp <= 1.0:
            break
        amp = max(1.0, amp / 2.0)
    if any(blocks):
        log(f"WARNING structure still between pivot and camera at A={amp}: {blocks}")
    times = []
    mid = (m - 1) / 2.0
    for i in range(m):
        if m % 2 == 1 and i == int(mid):
            cam.matrix_world = M0
            y = 0.0
        else:
            y = -amp * math.cos(math.pi * i / max(1, m - 1))
            cam.matrix_world = orbit_matrix(M0, P, y)
        bpy.context.view_layer.update()
        times.append(rv.render_to(scene, png=out / "orbit" / f"f_{i + 1:04d}.png"))
        log(f"orbit {i + 1}/{m} yaw {y:+.3f} {times[-1]:.1f}s")
    cam.matrix_world = M0
    bpy.context.view_layer.update()
    return {"count": m, "amplitude_deg": amp, "pivot": [round(x, 4) for x in P],
            "pivot_depth_m": round(dist, 4), "pivot_method": method, "amplitude_tries": tries,
            "seconds_per_frame": round(statistics.mean(times), 2)}


def seq_relight(scene, out: Path, ls: LightState, virtual: set, n: int) -> dict:
    E = EVENING
    times = []
    for f in range(1, n + 1):
        t = (f - 1) / max(1, n - 1)
        st = {}
        for name, (_e, c) in ls.base.items():
            if name in virtual:
                st[name] = (1 + t * (E["virtual_scale"] - 1), None)
            else:
                col = tuple(c[j] + t * (E["other_colour"][j] - c[j]) for j in range(3))
                st[name] = (1 + t * (E["other_scale"] - 1), col)
        st["world"] = (1 + t * (E["world_scale"] - 1), tuple(1 + t * (E["world_tint"][j] - 1) for j in range(3)))
        ls.apply(st)
        times.append(rv.render_to(scene, png=out / "relight" / f"f_{f:04d}.png"))
        log(f"relight {f}/{n} t={t:.3f} {times[-1]:.1f}s")
    ls.apply({})
    return {"count": n, "look": "warm evening", "end_state": E,
            "seconds_per_frame": round(statistics.mean(times), 2)}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--blend", default=None)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--orbit-deg", type=float, default=6.0)
    ap.add_argument("--orbit-frames", type=int, default=37)
    ap.add_argument("--relight-frames", type=int, default=13)
    ap.add_argument("--only", default=",".join(SEQUENCES))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(common.script_args())

    case = common.case_dir(a.case)
    only = [s.strip() for s in a.only.split(",") if s.strip()]
    bad = [s for s in only if s not in SEQUENCES]
    if bad:
        raise SystemExit(f"unknown sequence(s) {bad}; choose from {SEQUENCES}")
    k = a.iter if a.iter is not None else default_iteration(case)
    blend = Path(a.blend).resolve() if a.blend else default_blend(case, k)
    out = Path(a.out_dir).resolve() if a.out_dir else case / "timelapse" / "frames"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    bpy.ops.wm.open_mainfile(filepath=str(blend), use_scripts=False)
    scene = bpy.context.scene
    if scene.camera is None:
        raise RuntimeError("blend has no scene camera")
    device = rv.setup_device(scene)
    w0, h0 = rv.scene_size(scene)
    scale = a.scale if a.scale is not None else min(1.0, 1024.0 / w0)
    w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    rv.configure_cycles(scene, a.samples, w, h)  # view settings stay the blend's own
    scene.cycles.seed = SEED
    scene.cycles.use_animated_seed = False
    scene.render.use_persistent_data = True
    log(f"case {case.name} iter{k} blend {blend.name} {w}x{h} {a.samples} spp, {device}, "
        f"view {scene.view_settings.view_transform}")

    cats = object_categories(case, k)
    vis, ids = visible_pixels(case, k, mesh_objects(scene))
    lj_path = case / "backproject" / f"iter{k}" / "lights.json"
    lights_json = common.read_json(lj_path) if lj_path.is_file() else None
    virtual = {L["name"] for L in (lights_json or {}).get("lights", []) if L.get("virtual")}
    ls = LightState(scene)
    virtual |= {n for n in ls.base if n.startswith("virtual") and lights_json is None}

    fj = out / "frames.json"
    frames = common.read_json(fj) if fj.is_file() else {}
    frames.update({"case": case.name, "iteration": k, "blend": str(blend), "size": [w, h],
                   "samples": a.samples, "device": device,
                   "view_transform": scene.view_settings.view_transform})
    for s in only:
        if s == "reveal":
            frames["reveal"] = seq_reveal(scene, out, cats, vis)
        elif s == "lights":
            frames["lights"] = seq_lights(scene, out, ls, lights_json)
        elif s == "orbit":
            frames["orbit"] = seq_orbit(scene, out, case, k, cats, ids, a.orbit_deg, a.orbit_frames)
        elif s == "relight":
            frames["relight"] = seq_relight(scene, out, ls, virtual, a.relight_frames)
        common.write_json(fj, frames)
    frames["seconds_total"] = round(time.time() - t0, 1)
    common.write_json(fj, frames)
    log(f"done in {frames['seconds_total']}s -> {out}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
