"""relight: replace the scene's lights by a light rig fitted against the target, with Cycles as
the forward model.

    blender -b --factory-startup --python scripts/blender/relight.py -- --case CASE --iter K \
        [--basis-scale 0.5] [--basis-samples 256] [--sun-samples 96] [--final-samples 512] \
        [--reuse-basis DIR] [--prune-margin 0.01] [--prune-total-cap 0.03] \
        [--out-dir DIR] [--rerender-out PATH.png] [--rerender-scale 1.0] [--no-rerender]

Reads   backproject/iterK/scene.blend, decompose/iterK/lighting.json, target/iterK/target.png,
        gbuffers/iterK/{id.png, position.exr, normal.exr, camera.json}, blockout/iterK/objects.json
Writes  (--out-dir, default backproject/iterK/) scene_lit.blend, lights.json, fit.json,
        fit_preview.png, basis/, relight_log.txt; residual/iterK/rerender.png + .exr
        (or --rerender-out)

Candidates, all derived from the scene and the maps (no object names):
  world        the world background (an existing one is kept and tinted, else a uniform sky)
  openings     only if the scene is enclosed (rays from inside the visible scene mostly hit
               geometry): one area light per connected open-sky region of id.png (id 0, together
               with any `exterior` objects seen through it), placed on the plane fitted to the 3D
               positions of the geometry around the region, sized to the camera rays through the
               region, facing into the scene; plus a Cycles portal (emits nothing) at each
  suns         a grid of directions over the upper hemisphere, kept only where a ray test shows
               the sun reaching visible surfaces (through the openings when enclosed). Group
               "interior" (lights non-exterior surfaces) and, when exterior geometry is visible,
               "exterior" (lights only exterior surfaces). The solver keeps at most one per group
  emitters     one point light per on-object emitter in lighting.json
  input lights every light already in the scene (custom property b2r_input_light, or simply
               present before this stage), keeping its name, position, shape and type
  virtual      overhead fill, fill behind the camera, bounce uplight; positioned from the
               enclosure found by ray casting from the camera and the visible-scene centre, or
               from the visible geometry's bounding box for an open scene. virtual=True, hidden
               from camera and glossy rays
Each candidate is rendered alone at unit white strength (basis EXR); b2r.relight_solve fits
non-negative RGB weights, keeps one sun per group, prunes (world exempt), clamps tints. Weights
become native Blender units; zero-weight candidates are deleted.

The map analysis (connected regions, plane fits) runs in the user's Python: Blender re-invokes this
file with --analyze through common.python_exe().
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

INPUT_LIGHT_PROP = "b2r_input_light"
UNIT = {"world": 1.0, "area": 100.0, "sun": 3.0, "point": 50.0, "spot": 50.0}
UNIT_LABEL = {"world": "background strength (radiance multiplier)", "area": "W (radiant power)",
              "sun": "W/m^2 (irradiance)", "point": "W (radiant power)", "spot": "W (radiant power)"}
SUN_ANGLE_DEG = 0.53
SUN_ELEV = (5, 15, 25, 35, 45, 55, 65, 75, 85)
SUN_AZ_STEP = 15
MAX_SUNS_INTERIOR = 16
MAX_SUNS_EXTERIOR = 6
SUN_LIT_REL = 0.10        # keep a direction if it lights >= this fraction of the best direction's lit share
SUN_LIT_MIN = 0.003       # ... and at least this fraction of the sample points
EXT_ONLY_MAX_INTERIOR = 0.002
ENCLOSED_HIT_FRACTION = 0.8
N_SAMPLES = 2500
MIN_OPENING_FRAC = 5e-4   # open-sky region must cover this fraction of the image
PLANARITY_MAX = 0.15      # ring-point RMS off the fitted plane / opening size


# =========================================================================== map analysis (user's Python)
def analyze_main(job_path: Path) -> None:
    import cv2
    import numpy as np
    from scipy import ndimage
    job = json.loads(job_path.read_text())
    gb = Path(job["gbuffers"])
    W, H = int(job["width"]), int(job["height"])

    def rd(p):
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None or a.shape[:2] != (H, W):
            raise ValueError(f"{p} missing or not {W}x{H}")
        return a

    ids = rd(gb / "id.png").astype(np.int64)
    ids = ids[..., 0] if ids.ndim == 3 else ids
    pos = rd(gb / "position.exr")[..., :3][..., ::-1].astype(np.float64)
    nrm = rd(gb / "normal.exr")[..., :3][..., ::-1].astype(np.float64)
    cam = json.loads((gb / "camera.json").read_text())
    c2w = np.array(cam["cam_to_world"], np.float64)
    cpos = c2w[:3, 3]
    ext_ids = [int(k) for k, v in job["categories"].items() if v == "exterior"]
    is_ext = np.isin(ids, ext_ids)
    sky = ids == 0
    interior = (ids > 0) & ~is_ext
    out = {"sky_fraction": float(sky.mean()), "exterior_fraction": float(is_ext.mean()), "openings": []}

    # visible geometry
    P = pos[interior]
    if len(P) == 0:
        raise RuntimeError("no visible non-exterior geometry")
    out["visible_centre"] = np.median(P, 0).tolist()
    out["visible_bbox_p2_p98"] = [np.percentile(P, 2, 0).tolist(), np.percentile(P, 98, 0).tolist()]

    # sample points for the sun ray test
    for key, m in (("samples_interior", interior), ("samples_exterior", is_ext)):
        jj, ii = np.nonzero(m)
        if len(jj) > N_SAMPLES:
            sel = np.random.default_rng(0).choice(len(jj), N_SAMPLES, replace=False)
            jj, ii = jj[sel], ii[sel]
        out[key] = {"p": pos[jj, ii].round(5).tolist(), "n": nrm[jj, ii].round(5).tolist()}

    # openings: connected open-sky regions (with exterior objects seen through them)
    lab, n = ndimage.label(sky | is_ext, structure=np.ones((3, 3)))
    min_px = max(16, int(MIN_OPENING_FRAC * W * H))
    r = max(2, int(round(0.004 * math.hypot(W, H))))
    R = c2w[:3, :3]
    for l in range(1, n + 1):
        comp = lab == l
        nsky = int((comp & sky).sum())
        if nsky < min_px:
            continue
        rec = {"label": l, "sky_pixels": nsky, "pixels": int(comp.sum())}
        ys, xs = np.nonzero(comp)
        rec["bbox_xywh"] = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        border = int(comp[0].sum() + comp[-1].sum() + comp[:, 0].sum() + comp[:, -1].sum())
        rec["image_border_fraction"] = border / max(1, 2 * (rec["bbox_xywh"][2] + rec["bbox_xywh"][3]))
        ring = ndimage.binary_dilation(comp, iterations=r) & ~comp & interior
        pts = pos[ring]
        if len(pts) < 20:
            rec["skip"] = "too few geometry pixels around the region"
            out["openings"].append(rec)
            continue
        keep = np.ones(len(pts), bool)
        for _ in range(4):  # trimmed PCA plane fit
            c = pts[keep].mean(0)
            _, _, vt = np.linalg.svd(pts[keep] - c, full_matrices=False)
            nn = vt[2]
            res = np.abs((pts - c) @ nn)
            keep = res <= max(np.percentile(res, 80), 1e-6)
        if np.dot(nn, cpos - c) < 0:
            nn = -nn  # face into the scene (towards the camera)
        u = (xs + 0.5 - cam["cx"]) / cam["fx"]
        v = -(ys + 0.5 - cam["cy"]) / cam["fy"]
        d = np.stack([u, v, -np.ones_like(u)], 1) @ R.T
        den = d @ nn
        ok = np.abs(den) > 1e-6
        t = np.where(ok, ((c - cpos) @ nn) / np.where(ok, den, 1), -1)
        q = cpos + d[t > 0] * t[t > 0, None]
        if len(q) < 10:
            rec["skip"] = "region rays do not meet the fitted plane"
            out["openings"].append(rec)
            continue
        up = np.array([0.0, 0.0, 1.0])
        a2 = up - (up @ nn) * nn
        if np.linalg.norm(a2) < 0.3:  # opening in a floor / ceiling: use the in-plane PCA axis
            _, _, vq = np.linalg.svd(q - q.mean(0), full_matrices=False)
            a2 = vq[1] - (vq[1] @ nn) * nn
        a2 /= np.linalg.norm(a2)
        a1 = np.cross(a2, nn)
        s1, s2 = (q - c) @ a1, (q - c) @ a2
        lo1, hi1, lo2, hi2 = s1.min(), s1.max(), s2.min(), s2.max()
        size = [float(hi1 - lo1), float(hi2 - lo2)]
        rms = float(np.sqrt(np.mean(res[keep] ** 2)))
        rec.update({"centre": (c + a1 * (lo1 + hi1) / 2 + a2 * (lo2 + hi2) / 2).tolist(),
                    "inward_normal": nn.tolist(), "axis_x": a1.tolist(), "axis_y": a2.tolist(),
                    "size_m": size, "ring_points": int(len(pts)), "ring_rms_m": rms,
                    "planarity": rms / max(max(size), 1e-6)})
        out["openings"].append(rec)
    (job_path.parent / "analysis.json").write_text(json.dumps(out))


# =========================================================================== Blender
def blender_main() -> None:
    import bpy
    from mathutils import Matrix, Vector
    import render_view as rv

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--basis-scale", type=float, default=0.5, help="basis render scale, 1/integer")
    ap.add_argument("--basis-samples", type=int, default=256)
    ap.add_argument("--sun-samples", type=int, default=96)
    ap.add_argument("--final-samples", type=int, default=512)
    ap.add_argument("--reuse-basis", default=None, metavar="DIR")
    ap.add_argument("--prune-margin", type=float, default=0.01)
    ap.add_argument("--prune-total-cap", type=float, default=0.03)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rerender-out", default=None)
    ap.add_argument("--rerender-scale", type=float, default=1.0)
    ap.add_argument("--no-rerender", action="store_true")
    args = ap.parse_args(common.script_args())
    T0 = time.time()
    case, k = args.case, args.iter
    W, H = common.render_size(case)
    blk = int(round(1.0 / args.basis_scale))
    if blk < 1 or abs(1.0 / blk - args.basis_scale) > 1e-6:
        raise ValueError("--basis-scale must be 1/integer (1, 0.5, 0.25 ...)")
    bw, bh = W // blk, H // blk

    src_bp = common.stage_dir(case, "backproject", k, create=False)
    bp = Path(args.out_dir).resolve() if args.out_dir else src_bp
    bp.mkdir(parents=True, exist_ok=True)
    dec = common.stage_dir(case, "decompose", k, create=False)
    gb = common.stage_dir(case, "gbuffers", k, create=False)
    tgt = common.stage_dir(case, "target", k, create=False) / "target.png"
    objs_json = common.stage_dir(case, "blockout", k, create=False) / "objects.json"
    for p in (src_bp / "scene.blend", tgt, gb / "id.png", gb / "position.exr", gb / "normal.exr", gb / "camera.json"):
        if not p.is_file():
            raise FileNotFoundError(p)
    basis_dir = bp / "basis"
    basis_dir.mkdir(exist_ok=True)
    log_lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print("[relight]", s, flush=True)
        log_lines.append(s)

    lighting = common.read_json(dec / "lighting.json") if (dec / "lighting.json").is_file() else {}
    if not lighting:
        log("no decompose lighting.json: no on-object emitter candidates")
    bpy.ops.wm.open_mainfile(filepath=str(src_bp / "scene.blend"))
    scene = bpy.context.scene
    log("device", rv.setup_device(scene))
    cats = {}
    if objs_json.is_file():
        byname = {o["name"]: o.get("category", "") for o in common.read_json(objs_json)}
    else:
        byname = {}
    objects_meta = {}
    for o in scene.objects:
        if o.type == "MESH" and o.pass_index > 0:
            cats[str(o.pass_index)] = byname.get(o.name, o.get("category", ""))
            objects_meta[str(o.pass_index)] = {"name": o.name, "category": cats[str(o.pass_index)]}

    # ------------------------------------------------------------ map analysis
    job = {"gbuffers": str(gb), "width": W, "height": H, "categories": cats}
    jp = basis_dir / "analysis_job.json"
    jp.write_text(json.dumps(job))
    env = dict(os.environ, OPENCV_IO_ENABLE_OPENEXR="1")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([common.python_exe(), str(Path(__file__).resolve()), "--analyze", "--job", str(jp)], env=env)
    if r.returncode != 0:
        raise RuntimeError("map analysis failed")
    ana = json.loads((basis_dir / "analysis.json").read_text())

    dg = bpy.context.evaluated_depsgraph_get()
    cam = scene.camera
    cam_loc = cam.matrix_world.translation.copy()
    fwd = (cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
    centre = Vector(ana["visible_centre"])
    bmin, bmax = Vector(ana["visible_bbox_p2_p98"][0]), Vector(ana["visible_bbox_p2_p98"][1])
    diag = max((bmax - bmin).length, (centre - cam_loc).length, 1e-3)
    far = 100.0 * diag

    def cast(o, d, dist=far):
        hit, loc, _n, _i, _ob, _m = scene.ray_cast(dg, o, d, distance=dist)
        return loc if hit else None

    # probe point in free space between camera and visible centre
    to_c = centre - cam_loc
    h = cast(cam_loc, to_c.normalized(), to_c.length)
    probe = cam_loc + to_c * 0.5 if h is None else cam_loc + (h - cam_loc) * 0.5

    def fib(n):
        g = math.pi * (3 - math.sqrt(5))
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            rr = math.sqrt(1 - z * z)
            yield Vector((rr * math.cos(g * i), rr * math.sin(g * i), z))

    dirs = list(fib(256))
    hit_probe = sum(cast(probe, d) is not None for d in dirs) / len(dirs)
    hit_cam = sum(cast(cam_loc, d) is not None for d in dirs) / len(dirs)
    enclosed = (hit_probe + hit_cam) / 2 >= ENCLOSED_HIT_FRACTION
    gen = {"enclosure_rule": f"enclosed if the mean fraction of 256 sphere rays from the camera and from a free-space "
                             f"probe (halfway to the visible-scene centre) that hit geometry is >= "
                             f"{ENCLOSED_HIT_FRACTION}",
           "hit_fraction_camera": round(hit_cam, 3), "hit_fraction_probe": round(hit_probe, 3),
           "probe": list(probe), "enclosed": enclosed, "sky_fraction": ana["sky_fraction"]}
    log(f"enclosure: camera {hit_cam:.2f}, probe {hit_probe:.2f} -> {'enclosed' if enclosed else 'open'}; "
        f"sky {ana['sky_fraction']:.3f}")

    # ------------------------------------------------------------ input lights and world
    # Placeholder lights that the blockout / ingest stage added only to make the grey preview
    # readable (tagged b2r_preview_light) are not part of the scene: remove them, they are never
    # fit candidates. Every other light was in the input scene and is a candidate.
    input_lights = []
    removed_preview = []
    for o in list(scene.objects):
        if o.type != "LIGHT":
            continue
        if getattr(o.data.cycles, "is_portal", False):
            continue
        if o.get("b2r_preview_light"):
            removed_preview.append(o.name)
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        input_lights.append(o)
    log("removed placeholder preview lights:", removed_preview)
    bg = rv.world_background(scene)
    world_note = None
    if bg is None:
        world = bpy.data.worlds.new("world_sky")
        try:
            world.use_nodes = True
        except Exception:
            pass
        scene.world = world
        bg = rv.world_background(scene)
        world_note = "no world background in the input: uniform sky created"
    elif bg.inputs["Color"].is_linked:
        nt = scene.world.node_tree
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.name = mix.label = "b2r_world_tint"
        mix.inputs["Factor"].default_value = 1.0
        src = bg.inputs["Color"].links[0].from_socket
        nt.links.new(src, mix.inputs["A"])
        nt.links.new(mix.outputs["Result"], bg.inputs["Color"])
        world_note = "input world colour is textured: kept, fitted tint applied through b2r_world_tint"
    else:
        world_note = f"input world '{scene.world.name}' kept, colour set to white for the basis"
        bg.inputs["Color"].default_value = (1, 1, 1, 1)
    tint_node = scene.world.node_tree.nodes.get("b2r_world_tint")

    def set_world_rgb(rgb):
        if tint_node is not None:
            tint_node.inputs["B"].default_value = (*rgb, 1.0)
        else:
            bg.inputs["Color"].default_value = (*rgb, 1.0)

    set_world_rgb((1, 1, 1))

    def new_light(name, kind, loc, forward=None, size=None, virtual=False, desc=""):
        ld = bpy.data.lights.new(name, kind)
        ob = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(ob)
        ob.location = loc
        if forward is not None:
            ob.rotation_mode = "XYZ"
            ob.rotation_euler = rv.look_rotation(Vector(forward))
        if kind == "AREA" and size is not None:
            ld.shape = "RECTANGLE"
            ld.size, ld.size_y = max(size[0], 1e-3), max(size[1], 1e-3)
        ld.color = (1.0, 1.0, 1.0)
        ob["virtual"] = bool(virtual)
        ob["b2r_candidate"] = desc
        if kind != "SUN":
            ob.visible_camera = False
        if virtual:
            ob.visible_glossy = False
        return ob

    cands = [{"name": "world", "kind": "world", "virtual": False, "obj": None, "input": False,
              "desc": "world background (sky); " + world_note}]
    portals = []

    # ------------------------------------------------------------ openings
    gen["openings"] = []
    if enclosed:
        for i, op in enumerate(ana["openings"]):
            rec = {k_: op[k_] for k_ in ("sky_pixels", "bbox_xywh", "image_border_fraction") if k_ in op}
            if "skip" in op:
                rec["skipped"] = op["skip"]
            elif op["planarity"] > PLANARITY_MAX:
                rec["skipped"] = f"surrounding geometry not planar ({op['planarity']:.3f} > {PLANARITY_MAX})"
            else:
                c = Vector(op["centre"])
                nin = Vector(op["inward_normal"])
                ax, ay = Vector(op["axis_x"]), Vector(op["axis_y"])
                size = op["size_m"]

                def orient(ob):
                    # local X = axis_x, local Y = axis_y, local -Z = inward
                    z = -nin
                    y = ay
                    x = y.cross(z)
                    ob.matrix_world = Matrix.Translation(ob.location) @ Matrix((x, y, z)).transposed().to_4x4()

                ob = new_light(f"opening_{i}_light", "AREA", c, size=size,
                               desc="area light filling an open-sky region, facing into the scene")
                orient(ob)
                pt = new_light(f"opening_{i}_portal", "AREA", c - nin * 0.01 * max(size), size=size,
                               desc="Cycles light portal (emits nothing)")
                orient(pt)
                pt.data.cycles.is_portal = True
                portals.append(pt)
                cands.append({"name": ob.name, "kind": "area", "virtual": False, "obj": ob, "input": False,
                              "desc": "area light filling an open-sky region of the view, facing into the scene"})
                rec.update(light=ob.name, centre=[round(x, 3) for x in c], size_m=[round(x, 3) for x in size],
                           inward_normal=[round(x, 3) for x in nin], planarity=round(op["planarity"], 4))
            gen["openings"].append(rec)
    else:
        gen["openings_note"] = "open scene: no portal lights"
    log("openings:", json.dumps(gen["openings"]))

    # ------------------------------------------------------------ suns
    def lit_fraction(samples, L):
        if not samples["p"]:
            return 0.0
        eps = 1e-3 * diag + 1e-4
        lit = 0
        for p, nn in zip(samples["p"], samples["n"]):
            nv = Vector(nn)
            if nv.dot(L) <= 0.02:
                continue
            if cast(Vector(p) + nv * eps, L) is None:
                lit += 1
        return lit / len(samples["p"])

    t_s = time.time()
    tests = []
    for el in SUN_ELEV:
        for az in range(0, 360, SUN_AZ_STEP):
            e, a = math.radians(el), math.radians(az)
            L = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
            tests.append({"el": el, "az": az, "L": L, "fi": lit_fraction(ana["samples_interior"], L),
                          "fe": lit_fraction(ana["samples_exterior"], L)})
    log(f"sun ray test: {len(tests)} directions in {time.time() - t_s:.1f}s")

    def fps(pool, n, key):
        """farthest-point (angular) selection seeded with the best direction"""
        if not pool:
            return []
        chosen = [max(pool, key=key)]
        while len(chosen) < min(n, len(pool)):
            rest = [t for t in pool if t not in chosen]
            chosen.append(max(rest, key=lambda t: min(t["L"].angle(c["L"]) for c in chosen)))
        return chosen

    bi = max(t["fi"] for t in tests)
    pool_i = [t for t in tests if t["fi"] >= max(SUN_LIT_MIN, SUN_LIT_REL * bi)]
    groups = {("interior" if enclosed else "open"): fps(pool_i, MAX_SUNS_INTERIOR, lambda t: t["fi"])}
    if ana["samples_exterior"]["p"]:
        be = max(t["fe"] for t in tests)
        pool_e = [t for t in tests if t["fi"] <= EXT_ONLY_MAX_INTERIOR and t["fe"] >= max(SUN_LIT_MIN, SUN_LIT_REL * be)]
        groups["exterior"] = fps(pool_e, MAX_SUNS_EXTERIOR, lambda t: t["fe"])
    gen["suns"] = {"rule": f"directions on a grid (elevation {list(SUN_ELEV)} deg, azimuth every {SUN_AZ_STEP} deg "
                           f"about world Z from +X); a direction lights a sample point if n.L > 0.02 and a ray to the "
                           f"sun escapes; interior pool = directions lighting >= max({SUN_LIT_MIN}, {SUN_LIT_REL} x best) "
                           f"of non-exterior samples; exterior pool = lighting <= {EXT_ONLY_MAX_INTERIOR} of interior and "
                           f">= max({SUN_LIT_MIN}, {SUN_LIT_REL} x best) of exterior samples; farthest-point selection",
                   "directions_tested": len(tests), "best_interior_lit_fraction": round(bi, 4),
                   "selected": {g: len(v) for g, v in groups.items()}}
    sun_parent = Vector(((bmin.x + bmax.x) / 2, (bmin.y + bmax.y) / 2, bmax.z + 0.25 * diag))
    for g, sel in groups.items():
        for t in sel:
            nm = f"sun_{'exterior_' if g == 'exterior' else ''}e{t['el']:02d}_a{t['az']:03d}"
            ob = new_light(nm, "SUN", sun_parent, forward=-t["L"], desc=f"sun candidate ({g})")
            ob.data.angle = math.radians(SUN_ANGLE_DEG)
            cands.append({"name": ob.name, "kind": "sun", "virtual": False, "obj": ob, "input": False,
                          "sun_group": g, "elevation_deg": t["el"], "azimuth_deg_from_x": t["az"],
                          "direction_towards_sun": list(t["L"]), "lit_fraction_interior": round(t["fi"], 4),
                          "lit_fraction_exterior": round(t["fe"], 4), "desc": f"sun ({g} group)"})
    log("suns:", gen["suns"]["selected"])

    # ------------------------------------------------------------ on-object emitters
    for j, em in enumerate(lighting.get("emitters", [])):
        if int(em.get("on_object_id", 0) or 0) > 0 and em.get("position_world"):
            p = Vector(em["position_world"])
            p = p + (cam_loc - p).normalized() * 0.01 * diag
            ob = new_light(f"emitter_{em['on_object_id']}_{j}", "POINT", p, desc="on-object emitter")
            ob.data.shadow_soft_size = 0.01 * diag
            cands.append({"name": ob.name, "kind": "point", "virtual": False, "obj": ob, "input": False,
                          "desc": f"emitter on object id {em['on_object_id']} ({em.get('kind')})"})

    # ------------------------------------------------------------ input lights
    for o in input_lights:
        kind = o.data.type.lower()
        cands.append({"name": o.name, "kind": kind, "virtual": False, "obj": o, "input": True,
                      "sun_group": f"input:{o.name}" if kind == "sun" else None,
                      "was_hidden": bool(o.hide_render), "had_property": INPUT_LIGHT_PROP in o.keys(),
                      "original_energy": float(o.data.energy), "original_color": list(o.data.color),
                      "desc": "light from the input scene (kept name, position, type)"})
        o.data.color = (1, 1, 1)
    log("input lights:", [o.name for o in input_lights])

    # ------------------------------------------------------------ virtual fills
    right = (cam.matrix_world.to_3x3() @ Vector((1, 0, 0)))
    fh = Vector((fwd.x, fwd.y, 0))
    fh = fh.normalized() if fh.length > 1e-6 else Vector((0, 1, 0))
    rh = Vector((right.x, right.y, 0))
    rh = rh.normalized() if rh.length > 1e-6 else fh.cross(Vector((0, 0, 1)))
    if enclosed:
        origins = [probe, cam_loc, (probe + cam_loc) / 2]
        lo, hi = Vector(bmin), Vector(bmax)
        for ax in range(3):
            for sgn in (-1, 1):
                d = Vector((0, 0, 0))
                d[ax] = sgn
                vals = sorted(hh[ax] for hh in (cast(o_, d) for o_ in origins) if hh is not None)
                if vals:
                    v_ = vals[len(vals) // 2]
                    if sgn < 0:
                        lo[ax] = v_
                    else:
                        hi[ax] = v_
        ext = hi - lo
        ctr = (lo + hi) / 2
        hb = cast(cam_loc, -fh)
        d_back = (hb - cam_loc).length if hb is not None else 0.5 * (centre - cam_loc).length
        over = new_light("virtual_fill_overhead", "AREA", Vector((ctr.x, ctr.y, hi.z - 0.03 * ext.z)),
                         forward=Vector((0, 0, -1)), size=(0.5 * ext.x, 0.5 * ext.y), virtual=True,
                         desc="virtual overhead fill")
        zc = ctr.z
        gen["enclosure_box"] = [list(lo), list(hi)]
    else:
        lo, hi = Vector(bmin), Vector(bmax)
        ext = hi - lo
        ctr = (lo + hi) / 2
        d_back = 0.5 * (Vector((centre.x, centre.y, 0)) - Vector((cam_loc.x, cam_loc.y, 0))).length
        over = new_light("virtual_fill_overhead", "AREA",
                         Vector((ctr.x, ctr.y, hi.z + 0.5 * max(ext.z, 0.25 * max(ext.x, ext.y)))),
                         forward=Vector((0, 0, -1)), size=(max(ext.x, 1e-2), max(ext.y, 1e-2)), virtual=True,
                         desc="virtual overhead fill above the visible geometry")
        zc = cam_loc.z
        gen["visible_bbox_p2_p98"] = [list(lo), list(hi)]
    width = abs(rh.x) * ext.x + abs(rh.y) * ext.y
    along = abs(fh.x) * ext.x + abs(fh.y) * ext.y
    behind = new_light("virtual_fill_behind_camera", "AREA", cam_loc - fh * (d_back / 2) + Vector((0, 0, zc - cam_loc.z)),
                       forward=fh, size=(max(0.8 * width, 1e-2), max(0.7 * ext.z, 0.05 * width, 1e-2)), virtual=True,
                       desc="virtual fill behind the camera")
    up_loc = cam_loc + fh * (0.15 * along)
    up_loc.z = lo.z + 0.4 * ext.z if enclosed else lo.z + 0.05 * max(ext.z, 0.1)
    s_up = max(0.25 * min(ext.x, ext.y), 1e-2)
    upl = new_light("virtual_bounce_uplight", "AREA", up_loc, forward=Vector((0, 0, 1)), size=(s_up, s_up),
                    virtual=True, desc="virtual bounce uplight")
    for ob in (over, behind, upl):
        cands.append({"name": ob.name, "kind": "area", "virtual": True, "obj": ob, "input": False,
                      "desc": "VIRTUAL fill standing for an unseen source; safe to delete",
                      "location": [round(x, 3) for x in ob.location], "size_m": [ob.data.size, ob.data.size_y]})
    log(f"{len(cands)} candidates: " + ", ".join(c["name"] for c in cands))

    # ------------------------------------------------------------ basis renders
    def signature(c):
        o = c["obj"]
        if o is None:
            return "world"
        s = [o.data.type] + [round(x, 4) for x in o.matrix_world.translation] + \
            [round(x, 4) for x in o.matrix_world.to_euler()]
        if o.data.type == "AREA":
            s += [round(o.data.size, 4), round(o.data.size_y, 4)]
        return json.dumps(s)

    rv.set_colour_management(scene, "Standard", 0.0)
    exr_of = {c["name"]: basis_dir / f"{c['name']}.exr" for c in cands}
    times, reused = {}, {}
    if args.reuse_basis:
        rdir = Path(args.reuse_basis).resolve()
        rman = common.read_json(rdir / "manifest.json")
        if int(rman["block"]) != blk or {k_: float(v) for k_, v in rman["unit_strengths"].items()} != UNIT:
            raise RuntimeError("--reuse-basis: different basis block or unit strengths")
        old = {c["name"]: c for c in rman["candidates"]}
        for c in cands:
            p = rdir / f"{c['name']}.exr"
            if p.is_file() and old.get(c["name"], {}).get("signature") == signature(c):
                exr_of[c["name"]] = p
                reused[c["name"]] = str(p)
                times[c["name"]] = rman.get("seconds", {}).get(c["name"])
        log(f"reusing {len(reused)} basis EXRs from {rdir}")
    for c in cands:
        if c["name"] in reused:
            continue
        for o in cands:
            if o["obj"] is not None:
                o["obj"].hide_render = o is not c
                o["obj"].data.energy = UNIT[o["kind"]]
        bg.inputs["Strength"].default_value = UNIT["world"] if c["kind"] == "world" else 0.0
        spp = args.sun_samples if c["kind"] == "sun" else args.basis_samples
        rv.configure_cycles(scene, spp, bw, bh, adaptive_threshold=0.01)
        times[c["name"]] = round(rv.render_to(scene, exr=exr_of[c["name"]]), 2)
        log(f"basis {c['name']:34s} {spp:4d} spp {times[c['name']]:6.1f}s")
    common.write_json(basis_dir / "manifest.json", {
        "scale": 1.0 / blk, "block": blk, "basis_size": [bw, bh], "target": str(tgt), "id": str(gb / "id.png"),
        "objects": objects_meta, "unit_strengths": UNIT, "seconds": times, "reused_from": reused,
        "basis_samples": args.basis_samples, "sun_samples": args.sun_samples,
        "candidates": [{"name": c["name"], "kind": c["kind"], "virtual": c["virtual"], "input": c["input"],
                        "sun_group": c.get("sun_group"), "signature": signature(c), "exr": str(exr_of[c["name"]])}
                       for c in cands]})

    # ------------------------------------------------------------ solve (user's Python)
    res = subprocess.run([common.python_exe(), "-m", "b2r.relight_solve", "--work", str(bp),
                          "--prune-margin", str(args.prune_margin), "--prune-total-cap", str(args.prune_total_cap)],
                         env=env)
    if res.returncode != 0:
        raise RuntimeError("solver failed")
    sol = {c["name"]: c for c in common.read_json(basis_dir / "solution.json")["candidates"]}
    fit = common.read_json(bp / "fit.json")

    # ------------------------------------------------------------ apply
    lights_out, deleted = [], []
    for c in cands:
        s = sol[c["name"]]
        wv = s["rgb_weight_final"]
        mx = max(wv)
        if not s["keep"] or mx <= 0:
            deleted.append({"name": c["name"], "input_light": c["input"], "virtual": c["virtual"]})
            if c["obj"] is not None:
                ld = c["obj"].data
                bpy.data.objects.remove(c["obj"], do_unlink=True)
                if ld.users == 0:
                    bpy.data.lights.remove(ld)
            else:
                bg.inputs["Strength"].default_value = 0.0
            continue
        colour = [x / mx for x in wv]
        strength = UNIT[c["kind"]] * mx
        entry = {"name": c["name"], "virtual": c["virtual"], "input_light": c["input"], "description": c["desc"],
                 "colour_rgb_linear": colour, "strength": strength, "strength_unit": UNIT_LABEL[c["kind"]],
                 "energy_share_of_image": s["energy_share_final"]}
        if c["obj"] is None:
            set_world_rgb(colour)
            bg.inputs["Strength"].default_value = strength
            entry.update(type="WORLD_BACKGROUND", location=None)
        else:
            ob = c["obj"]
            ob.hide_render = False
            ob.data.color = colour
            ob.data.energy = strength
            if c["input"]:
                ob[INPUT_LIGHT_PROP] = True
            entry.update(type=ob.data.type, location=list(ob.matrix_world.translation),
                         rotation_euler_deg=[math.degrees(a) for a in ob.matrix_world.to_euler()],
                         visible_to_camera=bool(ob.visible_camera), visible_to_glossy=bool(ob.visible_glossy))
            if ob.data.type == "AREA":
                entry["size_m"] = [ob.data.size, ob.data.size_y]
            if ob.data.type == "SUN":
                entry["angle_deg"] = math.degrees(ob.data.angle)
                entry["sun_group"] = c.get("sun_group")
                if "direction_towards_sun" in c:
                    entry["direction_towards_sun"] = c["direction_towards_sun"]
            if c["input"]:
                entry.update(original_energy=c["original_energy"], original_color=c["original_color"])
        lights_out.append(entry)
    for pt in portals:
        lights_out.append({"name": pt.name, "type": "AREA (Cycles light portal, emits nothing)", "virtual": False,
                           "input_light": False, "description": "sampling helper for the sky through an opening",
                           "location": list(pt.location), "size_m": [pt.data.size, pt.data.size_y],
                           "strength": None, "energy_share_of_image": 0.0})
    log("kept:", [(e["name"], round(e["strength"], 4) if e["strength"] else None,
                   round(e["energy_share_of_image"], 3)) for e in lights_out])

    rv.set_colour_management(scene, "Standard", 0.0)
    rv.configure_cycles(scene, args.final_samples, W, H, adaptive_threshold=0.005)
    bpy.ops.file.pack_all()
    lit_path = bp / "scene_lit.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(lit_path), compress=True)
    log(f"saved {lit_path}")
    common.write_json(bp / "lights.json", {
        "scene": str(lit_path), "candidates_considered": len(cands), "candidates_deleted": deleted,
        "candidate_generation": gen, "unit_strengths_of_bases": UNIT,
        "energy_share_definition": "share of total fitted luminance over valid object pixels (basis resolution)",
        "virtual_note": "lights with virtual=true stand for sources the camera cannot see; they are hidden from "
                        "camera and glossy rays and may be deleted",
        "colour_management": {"view_transform": "Standard", "look": "None", "exposure": 0.0},
        "tint_clamped": fit["tint_clamped"], "pruned": [p["light"] for p in fit["pruning"]["pruned"]],
        "lights": lights_out})

    # ------------------------------------------------------------ re-render
    t_r = None
    if args.no_rerender:
        log("re-render skipped (--no-rerender)")
    else:
        png = Path(args.rerender_out).resolve() if args.rerender_out else \
            common.stage_dir(case, "residual", k) / "rerender.png"
        rv.configure_cycles(scene, args.final_samples, max(1, round(W * args.rerender_scale)),
                            max(1, round(H * args.rerender_scale)), adaptive_threshold=0.005)
        t_r = rv.render_to(scene, png=png, exr=png.with_suffix(".exr"))
        log(f"re-render {scene.render.resolution_x}x{scene.render.resolution_y} {args.final_samples} spp: "
            f"{t_r:.1f}s -> {png}")
    fit["timing_s"] = {"basis_total": round(sum(v for v in times.values() if v), 1), "basis_per_candidate": times,
                       "rerender": None if t_r is None else round(t_r, 1), "stage_total": round(time.time() - T0, 1),
                       "basis_reused": sorted(reused)}
    common.write_json(bp / "fit.json", fit)
    (bp / "relight_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    try:
        if IN_BLENDER:
            blender_main()
        else:
            ap = argparse.ArgumentParser()
            ap.add_argument("--analyze", action="store_true", required=True)
            ap.add_argument("--job", required=True)
            a = ap.parse_args(common.script_args())
            analyze_main(Path(a.job))
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
