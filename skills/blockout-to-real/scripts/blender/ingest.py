"""ingest: any .blend or mesh file -> a grey, UV-unwrapped blockout (same outputs as blockout.py).

    blender -b --factory-startup --python scripts/blender/ingest.py -- \
        --case CASE [--iter K] [--input PATH] [--camera NAME] [--group-by object|collection|parent]
        [--samples N] [--no-render]

--input defaults to the case input; --camera / --group-by default to case.json. Writes
CASE/blockout/iter<K>/: blockout.blend, objects.json, verification.json, preview.png.

What it does, in order
  1. Opens the .blend (its active scene; other scenes are removed) or imports the mesh file
     (.glb .gltf .obj .fbx .usd .usda .usdc .usdz .ply .stl) into an empty scene.
  2. Evaluates the scene with RENDER settings (modifier render visibility and levels, curve/text/
     metaball render resolution, particle render amount) and turns every renderable instance into a
     plain mesh: modifiers applied, instances (collections, particles, geometry nodes, verts/faces)
     realised, curves / surfaces / text / metaballs converted. The world transform is baked into the
     mesh (rotation 0, scale 1, no parents); the object keeps its world location as origin (a group:
     the bottom centre of its bounding box). Volumes, grease pencil, hair curves and point clouds
     cannot become ordinary meshes and are listed as skipped. Objects with no surface area are
     dropped and listed.
  3. --group-by: "object" (one per evaluated object instance), "collection" (one per collection of
     the top-level object or instancer), "parent" (one per root of the parent hierarchy).
  4. Camera: --camera, else the scene camera, else the first camera object, else a new camera
     "b2r_camera" that frames every object's bounding box from the front-right, 25 degrees above
     (stated in the log and verification.json). Only perspective cameras are accepted. The camera is
     frozen at its current world matrix (parents, constraints, animation removed).
  5. Size: if case.json "size" is null it is chosen now from the camera's aspect (the blend's
     render resolution x pixel aspect; an auto camera has none) by b2r.case.decide_size, and written
     to case.json together with the camera name.
  6. Lights: the scene's own render-visible lights are kept (instanced lights realised), frozen like
     the camera and tagged b2r_input_light=True: they are light-fit candidates later. The world is
     kept. Only when the scene has no light objects and no lit world, neutral preview lights
     (preview_*, b2r_preview_light=True) are added.
  7. Every object gets one neutral grey Principled material (b2r_grey). The original material
     names, base colours, image textures and emission are recorded in objects.json as hints; they
     are not used for rendering.
  8. Every object gets a NEW UV map "b2r_uv" (Smart UV Project + pack_islands with a margin, inside
     [0,1]; lightmap pack as a fallback), verified to have zero inter-island overlap. Existing UV maps
     are left in place and not used.
  9. ids: pass_index 1..N in name order. category: "structure" if the object's bounding box spans
     >= 60% of the scene's bounding box along at least two axes AND either encloses the camera
     position, or is a slab (shortest side <= 10% of its longest) whose two long axes contain the
     camera's coordinates (floors, walls, backdrops); every other object is "object".
     description: seeded from the object name and its original material hints; edit it.
 10. Reports unusual things (NOTE lines + verification.json "notes"): dropped zero-area objects,
     skipped object types, very high face counts, more than 60 objects (suggests --group-by),
     missing linked libraries or world images, odd scene scale.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import blockout as B  # noqa: E402  (shared Blender helpers)
from b2r import common  # noqa: E402
from b2r import case as case_mod  # noqa: E402

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

TAG = "ingest"
GEOM_TYPES = ("MESH", "CURVE", "SURFACE", "FONT", "META")
SKIP_TYPES = ("VOLUME", "GREASEPENCIL", "GPENCIL", "CURVES", "POINTCLOUD")
MANY_OBJECTS = 60
MANY_FACES_OBJECT = 250_000
MANY_FACES_TOTAL = 2_000_000
STRUCT_SPAN = 0.6
SLAB_RATIO = 0.10

notes: list[str] = []


def note(msg: str) -> None:
    notes.append(msg)
    B.log(TAG, "NOTE " + msg)


# =========================================================================== loading
def load_input(path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
        for lib in bpy.data.libraries:
            if not Path(bpy.path.abspath(lib.filepath)).is_file():
                note(f"linked library {lib.filepath!r} is missing: its data is absent from the blockout")
        return
    bpy.ops.wm.read_factory_settings(use_empty=True)
    fp = str(path)
    if ext in (".glb", ".gltf"):
        r = bpy.ops.import_scene.gltf(filepath=fp)
    elif ext == ".obj":
        r = bpy.ops.wm.obj_import(filepath=fp)
    elif ext == ".fbx":
        r = bpy.ops.import_scene.fbx(filepath=fp)
    elif ext in (".usd", ".usda", ".usdc", ".usdz"):
        r = bpy.ops.wm.usd_import(filepath=fp)
    elif ext == ".ply":
        r = bpy.ops.wm.ply_import(filepath=fp)
    elif ext == ".stl":
        r = bpy.ops.wm.stl_import(filepath=fp)
    else:
        raise SystemExit(f"unsupported input {ext}")
    if "FINISHED" not in r:
        raise RuntimeError(f"import of {path.name} failed: {r}")


def use_render_settings(scene) -> None:
    """Make the (viewport) depsgraph evaluate what a render would show."""
    for o in scene.objects:
        o.hide_viewport = o.hide_render
        o.hide_set(False)
        for m in getattr(o, "modifiers", []):
            m.show_viewport = m.show_render
            if m.type == "SUBSURF":
                m.levels = m.render_levels
            elif m.type == "MULTIRES":
                m.levels = m.render_levels
        for ps in getattr(o, "particle_systems", []):
            ps.settings.display_percentage = 100
            if hasattr(ps.settings, "rendered_child_count"):
                ps.settings.child_percentage = ps.settings.rendered_child_count
        d = o.data
        if o.type in ("CURVE", "SURFACE", "FONT") and getattr(d, "render_resolution_u", 0):
            d.resolution_u = d.render_resolution_u
        if o.type == "SURFACE" and getattr(d, "render_resolution_v", 0):
            d.resolution_v = d.render_resolution_v
        if o.type == "META":
            d.resolution = d.render_resolution
    for c in bpy.data.collections:
        c.hide_viewport = c.hide_render

    def walk(lc):
        lc.hide_viewport = False
        for ch in lc.children:
            walk(ch)
    walk(bpy.context.view_layer.layer_collection)


# =========================================================================== material hints
def _hex(rgb_srgb) -> str:
    return "#" + "".join(f"{int(round(max(0, min(1, c)) * 255)):02x}" for c in rgb_srgb[:3])


def lin_to_srgb(c: float) -> float:
    c = max(0.0, float(c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def colour_word(srgb) -> str:
    h, s, v = colorsys.rgb_to_hsv(*[max(0, min(1, x)) for x in srgb[:3]])
    if v < 0.12:
        return "black"
    if s < 0.12:
        return "white" if v > 0.88 else "light grey" if v > 0.62 else "grey" if v > 0.32 else "dark grey"
    hd = h * 360
    names = ((15, "red"), (40, "orange"), (65, "yellow"), (170, "green"), (200, "cyan"), (255, "blue"),
             (290, "purple"), (340, "pink"), (361, "red"))
    name = next(n for lim, n in names if hd < lim)
    if name in ("orange", "red") and v < 0.6 and s > 0.3:
        name = "brown" if name == "orange" else "dark red"
    elif v < 0.4:
        name = "dark " + name
    elif s < 0.35 and v > 0.7:
        name = "pale " + name
    return name


def _upstream(node, socket_name):
    s = node.inputs.get(socket_name)
    if s is None or not s.is_linked:
        return None, s
    return s.links[0].from_node, s


def material_hint(mat) -> dict:
    h = {"name": mat.name}
    base = None
    if mat.use_nodes and mat.node_tree:
        nodes = mat.node_tree.nodes
        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None) or \
            next((n for n in nodes if n.inputs.get("Base Color") or n.inputs.get("Color")), None)
        if bsdf is not None:
            key = "Base Color" if bsdf.inputs.get("Base Color") else "Color"
            src, sock = _upstream(bsdf, key)
            if src is None and sock is not None and hasattr(sock, "default_value"):
                base = list(sock.default_value)[:3]
            elif src is not None:
                seen, stack, imgs = set(), [src], []
                while stack:
                    n = stack.pop()
                    if n.name in seen:
                        continue
                    seen.add(n.name)
                    if n.type == "TEX_IMAGE" and n.image:
                        imgs.append(n.image.name)
                    elif n.type == "RGB":
                        base = list(n.outputs[0].default_value)[:3]
                    stack.extend(l.from_node for i in n.inputs for l in i.links)
                h["base_color_texture"] = imgs[0] if imgs else "procedural"
            if bsdf.type == "BSDF_PRINCIPLED":
                for k, name in (("Roughness", "roughness"), ("Metallic", "metallic")):
                    s = bsdf.inputs.get(k)
                    if s is not None and not s.is_linked:
                        h[name] = round(float(s.default_value), 3)
                es = bsdf.inputs.get("Emission Strength")
                ec = bsdf.inputs.get("Emission Color")
                if es is not None and ec is not None and (es.is_linked or es.default_value > 0) and \
                        (ec.is_linked or max(list(ec.default_value)[:3]) > 0):
                    h["emissive"] = True
        if any(n.type == "EMISSION" and n.outputs[0].is_linked for n in nodes):
            h["emissive"] = True
    if base is None and "base_color_texture" not in h:
        base = list(mat.diffuse_color)[:3]
    if base is not None:
        srgb = [lin_to_srgb(c) for c in base]
        h["base_color_srgb"] = _hex(srgb)
        h["colour"] = colour_word(srgb)
    return h


def clean_name(n: str) -> str:
    n = re.sub(r"\.\d{3,}$", "", n)
    n = re.sub(r"[_\-.]+", " ", n).strip()
    n = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n)
    return n.lower() or "object"


def seed_description(name: str, mats: list[dict]) -> str:
    d = clean_name(name)
    bits = []
    for m in mats[:3]:
        s = f"'{m['name']}'"
        extra = [m[k] for k in ("colour",) if k in m]
        if "base_color_srgb" in m:
            extra.append(m["base_color_srgb"])
        if "base_color_texture" in m:
            extra.append(f"texture {m['base_color_texture']}")
        if extra:
            s += f" ({', '.join(extra)})"
        bits.append(s)
    return d + (f"; original material {', '.join(bits)}" if bits else "")


# =========================================================================== conversion
def root_of(o):
    while o.parent is not None:
        o = o.parent
    return o


def group_key(mode: str, orig, instancer) -> str:
    top = instancer if instancer is not None else orig
    if mode == "collection":
        cols = [c for c in top.users_collection if c != bpy.context.scene.collection]
        return cols[0].name if cols else root_of(top).name
    if mode == "parent":
        return root_of(top).name
    return orig.name


def bake_world(me, M: Matrix) -> None:
    me.transform(M)
    if M.determinant() < 0:
        me.flip_normals()
    me.update()


def convert_scene(scene, mode: str):
    """-> (groups: key -> list of (mesh, single_location or None, orig name, instancer name)),
    light instances, stats)."""
    dg = bpy.context.evaluated_depsgraph_get()
    groups: dict[str, list] = {}
    light_insts = []
    counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    inst_counter: dict[str, int] = {}
    for inst in dg.object_instances:
        ob = inst.object
        t = ob.type
        if t == "LIGHT":
            if inst.is_instance:
                light_insts.append((ob.original.name, ob.data.original if ob.data else None,
                                    inst.matrix_world.copy(), inst.parent.original.name if inst.parent else None))
            continue
        if t in SKIP_TYPES:
            skipped[t] = skipped.get(t, 0) + 1
            continue
        if t not in GEOM_TYPES:
            continue
        orig = ob.original
        instancer = inst.parent.original if (inst.is_instance and inst.parent) else None
        try:
            me = bpy.data.meshes.new_from_object(ob, preserve_all_data_layers=True, depsgraph=dg)
        except RuntimeError as exc:
            note(f"could not convert {orig.name} ({t}): {exc}")
            continue
        counts[t] = counts.get(t, 0) + 1
        M = inst.matrix_world.copy()
        bake_world(me, M)
        if mode == "object":
            if instancer is not None:
                k = f"{instancer.name}_{orig.name}"
                inst_counter[k] = inst_counter.get(k, 0) + 1
                key = f"{k}_{inst_counter[k]:03d}"
            else:
                key = orig.name
        else:
            key = group_key(mode, orig, instancer)
        single_loc = M.translation.copy() if (mode == "object" and instancer is None) else None
        groups.setdefault(key, []).append((me, single_loc, orig.name, instancer.name if instancer else None,
                                           orig.users_collection[0].name if orig.users_collection else None))
    return groups, light_insts, counts, skipped


def join_group(scene, key: str, members: list, coll) -> bpy.types.Object | None:
    hints_mats, seen = [], set()
    for me, *_ in members:
        for m in me.materials:
            if m is not None and m.name not in seen:
                seen.add(m.name)
                hints_mats.append(material_hint(m))
    objs = []
    for me, *_ in members:
        o = bpy.data.objects.new(key, me)
        coll.objects.link(o)
        objs.append(o)
    obj = objs[0]
    if len(objs) > 1:
        with bpy.context.temp_override(active_object=obj, object=obj, selected_objects=objs,
                                       selected_editable_objects=objs):
            bpy.ops.object.join()
    me = obj.data
    me.name = obj.name
    if len(members) == 1 and members[0][1] is not None:
        t = members[0][1]
    else:
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        if len(co):
            lo, hi = co.min(0), co.max(0)
            t = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
        else:
            t = Vector((0, 0, 0))
    me.transform(Matrix.Translation(-t))
    obj.location = t
    src = {"objects": sorted({m[2] for m in members}),
           "instanced_by": sorted({m[3] for m in members if m[3]}) or None,
           "collections": sorted({m[4] for m in members if m[4]}) or None,
           "original_uv_maps": [uv.name for uv in me.uv_layers]}
    obj["b2r_hints"] = json.dumps({"source": src, "original_materials": hints_mats})
    obj["b2r_mats"] = json.dumps(hints_mats)
    return obj


def freeze(o) -> None:
    """Keep an object (camera / light) exactly where it renders now, with no dependencies."""
    M = o.matrix_world.copy()
    o.animation_data_clear()
    if o.data is not None and hasattr(o.data, "animation_data_clear"):
        o.data.animation_data_clear()
    for c in list(o.constraints):
        o.constraints.remove(c)
    o.parent = None
    o.matrix_world = M


# =========================================================================== camera
def pick_camera(scene, want: str | None):
    if want:
        cam = bpy.data.objects.get(want)
        if cam is None or cam.type != "CAMERA":
            raise SystemExit(f"--camera {want!r}: no camera object of that name "
                             f"(cameras: {[o.name for o in scene.objects if o.type == 'CAMERA']})")
        return cam, f"--camera {want}"
    if scene.camera is not None and scene.camera.type == "CAMERA":
        return scene.camera, "scene camera"
    cams = sorted((o for o in scene.objects if o.type == "CAMERA"), key=lambda o: o.name)
    if cams:
        note(f"the scene has no active camera; using camera object {cams[0].name!r}")
        return cams[0], "first camera object"
    return None, None


def auto_camera(scene, objs: list, size: dict):
    """A camera that frames every object's bounding box from the front-right, 25 deg above."""
    pts = []
    for o in objs:
        w = B.world_verts(o)
        lo, hi = w.min(0), w.max(0)
        pts += [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    centre = (lo + hi) / 2
    az, el = math.radians(30), math.radians(25)
    d_out = np.array([math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)])
    cd = bpy.data.cameras.new("b2r_camera")
    cd.lens, cd.sensor_width, cd.sensor_fit = 35.0, 36.0, "AUTO"
    cam = bpy.data.objects.new("b2r_camera", cd)
    scene.collection.objects.link(cam)
    fwd = Vector((-d_out).tolist())
    cam.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler("XYZ")
    R = np.array(cam.rotation_euler.to_matrix())
    right, up, f = R[:, 0], R[:, 1], -R[:, 2]
    W, H = size["width"], size["height"]
    fpx = cd.lens / cd.sensor_width * max(W, H)
    tx, ty = (W / 2) / fpx * 0.9, (H / 2) / fpx * 0.9
    rel = pts - centre
    x, y, z = rel @ right, rel @ up, rel @ f
    dist = float(max(np.max(np.abs(x) / tx - z), np.max(np.abs(y) / ty - z)))
    diag = float(np.linalg.norm(hi - lo))
    dist = max(dist, 0.05 * diag + 1e-3)
    cam.location = Vector((centre + d_out * dist).tolist())
    cd.clip_start = max(1e-3, min(0.1, dist * 0.01))
    cd.clip_end = dist + 2 * diag + 1.0
    scene.camera = cam
    note(f"no camera in the input: created 'b2r_camera' (35 mm, {size['aspect_ratio']}) framing all objects "
         f"from the front-right, 25 deg above, {dist:.2f} m from the bbox centre; check preview.png, "
         f"or re-run `new` with --camera after adding one")
    return cam


# =========================================================================== lights / world
def world_is_lit(scene) -> tuple[bool, str]:
    w = scene.world
    if w is None:
        return False, "no world"
    if not w.use_nodes or not w.node_tree:
        lit = max(w.color) > 0
        return lit, f"world colour {tuple(round(c, 3) for c in w.color)}"
    nodes = w.node_tree.nodes
    for n in nodes:
        if n.type == "TEX_ENVIRONMENT":
            img = n.image
            if img is None or (img.source == "FILE" and not img.packed_file
                               and not Path(bpy.path.abspath(img.filepath)).is_file()):
                note(f"world environment image {img.filepath if img else None!r} is missing")
                return False, "world environment image missing"
            return True, f"world environment image {img.name}"
    bg = [n for n in nodes if n.type == "BACKGROUND"]
    if bg:
        s = bg[0].inputs["Strength"]
        c = bg[0].inputs["Color"]
        lit = (s.is_linked or s.default_value > 0) and (c.is_linked or max(list(c.default_value)[:3]) > 0)
        return lit, f"world background strength {s.default_value:.3g}"
    return False, "world without background node"


# =========================================================================== category
def categorise(objs: list, cam_loc: np.ndarray) -> tuple[dict, dict]:
    bb = {}
    for o in objs:
        w = B.world_verts(o)
        bb[o.name] = (w.min(0), w.max(0))
    lo = np.min([b[0] for b in bb.values()], 0)
    hi = np.max([b[1] for b in bb.values()], 0)
    ext = np.maximum(hi - lo, 1e-9)
    pad = 0.05 * ext
    cats = {}
    for o in objs:
        a, b = bb[o.name]
        e = b - a
        span = e >= STRUCT_SPAN * ext
        inside = (cam_loc >= a - pad) & (cam_loc <= b + pad)
        is_struct = False
        if len(objs) > 1 and span.sum() >= 2:
            if inside.all():
                is_struct = True
            elif e.min() <= SLAB_RATIO * e.max():
                long_axes = np.argsort(e)[1:]
                is_struct = bool(inside[long_axes].all())
        cats[o.name] = "structure" if is_struct else "object"
    rule = (f"structure = bbox spans >= {STRUCT_SPAN:.0%} of the scene bbox along >= 2 axes AND (encloses the "
            f"camera OR is a slab, shortest side <= {SLAB_RATIO:.0%} of longest, whose two long axes contain the "
            "camera's coordinates); everything else = object")
    return cats, {"rule": rule, "scene_bbox_min": [round(float(x), 4) for x in lo],
                  "scene_bbox_max": [round(float(x), 4) for x in hi]}


# =========================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("--input", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--group-by", default=None, choices=("object", "collection", "parent"))
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(common.script_args())
    t_start = time.time()

    cfg = common.load_case(args.case)
    src = Path(args.input).expanduser().resolve() if args.input else case_mod.input_path(args.case, cfg)
    mode = args.group_by or cfg.get("settings", {}).get("group_by") or "object"
    want_cam = args.camera or cfg.get("camera")
    out = common.stage_dir(args.case, "blockout", args.iter)

    load_input(src)
    scene = bpy.context.scene
    for s in list(bpy.data.scenes):
        if s != scene:
            note(f"input has another scene {s.name!r}: only the active scene {scene.name!r} is used")
            bpy.data.scenes.remove(s)
    for vl in scene.view_layers:
        if vl != bpy.context.view_layer:
            vl.use = False
    unit = scene.unit_settings.scale_length
    use_render_settings(scene)
    bpy.context.view_layer.update()

    cam, cam_source = pick_camera(scene, want_cam)
    if cam is not None and cam.data.type != "PERSP":
        raise SystemExit(f"camera {cam.name!r} is {cam.data.type}; only perspective cameras are supported")
    cam_wh = B.camera_aspect(scene) if (cam is not None and src.suffix.lower() == ".blend") else None
    if cam is not None and cam_wh is None:
        cam_wh = B.camera_aspect(scene)
        note(f"camera aspect of a {src.suffix} camera taken from the importer's render size "
             f"{scene.render.resolution_x}x{scene.render.resolution_y}; pass --aspect to `new` if it is wrong")

    groups, light_insts, conv_counts, skipped = convert_scene(scene, mode)
    for t, n in skipped.items():
        note(f"{n} {t} object(s) cannot become ordinary meshes and were skipped")

    # lights and cameras to keep (frozen); instanced lights realised
    kept_lights = [o for o in scene.objects if o.type == "LIGHT" and not o.hide_render]
    keep = set(kept_lights) | {o for o in scene.objects if o.type == "CAMERA"}
    for o in keep:
        freeze(o)
    for name, ldata, M, parent in light_insts:
        lo = bpy.data.objects.new(f"{parent}_{name}" if parent else name, ldata.copy() if ldata else None)
        scene.collection.objects.link(lo)
        lo.matrix_world = M
        kept_lights.append(lo)
        keep.add(lo)
    for o in keep:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        scene.collection.objects.link(o)
    for o in list(scene.objects):
        if o not in keep:
            bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)

    # build the blockout objects
    work = bpy.data.collections.new("b2r_ingest")
    scene.collection.children.link(work)
    objs, dropped = [], []
    for key in sorted(groups):
        o = join_group(scene, key, groups[key], work)
        if o is None:
            continue
        area = B.surface_area(o)
        if len(o.data.polygons) == 0 or area < 1e-10:
            dropped.append({"name": o.name, "faces": len(o.data.polygons), "area_m2": area})
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        objs.append(o)
    for d in dropped:
        note(f"dropped {d['name']!r}: no surface ({d['faces']} faces, area {d['area_m2']:.2e} m2) renders nothing")
    if not objs:
        raise SystemExit("the input has no renderable surface geometry")
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    mat = B.grey_material()
    bpy.context.view_layer.update()

    # camera / size
    if cam is None:
        size = B.apply_case_size(args.case, scene, None, "auto camera", "b2r_camera")
        cam = auto_camera(scene, objs, size)
        cam_source = "auto (created)"
    else:
        size = B.apply_case_size(args.case, scene, cam_wh, cam_source, cam.name)
        scene.camera = cam
    B.log(TAG, f"camera {cam.name!r} ({cam_source}); size {size['width']}x{size['height']} {size['aspect_ratio']} "
               f"verified={size['size_verified']} ({size.get('aspect_source', 'case.json')})")

    # lights
    for o in kept_lights:
        o[B.TAG_INPUT_LIGHT] = True
    lit_world, world_note = world_is_lit(scene)
    emissive = sorted({o.name for o in objs for m in json.loads(o["b2r_mats"]) if m.get("emissive")})
    if emissive:
        note(f"objects with emissive original materials (their light is gone in the grey blockout): {emissive}")
    if kept_lights:
        lighting = f"kept {len(kept_lights)} input light(s) tagged {B.TAG_INPUT_LIGHT}; {world_note}"
    elif lit_world:
        lighting = f"no light objects; the input world lights the scene ({world_note})"
    else:
        R = np.array(cam.matrix_world)[:3, :3]
        sun_dir = (-R[:, 2] + 0.6 * R[:, 0] - np.array([0, 0, 1.0])).tolist()
        added = B.add_preview_lights(scene, {"sun_strength": 3.0, "sun_direction": sun_dir,
                                             "sky_color": (0.6, 0.6, 0.6), "sky_strength": 0.5})
        lighting = "the input has no lights and no lit world: added neutral preview lighting: " + "; ".join(added)
        note(lighting)
    B.log(TAG, lighting)

    # categories, ids, descriptions, grey, UVs
    cats, cat_info = categorise(objs, np.array(cam.matrix_world.translation))
    uv = {}
    for i, o in enumerate(sorted(objs, key=lambda x: x.name), start=1):
        mats = json.loads(o["b2r_mats"])
        del o["b2r_mats"]
        B.tag_object(o, i, cats[o.name], seed_description(o.name, mats))
        B.set_grey(o, mat)
        uv[o.name] = B.unwrap_checked(o, TAG)
    objs = B.mesh_objects(scene)
    for o in objs:
        work.objects.unlink(o)
        B.collection_for(scene, o["category"]).objects.link(o)
    bpy.data.collections.remove(work)
    bpy.context.view_layer.update()

    # unusual things
    faces = {o.name: len(o.data.polygons) for o in objs}
    big = {n: f for n, f in faces.items() if f > MANY_FACES_OBJECT}
    if big:
        note(f"very high face counts (UV unwrap and texture projection get slow): {big}")
    if sum(faces.values()) > MANY_FACES_TOTAL:
        note(f"{sum(faces.values())} faces in total")
    if len(objs) > MANY_OBJECTS and mode == "object":
        note(f"{len(objs)} objects: consider `new ... --group-by collection` or `--group-by parent` "
             "(each object becomes one material and one row in every per-object table)")
    ext = np.array(cat_info["scene_bbox_max"]) - np.array(cat_info["scene_bbox_min"])
    if ext.max() > 2000 or ext.max() < 0.05:
        note(f"scene extent {ext.max():.3g} m is unusual (unit scale {unit}); downstream stages assume metres")
    if unit != 1.0:
        note(f"scene unit scale_length is {unit}; geometry is used as stored (Blender units = metres)")
    if not any(cats[o.name] == "structure" for o in objs):
        B.log(TAG, "no object classified as structure (fine for open scenes; edit categories if needed)")

    report = B.verify(scene, cam, objs, uv, room=None)
    report["source"] = {"kind": cfg["input"]["kind"], "input": src.name, "sha256": case_mod.sha256_file(src),
                        "group_by": mode, "converted_instances_by_type": conv_counts, "skipped_by_type": skipped,
                        "dropped_no_surface": dropped}
    report["camera_source"] = cam_source
    report["lighting"] = lighting
    report["category"] = cat_info
    report["notes"] = notes
    report["size"] = size
    report["ingest_seconds"] = round(time.time() - t_start, 1)
    B.write_outputs(out, scene, cam, objs, uv, report, args.samples, not args.no_render, TAG)
    B.print_summary(TAG, report, out)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
