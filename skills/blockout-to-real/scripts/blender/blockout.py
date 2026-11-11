"""blockout: build a grey, UV-unwrapped scene from a JSON scene spec.

    blender -b --factory-startup --python scripts/blender/blockout.py -- \
        --case CASE [--iter K] [--spec PATH] [--samples N] [--no-render]

--spec defaults to the case input (case.json "input", kind "spec"). Writes CASE/blockout/iter<K>/:
    blockout.blend     meshes, one grey material, the spec camera, spec lights or preview lights
    scene_spec.json    exact byte copy of the spec used
    objects.json       one row per object: name, id, category, description, bbox, counts, UV stats,
                       geometry_sha256
    verification.json  UV range/overlap, pass_index, scale/rotation, interpenetration, support, in-frame
    preview.png        Cycles render from the spec camera at the case size

If case.json "size" is still null, the size is chosen here from the spec camera's aspect
(b2r.case.decide_size) and written to case.json before anything is rendered.

Spec schema (all lengths metres, angles degrees, Blender world axes, Z up)
---------------------------------------------------------------------------
{
  "units":  "free-text note",                                        optional
  "room":   {"width_x": W, "depth_y": D, "height_z": H, ...},       optional; only used by the
                                    support check (gap to the planes x=0,W / y=0,D) and to place a
                                    preview ceiling light
  "camera": {
      "name": "Camera",                       optional, default "Camera"
      "location": [x, y, z],
      "look_at": [x, y, z]  OR  "rotation": [rx, ry, rz],          (Blender XYZ euler)
      "focal_length_mm": 35  OR  "fov_deg": horizontal FOV,       default 35 mm
      "sensor_width_mm": 36,                  optional, sensor fit is horizontal
      "aspect_ratio": "16:9"  OR  "resolution": [w, h],          optional: the camera's aspect,
                                    used only to choose the case size when it is not set yet
      "clip_start": 0.05, "clip_end": 100     optional
  },
  "lights": [                                 optional: the scene's own lights. Kept, tagged
     {"name": "key", "type": "POINT|SUN|SPOT|AREA",            b2r_input_light=True
      "location": [x,y,z], "look_at": [x,y,z] OR "rotation": [..] OR "direction": [dx,dy,dz],
      "energy": W (sun: W/m2), "color": [r,g,b] linear, "size": m, "size_y": m,
      "shape": "SQUARE|RECTANGLE|DISK|ELLIPSE", "spot_size_deg": 45}
  ],
  "world": {"color": [r,g,b], "strength": 1.0},                 optional world background
  "preview_lighting": {"sun_strength": 4, "sun_direction": [dx,dy,dz],
                       "sky_color": [r,g,b], "sky_strength": 1, "ceiling_area_watts": 150},
                                    optional; used only when "lights" is absent: a sun, a sky
                                    colour and (with "room") one ceiling area light, all named
                                    preview_* and tagged b2r_preview_light=True
  "objects": [
    {
      "name": "unique name",                  becomes the Blender object name
      "id": 1,                                unique int >= 1, = pass_index; never reused
      "category": "free text",                "structure" / "exterior" are read by tools
      "description": "for the image model: material, colour, finish",
      "location": [x, y, z],                  object origin in world, default [0,0,0]
      "rotation": [rx, ry, rz],               baked into the mesh (object rotation stays 0)
      "parts": [                              primitives in OBJECT-LOCAL coordinates
        {"type": "box",      "size": [sx, sy, sz]},
        {"type": "cylinder", "radius": r, "depth": h, "segments": 32},
        {"type": "cone",     "radius_bottom": r1, "radius_top": r2, "depth": h, "segments": 32},
        {"type": "sphere",   "radius": r}  or  {"type": "sphere", "size": [dx, dy, dz]},
                                              size = full diameters (ellipsoid);
                                              "segments" (32) / "rings" (16)
        {"type": "plane",    "size": [sx, sy]}    in local XY, normal +Z
      ]
      every part also accepts "location" (centre, default 0), "rotation" (XYZ euler about the
      centre), "bevel" (width in metres, bevels edges sharper than 30 deg), "bevel_segments" (2).
      Cylinders and cones are along local Z, centred on "location".
    }
  ]
}
See templates/spec.example.json.

Build rules: the parts of an object are joined into ONE mesh object whose origin is the object
location and whose rotation and scale are applied. Normals are recalculated outward per closed part.
Every object gets a UV map "b2r_uv" (Smart UV Project + pack_islands with a margin, islands inside
[0,1]; verified non-overlapping). One shared flat grey Principled BSDF. pass_index = spec id.

This file is also the shared library of the Blender-side geometry stages (ingest.py, edit.py import
it): part building, b2r_uv unwrap + check, geometry checksums, verification, objects.json rows,
preview lighting and rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from b2r import common  # noqa: E402
from b2r import case as case_mod  # noqa: E402

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Euler, Matrix, Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

UV_NAME = "b2r_uv"
GREY = (0.5, 0.5, 0.5, 1.0)
GREY_NAME = "b2r_grey"
TAG_INPUT_LIGHT = "b2r_input_light"
TAG_PREVIEW_LIGHT = "b2r_preview_light"
PART_TYPES = ("box", "cylinder", "cone", "sphere", "plane")
PENETRATION_VERT_LIMIT = 300_000  # skip the (python) interpenetration check above this many vertices


def log(tag: str, *a) -> None:
    print(f"[{tag}]", *a, flush=True)


# =========================================================================== spec / parts
def validate_parts(parts, where: str) -> None:
    if not parts:
        raise ValueError(f"{where}: no parts")
    for p in parts:
        t = p.get("type")
        if t not in PART_TYPES:
            raise ValueError(f"{where}: unknown part type {t!r} (expected {PART_TYPES})")
        need = {"box": ("size",), "cylinder": ("radius", "depth"), "cone": ("radius_bottom", "radius_top", "depth"),
                "sphere": (), "plane": ("size",)}[t]
        for k in need:
            if k not in p:
                raise ValueError(f"{where}: {t} part needs {k!r}")
        if t == "sphere" and "radius" not in p and "size" not in p:
            raise ValueError(f"{where}: sphere part needs 'radius' or 'size'")


def validate_spec(spec: dict) -> None:
    if not isinstance(spec.get("objects"), list) or not spec["objects"]:
        raise ValueError("spec needs a non-empty 'objects' list")
    names, ids = set(), set()
    for o in spec["objects"]:
        for key in ("name", "id", "parts"):
            if key not in o:
                raise ValueError(f"object {o.get('name')!r} missing {key!r}")
        if o["name"] in names:
            raise ValueError(f"duplicate name {o['name']!r}")
        if not isinstance(o["id"], int) or o["id"] < 1 or o["id"] in ids:
            raise ValueError(f"bad or duplicate id {o['id']!r} ({o['name']})")
        validate_parts(o["parts"], f"object {o['name']!r}")
        names.add(o["name"])
        ids.add(o["id"])
    cam = spec.get("camera")
    if not cam or "location" not in cam or ("look_at" not in cam and "rotation" not in cam):
        raise ValueError("spec 'camera' needs 'location' and 'look_at' or 'rotation'")


def euler_deg(r) -> Euler:
    r = r or (0, 0, 0)
    return Euler([math.radians(a) for a in r], "XYZ")


def add_part(bm: bmesh.types.BMesh, part: dict) -> None:
    t = part["type"]
    before = set(bm.verts)
    if t == "box":
        sx, sy, sz = part["size"]
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Diagonal((sx, sy, sz, 1.0)))
    elif t in ("cylinder", "cone"):
        if t == "cylinder":
            r1 = r2 = part["radius"]
        else:
            r1, r2 = part["radius_bottom"], part["radius_top"]
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=int(part.get("segments", 32)),
                              radius1=r1, radius2=r2, depth=part["depth"])
    elif t == "sphere":
        if "size" in part:
            dx, dy, dz = part["size"]
        else:
            dx = dy = dz = 2.0 * part["radius"]
        bmesh.ops.create_uvsphere(bm, u_segments=int(part.get("segments", 32)),
                                  v_segments=int(part.get("rings", 16)), radius=0.5,
                                  matrix=Matrix.Diagonal((dx, dy, dz, 1.0)))
    elif t == "plane":
        sx, sy = part["size"]
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5,
                              matrix=Matrix.Diagonal((sx, sy, 1.0, 1.0)))
    else:
        raise ValueError(f"unknown part type {t!r}")

    verts = [v for v in bm.verts if v not in before]
    faces = list({f for v in verts for f in v.link_faces})
    if t != "plane":
        bmesh.ops.recalc_face_normals(bm, faces=faces)

    bevel = float(part.get("bevel", 0.0) or 0.0)
    if bevel > 0 and t != "plane":
        edges = {e for v in verts for e in v.link_edges}
        sharp = [e for e in edges if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(30)]
        if sharp:
            bmesh.ops.bevel(bm, geom=sharp, offset=bevel, offset_type="OFFSET",
                            segments=int(part.get("bevel_segments", 2)),
                            profile=0.5, affect="EDGES", clamp_overlap=True)
            verts = [v for v in bm.verts if v not in before]

    m = Matrix.Translation(Vector(part.get("location", (0, 0, 0)))) @ euler_deg(part.get("rotation")).to_matrix().to_4x4()
    bmesh.ops.transform(bm, matrix=m, verts=verts)


def mesh_from_parts(name: str, parts: list, rotation=None) -> bpy.types.Mesh:
    """Parts in object-local coordinates -> one mesh, object rotation baked in, flat shaded."""
    bm = bmesh.new()
    for part in parts:
        add_part(bm, part)
    rot = euler_deg(rotation).to_matrix().to_4x4()
    bmesh.ops.transform(bm, matrix=rot, verts=bm.verts[:])
    bm.normal_update()
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = False
    return mesh


def tag_object(obj, oid: int, category: str, description: str) -> None:
    obj.pass_index = int(oid)
    obj["b2r_id"] = int(oid)
    obj["category"] = str(category)
    obj["description"] = str(description)


def collection_for(scene, category: str):
    name = f"b2r_{category}" if category else "b2r_object"
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        scene.collection.children.link(col)
    return col


def build_spec_object(o: dict, material, scene) -> bpy.types.Object:
    mesh = mesh_from_parts(o["name"], o["parts"], o.get("rotation"))
    mesh.materials.append(material)
    obj = bpy.data.objects.new(o["name"], mesh)
    if obj.name != o["name"]:
        raise ValueError(f"object name {o['name']!r} is already taken in the scene")
    obj.location = Vector(o.get("location", (0, 0, 0)))
    tag_object(obj, o["id"], o.get("category", "object"), o.get("description", ""))
    collection_for(scene, obj["category"]).objects.link(obj)
    return obj


def grey_material():
    mat = bpy.data.materials.get(GREY_NAME)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(GREY_NAME)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = GREY
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Metallic"].default_value = 0.0
    mat.diffuse_color = GREY
    return mat


def set_grey(obj, mat) -> None:
    me = obj.data
    me.materials.clear()
    me.materials.append(mat)
    if len(me.polygons):
        me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), dtype=np.int32))
    me.update()


# =========================================================================== UVs
def _select_only(obj) -> None:
    vl = bpy.context.view_layer
    for o in bpy.context.scene.objects:
        if o is not None and o.name in vl.objects:
            o.select_set(False)
    vl.objects.active = obj
    obj.select_set(True)


def uv_unwrap(obj, method: str = "smart") -> None:
    """(Re)make the b2r_uv map: non-overlapping islands in [0,1] with a margin. Other UV maps are
    left untouched; b2r_uv becomes the active and render-active map."""
    me = obj.data
    uv = me.uv_layers.get(UV_NAME) or me.uv_layers.new(name=UV_NAME)
    if uv is None:
        raise RuntimeError(f"{obj.name}: cannot add a UV map (limit of 8 maps reached?)")
    me.uv_layers.active = uv
    uv.active_render = True
    _select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    if method == "smart":
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.01,
                                 area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(rotate=True, shape_method="AABB", margin_method="FRACTION", margin=0.006)
    elif method == "lightmap":
        bpy.ops.uv.lightmap_pack(PREF_CONTEXT="ALL_FACES", PREF_PACK_IN_ONE=True, PREF_NEW_UVLAYER=False,
                                 PREF_MARGIN_DIV=0.3)
    else:
        raise ValueError(method)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)


def uv_stats(obj, res: int = 1024) -> dict:
    """Islands of b2r_uv, range, and raster overlap between different islands."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    uvl = bm.loops.layers.uv.get(UV_NAME)
    if uvl is None:
        bm.free()
        return {"has_b2r_uv": False}
    bm.faces.ensure_lookup_table()
    parent = list(range(len(bm.faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def key(loop):
        u = loop[uvl].uv
        return (round(u.x, 5), round(u.y, 5))

    for e in bm.edges:
        lf = e.link_loops
        if len(lf) != 2:
            continue
        l1, l2 = lf
        if key(l1) == key(l2.link_loop_next) and key(l1.link_loop_next) == key(l2):
            a, b = find(l1.face.index), find(l2.face.index)
            if a != b:
                parent[a] = b
    island = [find(f.index) for f in bm.faces]
    uniq = {r: i for i, r in enumerate(sorted(set(island)))}

    tris, allu = [], []
    for f in bm.faces:
        uvs = [tuple(l[uvl].uv) for l in f.loops]
        allu.extend(uvs)
        for k in range(1, len(uvs) - 1):
            tris.append((uniq[island[f.index]], uvs[0], uvs[k], uvs[k + 1]))
    bm.free()
    if not allu:
        return {"has_b2r_uv": True, "n_islands": 0, "uv_in_unit_square": True, "uv_area_fraction": 0.0,
                "uv_min": [0, 0], "uv_max": [0, 0], "inter_island_overlap_px": 0, "raster_res": res}
    allu = np.array(allu)
    owner = np.full((res, res), -1, dtype=np.int32)
    conflict = np.zeros((res, res), dtype=bool)
    area = 0.0
    for isl, a, b, c in tris:
        a, b, c = np.array(a), np.array(b), np.array(c)
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        area += abs(den) / 2
        if abs(den) < 1e-14:
            continue
        lo = np.clip(np.floor(np.minimum(np.minimum(a, b), c) * res - 0.5).astype(int), 0, res - 1)
        hi = np.clip(np.ceil(np.maximum(np.maximum(a, b), c) * res - 0.5).astype(int), 0, res - 1)
        xs = (np.arange(lo[0], hi[0] + 1) + 0.5) / res
        ys = (np.arange(lo[1], hi[1] + 1) + 0.5) / res
        X, Y = np.meshgrid(xs, ys)
        w1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / den
        w2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / den
        inside = (w1 >= 0) & (w2 >= 0) & (1 - w1 - w2 >= 0)
        if not inside.any():
            continue
        sl = (slice(lo[1], hi[1] + 1), slice(lo[0], hi[0] + 1))
        own = owner[sl]
        conflict[sl] |= inside & (own >= 0) & (own != isl)
        own[inside & (own < 0)] = isl
    return {
        "has_b2r_uv": True,
        "n_islands": len(uniq),
        "uv_min": [float(x) for x in allu.min(0)],
        "uv_max": [float(x) for x in allu.max(0)],
        "uv_in_unit_square": bool((allu >= -1e-6).all() and (allu <= 1 + 1e-6).all()),
        "uv_area_fraction": round(float(area), 4),
        "raster_res": res,
        "raster_coverage_fraction": round(float((owner >= 0).mean()), 4),
        "inter_island_overlap_px": int(conflict.sum()),
    }


def unwrap_checked(obj, tag: str) -> dict:
    """b2r_uv with a verified result; falls back to lightmap packing if Smart UV overlaps.
    Raises if the map still overlaps or leaves [0,1]."""
    t0 = time.time()
    uv_unwrap(obj, "smart")
    st = uv_stats(obj)
    st["method"] = "smart_project+pack_islands"
    if st["inter_island_overlap_px"] or not st["uv_in_unit_square"]:
        log(tag, f"NOTE {obj.name}: smart UV overlap {st['inter_island_overlap_px']} px; retrying with lightmap pack")
        uv_unwrap(obj, "lightmap")
        st = uv_stats(obj)
        st["method"] = "lightmap_pack"
    if st["inter_island_overlap_px"] or not st["uv_in_unit_square"]:
        raise RuntimeError(f"{obj.name}: b2r_uv still overlaps ({st['inter_island_overlap_px']} px) or leaves [0,1]")
    log(tag, f"uv {obj.name}: {st['n_islands']} islands, {time.time() - t0:.1f}s")
    return st


# =========================================================================== geometry facts
def world_verts(obj) -> np.ndarray:
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    M = np.array(obj.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


def geometry_sha256(obj) -> str:
    """Checksum of what downstream stages see of an object: vertex positions, faces, the b2r_uv
    map and the world matrix. Identical checksum = identical geometry."""
    me = obj.data
    h = hashlib.sha256()
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    h.update(co.tobytes())
    for attr, n, dt in (("loop_start", len(me.polygons), np.int32), ("loop_total", len(me.polygons), np.int32)):
        a = np.empty(n, dtype=dt)
        me.polygons.foreach_get(attr, a)
        h.update(a.tobytes())
    lv = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    h.update(lv.tobytes())
    uv = me.uv_layers.get(UV_NAME)
    if uv is not None:
        d = np.empty(len(me.loops) * 2, dtype=np.float32)
        uv.data.foreach_get("uv", d)
        h.update(d.tobytes())
    h.update(np.array(obj.matrix_world, dtype=np.float64).tobytes())
    return h.hexdigest()


def surface_area(obj) -> float:
    me = obj.data
    if not len(me.polygons):
        return 0.0
    a = np.empty(len(me.polygons), dtype=np.float64)
    me.polygons.foreach_get("area", a)
    s = np.abs(np.linalg.det(np.array(obj.matrix_world)[:3, :3])) ** (2.0 / 3.0)
    return float(a.sum() * s)


def bvh_of(obj) -> BVHTree:
    co = world_verts(obj)
    polys = [tuple(p.vertices) for p in obj.data.polygons]
    return BVHTree.FromPolygons([tuple(v) for v in co], polys)


def penetration(obj_a, bvh_b, tol=2e-3) -> int:
    """number of vertices of A that lie more than tol inside closed mesh B."""
    n = 0
    for p in world_verts(obj_a):
        v = Vector(p)
        hit = bvh_b.find_nearest(v)
        if hit[0] is None:
            continue
        loc, normal, _, dist = hit
        if dist > tol and (v - loc).dot(normal) < 0:
            inside_votes = 0
            for d in (Vector((0, 0, 1)), Vector((1, 0, 0)), Vector((0.3, 0.7, 0.2)).normalized()):
                hits, o = 0, v.copy()
                for _ in range(64):
                    h = bvh_b.ray_cast(o, d)
                    if h[0] is None:
                        break
                    hits += 1
                    o = h[0] + d * 1e-5
                inside_votes += hits % 2
            if inside_votes >= 2:
                n += 1
    return n


def mesh_objects(scene) -> list:
    """Every renderable mesh object of the scene, in id order."""
    objs = [o for o in scene.objects if o.type == "MESH" and not o.hide_render]
    return sorted(objs, key=lambda o: (o.pass_index, o.name))


def object_row(obj, uv: dict, extra: dict | None = None) -> dict:
    w = world_verts(obj)
    lo, hi = (w.min(0), w.max(0)) if len(w) else (np.zeros(3), np.zeros(3))
    row = {
        "name": obj.name, "id": int(obj.pass_index),
        "category": str(obj.get("category", "object")),
        "description": str(obj.get("description", "")),
        "bbox_min": [round(float(x), 4) for x in lo],
        "bbox_max": [round(float(x), 4) for x in hi],
        "n_verts": len(obj.data.vertices), "n_faces": len(obj.data.polygons),
        "n_uv_islands": uv.get("n_islands"), "uv_area_fraction": uv.get("uv_area_fraction"),
        "uv_min": uv.get("uv_min"), "uv_max": uv.get("uv_max"),
        "uv_inter_island_overlap_px": uv.get("inter_island_overlap_px"),
        "uv_method": uv.get("method"),
        "geometry_sha256": geometry_sha256(obj),
    }
    hints = obj.get("b2r_hints")
    if hints:
        row.update(json.loads(hints))
    if extra:
        row.update(extra)
    return row


def verify(scene, cam, objs: list, uv: dict, room: dict | None = None, check_penetration: bool = True) -> dict:
    """Automatic checks on the built scene. objs: mesh objects; uv: name -> uv_stats."""
    report = {}
    ids = [o.pass_index for o in objs]
    report["n_objects"] = len(objs)
    report["pass_index_unique"] = len(ids) == len(set(ids))
    report["pass_index_all_ge_1"] = all(i >= 1 for i in ids)
    report["all_scale_one"] = all(all(abs(s - 1) < 1e-9 for s in o.scale) for o in objs)
    report["all_rotation_zero"] = all(all(abs(r) < 1e-9 for r in o.rotation_euler) for o in objs)
    report["no_parents"] = all(o.parent is None for o in objs)
    report["uv_b2r_on_every_object"] = all(uv[o.name].get("has_b2r_uv") for o in objs)
    report["uv_all_in_unit_square"] = all(uv[o.name].get("uv_in_unit_square", False) for o in objs)
    report["uv_total_inter_island_overlap_px"] = int(sum(uv[o.name].get("inter_island_overlap_px", 0) or 0 for o in objs))
    report["uv_overlap_by_object"] = {o.name: uv[o.name].get("inter_island_overlap_px") for o in objs
                                      if uv[o.name].get("inter_island_overlap_px")}
    cats: dict = {}
    for o in objs:
        c = str(o.get("category", "object"))
        cats[c] = cats.get(c, 0) + 1
    report["categories"] = cats
    report["faces_total"] = int(sum(len(o.data.polygons) for o in objs))

    bbox = {}
    for o in objs:
        w = world_verts(o)
        bbox[o.name] = (w.min(0), w.max(0))

    in_frame = []
    nonstruct = [o for o in objs if o.get("category") != "structure"]
    for o in nonstruct:
        lo, hi = bbox[o.name]
        c = world_to_camera_view(scene, cam, Vector(((lo + hi) / 2).tolist()))
        ok = 0 <= c.x <= 1 and 0 <= c.y <= 1 and c.z > 0
        in_frame.append({"name": o.name, "ndc": [round(c.x, 3), round(c.y, 3)], "depth": round(c.z, 3),
                         "in_frame": ok})
    report["non_structure_count"] = len(nonstruct)
    report["non_structure_centres_in_frame"] = sum(f["in_frame"] for f in in_frame)
    report["in_frame"] = in_frame

    nverts = sum(len(o.data.vertices) for o in objs)
    pen = []
    if check_penetration and nverts <= PENETRATION_VERT_LIMIT:
        bvhs = {o.name: bvh_of(o) for o in objs}
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                la, ha = bbox[a.name]
                lb, hb = bbox[b.name]
                if (la > hb + 0.01).any() or (lb > ha + 0.01).any():
                    continue
                nab = penetration(a, bvhs[b.name])
                nba = penetration(b, bvhs[a.name])
                if nab or nba:
                    pen.append({"a": a.name, "b": b.name, "verts_a_in_b": nab, "verts_b_in_a": nba})
        report["interpenetrating_pairs"] = pen
    else:
        report["interpenetrating_pairs"] = None
        report["interpenetration_skipped"] = (f"{nverts} vertices > {PENETRATION_VERT_LIMIT}" if check_penetration
                                              else "not requested")

    support = []
    tops = {n: bb[1][2] for n, bb in bbox.items()}
    for o in nonstruct:
        lo, hi = bbox[o.name]
        under = [n for n in bbox if n != o.name
                 and bbox[n][0][0] <= hi[0] and bbox[n][1][0] >= lo[0]
                 and bbox[n][0][1] <= hi[1] and bbox[n][1][1] >= lo[1]
                 and tops[n] <= lo[2] + 1e-4]
        sz = max([tops[n] for n in under], default=None)
        row = {"name": o.name, "min_z": round(float(lo[2]), 4),
               "highest_support_below_top_z": None if sz is None else round(float(sz), 4),
               "gap_to_support": None if sz is None else round(float(lo[2] - sz), 4)}
        if room and "width_x" in room and "depth_y" in room:
            W, D = room["width_x"], room["depth_y"]
            row["min_gap_to_wall_planes"] = round(float(min(lo[0], W - hi[0], lo[1], D - hi[1])), 4)
        support.append(row)
    report["support"] = support
    return report


# =========================================================================== camera / size / lights / render
def camera_aspect(scene) -> tuple[float, float]:
    r = scene.render
    return (r.resolution_x * r.pixel_aspect_x, r.resolution_y * r.pixel_aspect_y)


def apply_case_size(case: str, scene, camera_wh, source: str, camera_name: str) -> dict:
    """Choose the case size once (case.decide_size), record the camera, set the render size."""
    cfg = common.load_case(case)
    size = case_mod.decide_size(cfg, camera_wh, source)
    cfg["camera"] = camera_name
    common.save_case(case, cfg)
    scene.render.resolution_x = int(size["width"])
    scene.render.resolution_y = int(size["height"])
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    return size


def aim(obj, target) -> None:
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler("XYZ")


def setup_spec_camera(spec: dict, scene) -> bpy.types.Object:
    c = spec["camera"]
    cd = bpy.data.cameras.new(c.get("name", "Camera"))
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = float(c.get("sensor_width_mm", 36.0))
    if "focal_length_mm" in c:
        cd.lens = float(c["focal_length_mm"])
    elif "fov_deg" in c:
        cd.lens = cd.sensor_width / (2 * math.tan(math.radians(c["fov_deg"]) / 2))
    else:
        cd.lens = 35.0
    cd.clip_start = float(c.get("clip_start", 0.05))
    cd.clip_end = float(c.get("clip_end", 100.0))
    cam = bpy.data.objects.new(c.get("name", "Camera"), cd)
    scene.collection.objects.link(cam)
    cam.location = Vector(c["location"])
    if "look_at" in c:
        aim(cam, c["look_at"])
    else:
        cam.rotation_euler = euler_deg(c["rotation"])
    scene.camera = cam
    return cam


def spec_camera_aspect(spec: dict):
    c = spec["camera"]
    if "resolution" in c:
        return float(c["resolution"][0]), float(c["resolution"][1])
    if "aspect_ratio" in c:
        return case_mod.parse_aspect(c["aspect_ratio"]), 1.0
    return None


def add_light(scene, d: dict, tag: str) -> bpy.types.Object:
    lt = d.get("type", "POINT").upper()
    ld = bpy.data.lights.new(d.get("name", lt.lower()), lt)
    ld.energy = float(d.get("energy", 100.0))
    ld.color = tuple(d.get("color", (1.0, 1.0, 1.0)))[:3]
    if lt == "AREA":
        ld.shape = d.get("shape", "SQUARE")
        ld.size = float(d.get("size", 1.0))
        ld.size_y = float(d.get("size_y", d.get("size", 1.0)))
    elif lt in ("POINT", "SPOT"):
        ld.shadow_soft_size = float(d.get("size", 0.1))
        if lt == "SPOT":
            ld.spot_size = math.radians(float(d.get("spot_size_deg", 45.0)))
    elif lt == "SUN":
        ld.angle = math.radians(float(d.get("angle_deg", 2.0)))
    ob = bpy.data.objects.new(d.get("name", lt.lower()), ld)
    scene.collection.objects.link(ob)
    ob.location = Vector(d.get("location", (0, 0, 0)))
    if "look_at" in d:
        aim(ob, d["look_at"])
    elif "direction" in d:
        ob.rotation_euler = Vector(d["direction"]).normalized().to_track_quat("-Z", "Y").to_euler()
    elif "rotation" in d:
        ob.rotation_euler = euler_deg(d["rotation"])
    ob[tag] = True
    return ob


def set_world(scene, color, strength: float, name: str):
    world = bpy.data.worlds.new(name)
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*color[:3], 1.0)
    bg.inputs["Strength"].default_value = float(strength)
    scene.world = world
    return world


def add_preview_lights(scene, pl: dict | None, room: dict | None = None, sun_dir=None) -> list[str]:
    """Neutral placeholder lighting (named preview_*, tagged b2r_preview_light); later stages fit
    real lights. Used only when the scene has no lights of its own."""
    pl = pl or {}
    added = []
    sun = add_light(scene, {"name": "preview_sun", "type": "SUN", "energy": float(pl.get("sun_strength", 4.0)),
                            "direction": pl.get("sun_direction", sun_dir or (-0.35, -1.0, -0.55)),
                            "angle_deg": 2.0}, TAG_PREVIEW_LIGHT)
    added.append(f"sun {sun.data.energy} W/m2")
    if room and all(k in room for k in ("width_x", "depth_y", "height_z")):
        a = add_light(scene, {"name": "preview_ceiling", "type": "AREA", "shape": "RECTANGLE", "size": 1.2,
                              "size_y": 1.2, "energy": float(pl.get("ceiling_area_watts", 150.0)),
                              "location": (room["width_x"] / 2, room["depth_y"] / 2, room["height_z"] - 0.02)},
                      TAG_PREVIEW_LIGHT)
        added.append(f"ceiling area light {a.data.energy} W at {tuple(round(x, 3) for x in a.location)}")
    sky = pl.get("sky_color", (0.55, 0.7, 1.0))
    set_world(scene, sky, float(pl.get("sky_strength", 1.0)), "preview_world")
    added.append(f"world colour {tuple(sky)} strength {pl.get('sky_strength', 1.0)}")
    return added


def setup_cycles(scene, samples: int, tag: str) -> str:
    scene.render.engine = "CYCLES"
    device = "CPU"
    try:
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
                device = f"GPU ({backend}): " + ", ".join(names)
                break
    except Exception as exc:  # pragma: no cover
        log(tag, f"NOTE GPU setup failed, using CPU: {exc}")
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    return device


def save_blend(path: Path) -> None:
    bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=False)


def render_preview(scene, path: Path, tag: str) -> float:
    scene.render.filepath = str(path)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    if not Path(path).is_file():
        raise RuntimeError(f"preview render did not write {path}")
    dt = time.time() - t0
    log(tag, f"preview {path.name}: {dt:.1f}s")
    return dt


def light_summary(scene) -> list[dict]:
    out = []
    for o in scene.objects:
        if o.type == "LIGHT" and not o.hide_render:
            out.append({"name": o.name, "type": o.data.type, "energy": round(float(o.data.energy), 4),
                        "color": [round(float(c), 4) for c in o.data.color],
                        "location": [round(float(x), 4) for x in o.matrix_world.translation],
                        "input_light": bool(o.get(TAG_INPUT_LIGHT, False)),
                        "preview_light": bool(o.get(TAG_PREVIEW_LIGHT, False))})
    return out


def camera_summary(scene, cam) -> dict:
    return {"name": cam.name, "location": [round(float(x), 5) for x in cam.matrix_world.translation],
            "rotation_deg": [round(math.degrees(a), 4) for a in cam.matrix_world.to_euler("XYZ")],
            "focal_length_mm": cam.data.lens, "sensor_width_mm": cam.data.sensor_width,
            "sensor_fit": cam.data.sensor_fit,
            "hfov_deg": round(math.degrees(cam.data.angle_x), 4), "vfov_deg": round(math.degrees(cam.data.angle_y), 4),
            "resolution": [scene.render.resolution_x, scene.render.resolution_y]}


def write_outputs(out: Path, scene, cam, objs: list, uv: dict, report: dict, samples: int, render: bool,
                  tag: str) -> None:
    """objects.json, verification.json, blockout.blend, preview.png (in that order)."""
    rows = [object_row(o, uv[o.name]) for o in objs]
    report["camera"] = camera_summary(scene, cam)
    report["lights"] = light_summary(scene)
    report["render_device"] = setup_cycles(scene, samples, tag)
    report["max_id"] = max([r["id"] for r in rows], default=0)
    common.write_json(out / "objects.json", rows)
    common.write_json(out / "verification.json", report)
    save_blend(out / "blockout.blend")
    if render:
        report["preview_seconds"] = round(render_preview(scene, out / "preview.png", tag), 1)
        common.write_json(out / "verification.json", report)


def print_summary(tag: str, report: dict, out: Path) -> None:
    keys = ("n_objects", "pass_index_unique", "all_scale_one", "all_rotation_zero", "uv_b2r_on_every_object",
            "uv_all_in_unit_square", "uv_total_inter_island_overlap_px", "non_structure_count",
            "non_structure_centres_in_frame", "categories")
    log(tag, "summary " + ", ".join(f"{k}={report.get(k)}" for k in keys))
    pen = report.get("interpenetrating_pairs")
    if pen:
        log(tag, f"NOTE interpenetrating pairs: {[(p['a'], p['b']) for p in pen]}")
    log(tag, "wrote", out)


# =========================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("--spec", default=None, help="spec JSON (default: the case input)")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(common.script_args())
    tag = "blockout"

    cfg = common.load_case(args.case)
    if args.spec:
        spec_path = Path(args.spec).expanduser().resolve()
    else:
        if cfg["input"]["kind"] != "spec":
            raise SystemExit(f"case input is a {cfg['input']['kind']}, not a spec: use ingest.py, or pass --spec")
        spec_path = case_mod.input_path(args.case, cfg)
    spec = common.read_json(spec_path)
    validate_spec(spec)
    out = common.stage_dir(args.case, "blockout", args.iter)
    if spec_path.resolve() != (out / "scene_spec.json").resolve():
        shutil.copyfile(spec_path, out / "scene_spec.json")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    mat = grey_material()

    objs = [build_spec_object(o, mat, scene) for o in spec["objects"]]
    uv = {o.name: unwrap_checked(o, tag) for o in objs}

    cam = setup_spec_camera(spec, scene)
    size = apply_case_size(args.case, scene, spec_camera_aspect(spec), "spec camera", cam.name)
    log(tag, f"size {size['width']}x{size['height']} {size['aspect_ratio']} ({size.get('aspect_source', 'case.json')})")

    room = spec.get("room")
    if spec.get("lights"):
        lights = [add_light(scene, d, TAG_INPUT_LIGHT).name for d in spec["lights"]]
        light_note = f"spec lights kept and tagged {TAG_INPUT_LIGHT}: {lights}"
        if spec.get("world"):
            set_world(scene, spec["world"].get("color", (0.05, 0.05, 0.05)), spec["world"].get("strength", 1.0),
                      "spec_world")
    else:
        light_note = "no spec lights: preview lighting " + "; ".join(
            add_preview_lights(scene, spec.get("preview_lighting"), room))
        if spec.get("world"):
            set_world(scene, spec["world"].get("color", (0.05, 0.05, 0.05)), spec["world"].get("strength", 1.0),
                      "spec_world")
    log(tag, light_note)
    bpy.context.view_layer.update()

    t0 = time.time()
    report = verify(scene, cam, objs, uv, room=room)
    report["verify_seconds"] = round(time.time() - t0, 1)
    report["source"] = {"kind": "spec", "spec": spec_path.name, "spec_sha256": case_mod.sha256_file(spec_path)}
    report["pass_index_matches_spec"] = all(o.pass_index == s["id"] for o, s in zip(objs, spec["objects"]))
    report["lighting"] = light_note
    report["size"] = size
    write_outputs(out, scene, cam, objs, uv, report, args.samples, not args.no_render, tag)
    print_summary(tag, report, out)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
