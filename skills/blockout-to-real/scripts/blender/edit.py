"""edit: geometry edits on blockout iteration K -> blockout iteration K+1.

    blender -b --factory-startup --python scripts/blender/edit.py -- \
        --case CASE --iter K (--ops OPS.json | --script EDIT.py) [--samples N] [--no-render]

Opens CASE/blockout/iter<K>/blockout.blend, applies the edits, writes CASE/blockout/iter<K+1>/:
blockout.blend, objects.json, verification.json, preview.png, edits.json.

OPS.json: a list of operations, applied in order (lengths metres, angles degrees, Z up):
  {"op": "add", "name": "lamp", "category": "object", "description": "...",
   "location": [x,y,z], "rotation": [rx,ry,rz], "parts": [...]}
        new object from primitive parts (same part schema as the scene spec, see blockout.py).
        Its id is the next one after the highest id ever used in this case ("id" may be given if
        it is higher still). The rotation is baked into the mesh.
  {"op": "remove", "name": "..."}
  {"op": "transform", "name": "...", "location": [x,y,z] | "translate": [dx,dy,dz],
   "rotate": [rx,ry,rz], "scale": s | [sx,sy,sz]}
        rotate / scale act about the object origin and are baked into the mesh (the b2r_uv map
        stays valid: UVs do not change).
  {"op": "replace_parts", "name": "...", "parts": [...], "location"?: [...], "rotation"?: [...]}
        new geometry for an existing object; keeps its name, id, category and description.
  {"op": "set", "name": "...", "description"?: "...", "category"?: "..."}

--script EDIT.py runs a bpy script with the scene open (globals: bpy, scene, case, iteration), for
edits primitives cannot express. Afterwards the same bookkeeping runs: new renderable objects are
converted to meshes, get grey material, b2r_uv and new ids (a duplicated object's copied id is
replaced); objects whose mesh changed get a new b2r_uv; rotation/scale are applied; descriptions and
categories may be set by the script as custom properties "description" / "category".

Guarantees (checked; any violation -> exit 1): ids unique, new ids > every id used before in the
case, never reused; every object has a non-overlapping b2r_uv; every object the edits did not touch
has the same geometry_sha256 (vertices, faces, b2r_uv, world matrix) as in iteration K.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import blockout as B  # noqa: E402
from b2r import common  # noqa: E402
from b2r import case as case_mod  # noqa: E402

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

TAG = "edit"


def max_id_ever(case: str) -> int:
    m = 0
    for p in (common.case_dir(case) / "blockout").glob("iter*/objects.json"):
        m = max([m] + [int(r["id"]) for r in common.read_json(p)])
    for p in (common.case_dir(case) / "blockout").glob("iter*/edits.json"):
        e = common.read_json(p)
        m = max([m] + [int(r["id"]) for r in e.get("removed", [])] + [int(r["id"]) for r in e.get("added", [])])
    return m


def get_obj(scene, name: str):
    o = scene.objects.get(name)
    if o is None or o.type != "MESH":
        raise ValueError(f"no mesh object named {name!r}")
    return o


def bake(o, M: Matrix) -> None:
    o.data.transform(M)
    if M.determinant() < 0:
        o.data.flip_normals()
    o.data.update()


def apply_ops(scene, ops: list, next_id: int, log: dict) -> tuple[set, int]:
    touched = set()
    mat = B.grey_material()
    for i, op in enumerate(ops):
        kind = op.get("op")
        where = f"op {i} ({kind})"
        if kind == "add":
            B.validate_parts(op.get("parts"), where)
            if scene.objects.get(op["name"]):
                raise ValueError(f"{where}: name {op['name']!r} exists")
            oid = int(op.get("id", next_id))
            if oid < next_id:
                raise ValueError(f"{where}: id {oid} would reuse an id (next free: {next_id})")
            o = B.build_spec_object({**op, "id": oid, "category": op.get("category", "object")}, mat, scene)
            next_id = oid + 1
            touched.add(o.name)
            log["added"].append({"name": o.name, "id": oid})
        elif kind == "remove":
            o = get_obj(scene, op["name"])
            log["removed"].append({"name": o.name, "id": int(o.pass_index)})
            bpy.data.objects.remove(o, do_unlink=True)
        elif kind == "transform":
            o = get_obj(scene, op["name"])
            if "location" in op:
                o.location = Vector(op["location"])
            if "translate" in op:
                o.location = o.location + Vector(op["translate"])
            M = Matrix.Identity(4)
            if "rotate" in op:
                M = B.euler_deg(op["rotate"]).to_matrix().to_4x4() @ M
            if "scale" in op:
                s = op["scale"]
                s = [s, s, s] if isinstance(s, (int, float)) else s
                M = Matrix.Diagonal((*s, 1.0)) @ M
            bake(o, M)
            touched.add(o.name)
            log["transformed"].append({"name": o.name, **{k: op[k] for k in op if k not in ("op", "name")}})
        elif kind == "replace_parts":
            B.validate_parts(op.get("parts"), where)
            o = get_obj(scene, op["name"])
            old = o.data
            o.data = B.mesh_from_parts(o.name, op["parts"], op.get("rotation"))
            o.data.materials.append(mat)
            bpy.data.meshes.remove(old)
            if "location" in op:
                o.location = Vector(op["location"])
            touched.add(o.name)
            log["replaced"].append({"name": o.name, "id": int(o.pass_index), "parts": len(op["parts"])})
        elif kind == "set":
            o = get_obj(scene, op["name"])
            for k in ("description", "category"):
                if k in op:
                    o[k] = str(op[k])
            log["set"].append({"name": o.name, **{k: op[k] for k in ("description", "category") if k in op}})
        else:
            raise ValueError(f"{where}: unknown op {kind!r}")
    return touched, next_id


def run_script(scene, path: Path, case: str, k: int, before: dict, next_id: int, log: dict) -> tuple[set, int]:
    code = path.read_text()
    exec(compile(code, str(path), "exec"), {"bpy": bpy, "scene": scene, "case": case, "iteration": k,
                                             "__name__": "__b2r_edit__"})
    bpy.context.view_layer.update()
    touched = set()
    mat = B.grey_material()
    dg = bpy.context.evaluated_depsgraph_get()
    # non-mesh renderables (curves, text, ...) and modifiers -> plain meshes
    for o in list(scene.objects):
        if o.hide_render or o.type in ("LIGHT", "CAMERA", "EMPTY"):
            continue
        if o.type != "MESH" or len(o.modifiers):
            me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), depsgraph=dg)
            M = o.matrix_world.copy()
            name = o.name
            props = {k: o[k] for k in ("description", "category", "b2r_hints") if k in o}
            bpy.data.objects.remove(o, do_unlink=True)
            n = bpy.data.objects.new(name, me)
            scene.collection.objects.link(n)
            n.matrix_world = M
            for kk, v in props.items():
                n[kk] = v
            touched.add(n.name)
    seen_ids = set()
    for o in B.mesh_objects(scene):
        if o.parent is not None:
            M = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = M
        loc = o.matrix_world.translation.copy()
        R = o.matrix_world.to_3x3().to_4x4()
        if any(abs(R[i][j] - (i == j)) > 1e-9 for i in range(3) for j in range(3)):
            bake(o, R)
            o.matrix_world = Matrix.Translation(loc)
            touched.add(o.name)
        b = before.get(o.name)
        is_new = b is None or o.pass_index != b["id"] or o.pass_index in seen_ids or o.pass_index < 1
        if is_new:
            B.tag_object(o, next_id, o.get("category", "object"), o.get("description", B_clean(o.name)))
            log["added"].append({"name": o.name, "id": next_id})
            next_id += 1
            touched.add(o.name)
        seen_ids.add(o.pass_index)
        if is_new or o.name in touched or B.geometry_sha256(o) != b["geometry_sha256"]:
            if not is_new and o.name not in touched:
                log["changed_by_script"].append(o.name)
            touched.add(o.name)
        if o.name in touched:
            B.set_grey(o, mat)
    for name, b in before.items():
        if scene.objects.get(name) is None:
            log["removed"].append({"name": name, "id": b["id"]})
    return touched, next_id


def B_clean(name: str) -> str:
    return name.replace("_", " ").replace(".", " ").strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ops")
    g.add_argument("--script")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(common.script_args())
    t0 = time.time()

    src = common.stage_dir(args.case, "blockout", args.iter, create=False)
    blend = src / "blockout.blend"
    if not blend.is_file():
        raise SystemExit(f"no {blend}")
    out_dir = common.case_dir(args.case) / "blockout" / f"iter{args.iter + 1}"
    if (out_dir / "blockout.blend").exists():
        raise SystemExit(f"{out_dir} already has a blockout; remove it first to redo the edit")
    before = {r["name"]: r for r in common.read_json(src / "objects.json")}
    next_id = max_id_ever(args.case) + 1

    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False)
    scene = bpy.context.scene
    log = {"from_iter": args.iter, "to_iter": args.iter + 1, "added": [], "removed": [], "transformed": [],
           "replaced": [], "set": [], "changed_by_script": []}
    if args.ops:
        ops_path = Path(args.ops).expanduser().resolve()
        ops = common.read_json(ops_path)
        if isinstance(ops, dict):
            ops = ops.get("ops", [])
        log["ops_file"] = {"name": ops_path.name, "sha256": case_mod.sha256_file(ops_path)}
        log["ops"] = ops
        touched, next_id = apply_ops(scene, ops, next_id, log)
    else:
        sp = Path(args.script).expanduser().resolve()
        log["script"] = {"name": sp.name, "sha256": case_mod.sha256_file(sp)}
        touched, next_id = run_script(scene, sp, args.case, args.iter, before, next_id, log)

    out = common.stage_dir(args.case, "blockout", args.iter + 1)
    bpy.context.view_layer.update()
    objs = B.mesh_objects(scene)
    uv = {}
    for o in objs:
        if o.name in touched:
            if o.name not in {r["name"] for r in log["transformed"]} or o.data.uv_layers.get(B.UV_NAME) is None:
                uv[o.name] = B.unwrap_checked(o, TAG)
            else:
                uv[o.name] = B.uv_stats(o)
                uv[o.name]["method"] = "kept (transform only)"
        else:
            uv[o.name] = B.uv_stats(o)
            uv[o.name]["method"] = "kept (untouched)"
    # bookkeeping checks
    failures = []
    ids = [o.pass_index for o in objs]
    if len(ids) != len(set(ids)):
        failures.append("duplicate ids")
    old_ids = {r["id"] for r in before.values()}
    for a in log["added"]:
        if a["id"] in old_ids or a["id"] <= max(old_ids, default=0) and a["id"] in old_ids:
            failures.append(f"id {a['id']} reused")
    untouched_ok, mismatched = 0, []
    for o in objs:
        if o.name in touched or o.name not in before:
            continue
        if B.geometry_sha256(o) == before[o.name]["geometry_sha256"]:
            untouched_ok += 1
        else:
            mismatched.append(o.name)
    if mismatched:
        failures.append(f"untouched objects changed geometry: {mismatched}")
    for o in objs:
        u = uv[o.name]
        if not u.get("has_b2r_uv") or u.get("inter_island_overlap_px") or not u.get("uv_in_unit_square"):
            failures.append(f"{o.name}: bad b2r_uv")
    log["untouched_geometry_identical"] = untouched_ok
    log["untouched_geometry_changed"] = mismatched
    log["touched"] = sorted(touched)
    log["next_id"] = next_id
    log["failures"] = failures
    if failures:
        common.write_json(out / "edits_FAILED.json", log)
        raise SystemExit("[edit] FAILED: " + "; ".join(failures))
    cam = scene.camera
    report = B.verify(scene, cam, objs, uv, room=None)
    report["source"] = {"kind": "edit", "from_iter": args.iter}
    report["edit_seconds"] = round(time.time() - t0, 1)
    common.write_json(out / "edits.json", log)
    B.write_outputs(out, scene, cam, objs, uv, report, args.samples, not args.no_render, TAG)
    B.log(TAG, f"summary added={[a['name'] for a in log['added']]} removed={[r['name'] for r in log['removed']]} "
               f"replaced={[r['name'] for r in log['replaced']]} untouched_identical={untouched_ok}/"
               f"{untouched_ok + len(mismatched)}")
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
