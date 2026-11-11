"""detail: turn the residual's image-space detail height into real surface relief, keep it only on the
objects where it measurably helps, and score the result. One command does the whole step.

    blender -b --factory-startup --python scripts/blender/detail.py -- \
        --case CASE --iter K [--strength 0.4] [--distance 0.05] [--drop-tol 0.002] \
        [--samples 512] [--reuse-textures] [--dry-run]

Steps
  1. project residual/iterK/detail_height.png into the UV space of every "small" object (class "small" in
     the UNTAGGED residual/iterK/per_object.json; the open-sky row has no surface and is skipped)
     -> backproject/iterK/textures/<obj>_detail_height.png, packed;
  2. add a Bump node "detail_bump" (Strength --strength, Distance --distance) after the material's existing
     chain into the Principled BSDF Normal; save backproject/iterK/scene_lit_detail_all.blend;
  3. render it with render_view.py -> residual/iterK/rerender_detail_all.png; score with
     `b2r.residual --tag detail_all`;
  4. per-object selection: an object is DROPPED (Strength 0; node and texture stay so a person can turn it
     back on) when its per-object LPIPS-Alex with detail exceeds the untagged baseline by more than --drop-tol;
  5. if anything was dropped, render the selected configuration -> rerender_detail.png (else a copy of
     rerender_detail_all.png) and score it with --tag detail;
  6. if the selected LPIPS-Alex is not strictly below the baseline, every detail_bump is set to 0 in the saved
     blend ("improved": false);
  7. save backproject/iterK/scene_lit_detail.blend and residual/iterK/detail_result.json.

Never written: the untagged residual outputs; they are hashed before and checked after every scoring call and
the run fails if any changed. Lights are verified unchanged.

Projection reuses the back-projection stage (scripts/blender/backproject.py, core_main) unchanged: the job's
maps dir is a scratch folder whose height.png is detail_height.png (the other maps are links to the real
decompose maps; those outputs are discarded), so the same texel rasteriser, camera projector, visibility test
and same-id sampling apply. Two things are captured from it without editing it (by wrapping
scipy.ndimage.distance_transform_edt and cv2.imwrite during the call): each texture's UV-covered mask and the
files it writes. Afterwards only texels marked *seen* keep their projected value; every other covered texel is
set to neutral 0.5, and UV gutters are re-dilated from the nearest covered texel as backproject does.
The triangle export (Blender side) mirrors backproject.blender_main's, which is inline there.

The detail map is derived from the target-vs-render residual at this camera and selection is scored at the same
camera, so part of the gain is view-fitted (recorded in detail_result.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # .../scripts
sys.path.insert(1, str(HERE))         # .../scripts/blender
from b2r import common  # noqa: E402

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

import numpy as np  # noqa: E402

NEUTRAL = 0.5
DETAIL_NODE = "detail_bump"
UNTAGGED = ("lpips.json", "per_object.json", "contact_sheet.png", "rerender.png", "rerender.exr",
            "detail_height.png", "detail.json", "error_map.png", "error_map.npy")
VIEW_FIT_CAVEAT = (
    "The detail height map is derived from the band-passed target-vs-rerender luminance residual at this exact "
    "camera, and selection is scored at the same camera, so part of the LPIPS gain is view-fitted; it has not "
    "been validated from a novel view. It is a shape-from-shading heuristic: shadow/contact edges and "
    "print/albedo detail can be turned into false relief.")
MAPS_LINKED = ("albedo", "roughness", "metallic", "normal_cam")


def import_backproject():
    """The back-projection port. Fails with a message naming the missing module; nothing is vendored."""
    try:
        import backproject  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"detail: cannot import the back-projection stage scripts/blender/backproject.py "
                         f"({e}); the detail step reuses its projector and has no copy of its own") from e
    if not hasattr(backproject, "core_main"):
        raise SystemExit("detail: scripts/blender/backproject.py has no core_main(work); cannot project")
    return backproject


# =========================================================================== user's Python side
def core_main(work: Path) -> None:
    import cv2
    import scipy.ndimage as ndi
    bpj = import_backproject()

    job = json.loads((work / "job.json").read_text())
    final_dir = Path(job["final_tex_dir"])
    covered, last = {}, {}
    orig_edt, orig_imwrite = ndi.distance_transform_edt, cv2.imwrite

    def edt_wrap(inp, *a, **k):
        last["cov"] = ~np.asarray(inp, bool)
        return orig_edt(inp, *a, **k)

    def imwrite_wrap(path, img, *a, **k):
        p = str(path)
        if p.endswith("_coverage.png"):
            if "cov" not in last:
                raise RuntimeError("backproject wrote a coverage image without a UV-gutter dilation first; "
                                   "its internals changed and the covered mask cannot be captured")
            covered[Path(p).name[: -len("_coverage.png")]] = last["cov"].copy()
        return orig_imwrite(path, img, *a, **k)

    ndi.distance_transform_edt = edt_wrap
    cv2.imwrite = imwrite_wrap
    try:
        bpj.core_main(work)  # unchanged projector / visibility / sampling
    finally:
        ndi.distance_transform_edt = orig_edt
        cv2.imwrite = orig_imwrite

    core = json.loads((work / "core_result.json").read_text())
    tex = Path(job["tex_dir"])
    out = {}
    for e in job["objects"]:
        name = e["name"]
        h = cv2.imread(str(tex / f"{name}_height.png"), cv2.IMREAD_UNCHANGED)
        cimg = cv2.imread(str(tex / f"{name}_coverage.png"), cv2.IMREAD_UNCHANGED)
        if h is None or h.dtype != np.uint16 or cimg is None:
            raise RuntimeError(f"{name}: projected height missing or not 16-bit")
        seen = cimg > 0
        cov = covered.get(name)
        if cov is None or cov.shape != h.shape or (seen & ~cov).any():
            raise RuntimeError(f"{name}: captured covered mask inconsistent with backproject's outputs")
        v = h.astype(np.float64) / 65535.0
        v[~seen] = NEUTRAL
        _, (ir, ic) = orig_edt(~cov, return_indices=True)
        v = v[ir, ic]
        img = np.clip(np.rint(v * 65535), 0, 65535).astype(np.uint16)
        dst = final_dir / f"{name}_detail_height.png"
        if not orig_imwrite(str(dst), img):
            raise RuntimeError(f"cannot write {dst}")
        dv = v[seen] - NEUTRAL
        out[name] = {
            "file": dst.name, "resolution": int(h.shape[0]),
            "seen_fraction_of_texture": float(seen.mean()),
            "covered_fraction_of_texture": float(cov.mean()),
            "seen_dev_from_neutral": {"std": float(dv.std()) if dv.size else 0.0,
                                      "p99_abs": float(np.percentile(np.abs(dv), 99)) if dv.size else 0.0},
        }
        print(f"[detail-core] {name}: R={h.shape[0]} seen {seen.mean():.3f} std(dev) "
              f"{out[name]['seen_dev_from_neutral']['std']:.4f}", flush=True)
    (work / "detail_result.json").write_text(json.dumps({"objects": out,
                                                         "backproject_parameters": core.get("parameters")}, indent=2))


# =========================================================================== Blender side
def export_meshes(meshes, work: Path, uv_name: str) -> list[dict]:
    """Triangle export in backproject's job format (mesh_<id>.npz)."""
    entries = []
    for o in meshes:
        me = o.data
        if uv_name in me.uv_layers:
            uv_layer = me.uv_layers[uv_name]
        elif me.uv_layers.active is not None:
            uv_layer = me.uv_layers.active
        else:
            raise RuntimeError(f"{o.name}: no UV map")
        if me.users > 1:
            raise RuntimeError(f"{o.name}: mesh data shared by {me.users} users")
        me.calc_loop_triangles()
        nl, nv, nt = len(me.loops), len(me.vertices), len(me.loop_triangles)
        uv = np.empty(nl * 2, np.float64)
        uv_layer.data.foreach_get("uv", uv)
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
        tri_pos, tri_uv = cow[lv[tl]], uv[tl]
        tri_nrm = cn[tl] @ Nw.T
        tri_nrm /= np.maximum(np.linalg.norm(tri_nrm, axis=-1, keepdims=True), 1e-12)
        e1, e2 = tri_pos[:, 1] - tri_pos[:, 0], tri_pos[:, 2] - tri_pos[:, 0]
        ng = np.cross(e1, e2)
        ng /= np.maximum(np.linalg.norm(ng, axis=1, keepdims=True), 1e-12)
        ng[np.sum(ng * tri_nrm.mean(1), 1) < 0] *= -1
        d1, d2 = tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0]
        det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        dets = np.where(np.abs(det) > 1e-14, det, 1e-14)
        tan = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / dets[:, None]
        bit = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / dets[:, None]
        sign = np.where(np.sum(np.cross(ng, tan) * bit, 1) < 0, -1.0, 1.0)
        f = f"mesh_{o.pass_index}.npz"
        np.savez(work / f, tri_pos=tri_pos, tri_uv=tri_uv, tri_nrm=tri_nrm, tri_ng=ng, tri_tan=tan,
                 tri_bit=bit, tri_sign=sign)
        entries.append({"name": o.name, "id": int(o.pass_index), "category": "", "file": f})
    return entries


def light_signature(scene) -> dict:
    sig = {}
    for o in scene.objects:
        if o.type == "LIGHT":
            d = o.data
            sig[o.name] = [d.type, round(float(d.energy), 6), [round(c, 6) for c in d.color],
                           [round(x, 6) for x in np.array(o.matrix_world).ravel()]]
    w = scene.world
    if w is not None and w.node_tree is not None:
        for n in w.node_tree.nodes:
            if n.bl_idname == "ShaderNodeBackground":
                sig["_world_" + n.name] = [round(float(n.inputs["Strength"].default_value), 6),
                                           [round(c, 6) for c in n.inputs["Color"].default_value]]
    return sig


def _hash(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def object_material(o):
    """The object's one material with a Principled BSDF and a UV Map node (as backproject builds it)."""
    mats = [m for m in o.data.materials if m is not None]
    if len(mats) != 1 or mats[0].node_tree is None:
        raise RuntimeError(f"{o.name}: expected exactly one node material, got {[m.name for m in mats]}")
    nt = mats[0].node_tree
    bsdf = [n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"]
    uvn = [n for n in nt.nodes if n.bl_idname == "ShaderNodeUVMap"]
    if len(bsdf) != 1 or not uvn:
        raise RuntimeError(f"{mats[0].name}: expected one Principled BSDF and a UV Map node")
    return mats[0], bsdf[0], uvn[0]


def blender_main() -> None:
    import bpy

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--distance", type=float, default=0.05)
    ap.add_argument("--drop-tol", type=float, default=0.002,
                    help="drop detail on an object if its LPIPS-Alex worsens by more than this")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--reuse-textures", action="store_true",
                    help="skip projection; use existing textures/<obj>_detail_height.png")
    ap.add_argument("--dry-run", action="store_true", help="check inputs and print the plan; write nothing")
    args = ap.parse_args(common.script_args())
    t_start = time.time()
    case = common.case_dir(args.case)
    bpj = import_backproject()

    bp = common.stage_dir(case, "backproject", args.iter, create=False)
    rs = common.stage_dir(case, "residual", args.iter, create=False)
    gb = common.stage_dir(case, "gbuffers", args.iter, create=False)
    dec = common.stage_dir(case, "decompose", args.iter, create=False)
    src_blend, all_blend, out_blend = bp / "scene_lit.blend", bp / "scene_lit_detail_all.blend", bp / "scene_lit_detail.blend"
    detail_png, tex_dir = rs / "detail_height.png", bp / "textures"
    blender_exe = common.find_blender()
    needed = [src_blend, detail_png, rs / "per_object.json", rs / "lpips.json", gb / "camera.json",
              gb / "position.exr", gb / "normal.exr", gb / "id.png", HERE / "render_view.py"]
    if not args.reuse_textures:
        needed += [dec / f"{n}.png" for n in MAPS_LINKED]
    missing = [str(p) for p in needed if not p.exists()] + ([] if blender_exe else ["blender executable"])
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    base_objs = common.read_json(rs / "per_object.json")["objects"]
    base_lp = common.read_json(rs / "lpips.json")["rerender_vs_target"]
    small = {v["name"] for v in base_objs.values() if v["class"] == "small" and not v.get("is_open_sky")}
    print(f"[detail] plan: {len(small)} small objects, strength {args.strength}, distance {args.distance}, "
          f"drop-tol {args.drop_tol}, {args.samples} spp; baseline LPIPS-Alex {base_lp['lpips_alex']:.4f}", flush=True)
    if args.dry_run:
        print("[detail] dry run: inputs present, nothing written", flush=True)
        return

    untagged_before = {n: _hash(rs / n) for n in UNTAGGED}

    def check_untagged(when):
        changed = [n for n in UNTAGGED if _hash(rs / n) != untagged_before[n]]
        if changed:
            raise RuntimeError(f"untagged residual outputs changed {when}: {changed}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    def render(blend: Path, png: Path) -> float:
        t0 = time.time()
        r = subprocess.run([blender_exe, "-b", "--factory-startup", "--python", str(HERE / "render_view.py"), "--",
                            "--blend", str(blend), "--out", str(png), "--samples", str(args.samples)], env=env)
        if r.returncode != 0 or not png.is_file():
            raise RuntimeError(f"render of {blend} failed")
        return round(time.time() - t0, 1)

    def score(png: Path, tag: str):
        r = subprocess.run([common.python_exe(), "-m", "b2r.residual", "--case", str(case), "--iter",
                            str(args.iter), "--rerender", str(png), "--tag", tag], env=env)
        if r.returncode != 0:
            raise RuntimeError(f"b2r.residual --tag {tag} failed")
        check_untagged(f"after scoring --tag {tag}")
        return (common.read_json(rs / f"lpips_{tag}.json")["rerender_vs_target"],
                common.read_json(rs / f"per_object_{tag}.json")["objects"])

    bpy.ops.wm.open_mainfile(filepath=str(src_blend))
    scene = bpy.context.scene
    lights_before = light_signature(scene)
    meshes = [o for o in scene.objects if o.type == "MESH" and not o.hide_render and o.name in small]
    print(f"[detail] small objects with meshes: {[o.name for o in meshes]}", flush=True)
    if not meshes:
        raise RuntimeError("no small object has a mesh in scene_lit.blend; nothing to add detail to")

    # ------------------------------------------------------------- 1. projection
    proj_report = None
    if not args.reuse_textures:
        work = bp / "_detail_work"
        if work.exists():
            shutil.rmtree(work)
        maps = work / "maps"
        maps.mkdir(parents=True)
        for n in MAPS_LINKED:
            os.symlink(dec / f"{n}.png", maps / f"{n}.png")
        os.symlink(detail_png, maps / "height.png")
        entries = export_meshes(meshes, work, getattr(bpj, "UV_NAME", "b2r_uv"))
        W, H = common.render_size(case)
        job = {"gbuffers": str(gb), "maps_dir": str(maps), "tex_dir": str(work / "tex"),
               "final_tex_dir": str(tex_dir), "width": W, "height": H, "objects": entries, "normal_strength": 1.0}
        (work / "job.json").write_text(json.dumps(job, indent=2))
        tex_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = subprocess.run([common.python_exe(), str(Path(__file__).resolve()), "--core", "--work", str(work)], env=env)
        if r.returncode != 0:
            raise RuntimeError("detail projection failed (see output above)")
        proj_report = json.loads((work / "detail_result.json").read_text())
        proj_report["core_seconds"] = round(time.time() - t0, 1)
        shutil.rmtree(work, ignore_errors=True)

    # ------------------------------------------------------------- 2. material edit
    added = {}
    for o in meshes:
        f = tex_dir / f"{o.name}_detail_height.png"
        if not f.is_file():
            raise FileNotFoundError(f)
        mat, bsdf, uvn = object_material(o)
        nt = mat.node_tree
        if DETAIL_NODE in nt.nodes:
            raise RuntimeError(f"{mat.name} already has {DETAIL_NODE}")
        prev = [l for l in nt.links if l.to_node == bsdf and l.to_socket.name == "Normal"]
        if len(prev) > 1:
            raise RuntimeError(f"{mat.name}: {len(prev)} links into the BSDF Normal")
        prev_sock = prev[0].from_socket if prev else None
        if prev:
            nt.links.remove(prev[0])
        img = bpy.data.images.load(str(f), check_existing=False)
        img.name = f.stem
        img.colorspace_settings.name = "Non-Color"
        img.pack()
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.name = tn.label = f"{o.name}_detail_height_tex"
        tn.image = img
        tn.interpolation = "Linear"
        tn.location = (bsdf.location[0] - 950, bsdf.location[1] - 900)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.name = DETAIL_NODE
        bump.label = "detail_bump (residual relief)"
        bump.location = (bsdf.location[0] - 190, bsdf.location[1] - 760)
        bump.inputs["Strength"].default_value = args.strength
        bump.inputs["Distance"].default_value = args.distance
        nt.links.new(uvn.outputs["UV"], tn.inputs["Vector"])
        nt.links.new(tn.outputs["Color"], bump.inputs["Height"])
        if prev_sock is not None:
            nt.links.new(prev_sock, bump.inputs["Normal"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        added[o.name] = {"material": mat.name, "image": img.name,
                         "fed_from": f"{prev_sock.node.name}.{prev_sock.name}" if prev_sock else None}

    def set_strengths(strengths: dict):
        for name, s in strengths.items():
            mat, _, _ = object_material(bpy.data.objects[name])
            mat.node_tree.nodes[DETAIL_NODE].inputs["Strength"].default_value = s

    def save(path: Path):
        if light_signature(bpy.context.scene) != lights_before:
            raise RuntimeError("lights changed - refusing to save")
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True, copy=True)
        print(f"[detail] saved {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)

    save(all_blend)

    # ------------------------------------------------------------- 3. all-objects pass
    times = {}
    png_all = rs / "rerender_detail_all.png"
    times["render_all"] = render(all_blend, png_all)
    lp_all, obj_all = score(png_all, "detail_all")

    # ------------------------------------------------------------- 4. per-object selection
    by_base = {v["name"]: v for v in base_objs.values()}
    by_all = {v["name"]: v for v in obj_all.values()}
    per_obj, kept = {}, []
    for name in added:
        b = by_base.get(name, {}).get("lpips_rerender_target")
        a_ = by_all.get(name, {}).get("lpips_rerender_target")
        if b is None or a_ is None:
            keep, why = False, "no per-object score (not visible?)"
        else:
            keep = (a_ - b) <= args.drop_tol
            why = f"delta {a_ - b:+.4f} {'<=' if keep else '>'} tol {args.drop_tol}"
        per_obj[name] = {"lpips_baseline": b, "lpips_detail_all": a_, "kept": keep, "reason": why}
        if keep:
            kept.append(name)
    dropped = [n for n in added if n not in kept]
    print(f"[detail] kept {kept}; dropped {dropped}", flush=True)

    # ------------------------------------------------------------- 5. selected pass
    png_sel = rs / "rerender_detail.png"
    set_strengths({n: (args.strength if n in kept else 0.0) for n in added})
    if dropped:
        sel_blend = bp / "_scene_lit_detail_selected.blend"
        save(sel_blend)
        times["render_selected"] = render(sel_blend, png_sel)
        sel_blend.unlink()
        sel_note = "selected configuration rendered and scored with --tag detail"
    else:
        shutil.copyfile(png_all, png_sel)
        sel_note = "nothing dropped: rerender_detail.png is a copy of rerender_detail_all.png"
    lp_sel, obj_sel = score(png_sel, "detail")
    by_sel = {v["name"]: v for v in obj_sel.values()}
    for name in per_obj:
        per_obj[name]["lpips_selected"] = by_sel.get(name, {}).get("lpips_rerender_target")

    # ------------------------------------------------------------- 6/7. decide + save + verify
    improved = bool(lp_sel["lpips_alex"] < base_lp["lpips_alex"]) and bool(kept)
    if not improved:
        set_strengths({n: 0.0 for n in added})
    save(out_blend)
    bpy.ops.wm.open_mainfile(filepath=str(out_blend))
    issues = []
    if light_signature(bpy.context.scene) != lights_before:
        issues.append("light signature differs from scene_lit.blend")
    saved_strengths = {}
    for name in added:
        mat, bsdf, _ = object_material(bpy.data.objects[name])
        b = mat.node_tree.nodes.get(DETAIL_NODE)
        if b is None or not any(l.from_node == b and l.to_node == bsdf for l in mat.node_tree.links):
            issues.append(f"{name}: detail_bump not wired to the BSDF")
            continue
        saved_strengths[name] = round(float(b.inputs["Strength"].default_value), 4)
    unpacked = [i.name for i in bpy.data.images if i.source == "FILE" and i.packed_file is None]
    if unpacked:
        issues.append(f"unpacked images {unpacked}")
    check_untagged("at end")

    keys = ("lpips_alex", "lpips_vgg", "psnr_db", "ssim")
    pick = lambda d: {k: round(float(d[k]), 4) for k in keys}  # noqa: E731
    result = {
        "iter": args.iter, "strength": args.strength, "distance_m": args.distance,
        "drop_rule": (f"drop detail on an object if per-object LPIPS-Alex (detail_all) - baseline > {args.drop_tol}; "
                      f"dropped objects keep detail_bump at Strength 0 (re-enable by setting it to {args.strength})"),
        "improved_rule": ("selected full-image LPIPS-Alex strictly below baseline; otherwise all detail_bump "
                          "strengths are 0 in the saved blend"),
        "baseline_no_detail": pick(base_lp), "all_objects": pick(lp_all), "selected": pick(lp_sel),
        "improved": improved, "kept": kept, "dropped": dropped, "per_object": per_obj, "nodes_added": added,
        "saved_blend": str(out_blend.relative_to(case)), "saved_detail_bump_strengths": saved_strengths,
        "renders": {"all_objects": str(png_all.relative_to(case)), "selected": str(png_sel.relative_to(case)),
                    "settings": f"render_view.py {args.samples} spp", "note": sel_note, "seconds": times},
        "scores": {"all_objects": f"residual/iter{args.iter}/lpips_detail_all.json",
                   "selected": f"residual/iter{args.iter}/lpips_detail.json"},
        "caveat_view_fitted": VIEW_FIT_CAVEAT,
        "note": ("detail kept on objects listed in 'kept'" if improved else
                 "detail did not improve LPIPS-Alex over the no-detail baseline; every detail_bump is at "
                 "Strength 0 in the saved blend"),
        "verify_issues": issues,
        "seconds": round(time.time() - t_start, 1),
    }
    if proj_report is not None:
        result["projection"] = proj_report
    common.write_json(rs / "detail_result.json", result)
    print(f"[detail] baseline {base_lp['lpips_alex']:.4f} all {lp_all['lpips_alex']:.4f} selected "
          f"{lp_sel['lpips_alex']:.4f} improved={improved}; issues {issues}", flush=True)
    if issues:
        sys.exit(1)


if __name__ == "__main__":
    try:
        if IN_BLENDER:
            blender_main()
        else:
            ap = argparse.ArgumentParser()
            ap.add_argument("--core", action="store_true", required=True)
            ap.add_argument("--work", required=True)
            a = ap.parse_args(common.script_args())
            core_main(Path(a.work))
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
