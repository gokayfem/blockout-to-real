"""Summarise a .blend so the "ordinary editable scene" claim can be checked, not asserted.

    blender -b --factory-startup --python scripts/blender/inspect_blend.py -- --blend PATH --out PATH.json

Per mesh object: vertex / face counts, UV layers, modifiers, material, the shader node types in
that material, which Principled BSDF inputs are driven by image textures, bump nodes.
Also: images (size, packed, colour space), lights (type, energy, colour, flags, custom
properties), the world, colour management, and anything that would make the scene NOT an
ordinary one (a material without a Principled BSDF, emission shaders, a compositor node group,
unpacked images). Makes no assumption about what the scene contains.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # .../scripts
sys.path.insert(1, str(Path(__file__).resolve().parent))         # .../scripts/blender
from b2r import common  # noqa: E402
import blockout as B  # noqa: E402  (script_carriers)


def linked_image(sock, depth=0):
    """Follow links upstream from a socket and return the first image texture's name."""
    if depth > 8 or not sock.is_linked:
        return None
    node = sock.links[0].from_node
    if node.type == "TEX_IMAGE":
        return node.image.name if node.image else None
    for inp in node.inputs:
        got = linked_image(inp, depth + 1)
        if got:
            return got
    return None


def _props(idblock) -> dict:
    out = {}
    for k in idblock.keys():
        v = idblock[k]
        if isinstance(v, (bool, int, float, str)):
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(common.script_args())
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()), use_scripts=False)
    scene = bpy.context.scene

    objects, flags = [], []
    for ob in sorted(scene.objects, key=lambda o: o.name):
        if ob.type != "MESH":
            continue
        me = ob.data
        entry = {
            "name": ob.name,
            "pass_index": ob.pass_index,
            "verts": len(me.vertices),
            "faces": len(me.polygons),
            "uv_layers": [uv.name for uv in me.uv_layers],
            "active_uv": me.uv_layers.active.name if me.uv_layers.active else None,
            "modifiers": [m.type for m in ob.modifiers],
            "hide_render": bool(ob.hide_render),
            "materials": [],
        }
        if not ob.material_slots:
            flags.append(f"{ob.name}: no material")
        for slot in ob.material_slots:
            mat = slot.material
            nt = getattr(mat, "node_tree", None) if mat else None
            if nt is None:
                entry["materials"].append({"name": mat.name if mat else None, "nodes": None})
                flags.append(f"{ob.name}: material slot without a node material")
                continue
            nodes = nt.nodes
            types = sorted({n.type for n in nodes})
            m = {"name": mat.name, "node_types": types, "principled_inputs_from_images": {}}
            for n in nodes:
                if n.type == "BSDF_PRINCIPLED":
                    for key in ("Base Color", "Roughness", "Metallic", "Normal"):
                        m["principled_inputs_from_images"][key] = linked_image(n.inputs[key])
                    em = n.inputs.get("Emission Strength")
                    m["emission_strength"] = float(em.default_value) if em is not None else None
                    if em is not None and em.default_value > 0 and n.inputs["Emission Color"].is_linked:
                        flags.append(f"{ob.name}: Principled emission driven by a texture")
                if n.type == "BUMP":
                    m.setdefault("bump_nodes", {})[n.name] = round(float(n.inputs["Strength"].default_value), 4)
                if n.type == "UVMAP":
                    m.setdefault("uv_maps_used", []).append(n.uv_map)
            if "BSDF_PRINCIPLED" not in types:
                flags.append(f"{ob.name}: material {mat.name} has no Principled BSDF")
            if "EMISSION" in types:
                flags.append(f"{ob.name}: material {mat.name} contains an Emission shader")
            entry["materials"].append(m)
        objects.append(entry)

    images = [
        {"name": im.name, "size": list(im.size), "packed": im.packed_file is not None,
         "colorspace": im.colorspace_settings.name}
        for im in bpy.data.images if im.type == "IMAGE" and im.size[0] > 0
    ]
    lights = []
    for ob in scene.objects:
        if ob.type != "LIGHT":
            continue
        ld = ob.data
        e = {"name": ob.name, "type": ld.type, "energy": round(float(ld.energy), 4),
             "color": [round(c, 3) for c in ld.color],
             "location": [round(v, 3) for v in ob.matrix_world.translation],
             "hide_render": bool(ob.hide_render),
             "visible_camera": bool(ob.visible_camera), "visible_glossy": bool(ob.visible_glossy),
             "is_portal": bool(getattr(getattr(ld, "cycles", None), "is_portal", False)),
             "custom_properties": _props(ob)}
        if ld.type == "AREA":
            e["size_m"] = [round(ld.size, 4), round(ld.size_y, 4)]
        lights.append(e)
    world = None
    if scene.world is not None and scene.world.node_tree is not None:
        for n in scene.world.node_tree.nodes:
            if n.type == "BACKGROUND":
                world = {"name": scene.world.name,
                         "color": [round(c, 3) for c in n.inputs["Color"].default_value[:3]],
                         "strength": round(float(n.inputs["Strength"].default_value), 4),
                         "color_from_image": linked_image(n.inputs["Color"])}
    # Blender 5.x: the compositor is scene.compositing_node_group (scene.node_tree is gone)
    comp = getattr(scene, "compositing_node_group", None)
    comp_nodes = sorted({n.type for n in comp.nodes}) if comp is not None else []
    if comp is not None and any(t not in ("R_LAYERS", "COMPOSITE", "GROUP_OUTPUT", "VIEWER") for t in comp_nodes):
        flags.append(f"compositor node group present: {comp_nodes}")
    unpacked = [i["name"] for i in images if not i["packed"]]
    if unpacked:
        flags.append(f"{len(unpacked)} images not packed")
    scripts = B.script_carriers()
    if scripts["text_blocks"]:
        auto = sum(t["auto_run"] for t in scripts["text_blocks"])
        flags.append(f"{len(scripts['text_blocks'])} text block(s) stored in the file ({auto} set to run on load)")
    if scripts["drivers"]:
        flags.append(f"{scripts['drivers']} driver(s) stored in the file")

    out = {
        "blend": str(Path(args.blend).resolve()),
        "blender": bpy.app.version_string,
        "mesh_objects": len(objects),
        "total_faces": sum(o["faces"] for o in objects),
        "images": len(images),
        "images_packed": sum(1 for i in images if i["packed"]),
        "images_unpacked": unpacked,
        "lights": lights,
        "world": world,
        "camera": scene.camera.name if scene.camera else None,
        "render_engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y,
                       scene.render.resolution_percentage],
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "compositor_nodes": comp_nodes,
        "text_blocks": scripts["text_blocks"],
        "drivers": scripts["drivers"],
        "flags_not_ordinary": flags,
        "objects": objects,
        "image_list": images,
    }
    common.write_json(args.out, out)
    print("[inspect]", json.dumps({k: out[k] for k in ("mesh_objects", "total_faces", "images", "images_packed",
                                                      "flags_not_ordinary")}))
    print("[inspect] lights:", json.dumps([{k: l[k] for k in ("name", "type", "energy")} for l in lights]))
    print("[inspect] world:", json.dumps(world))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
