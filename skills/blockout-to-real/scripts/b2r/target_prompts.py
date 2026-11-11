"""Prompt construction for the target stage: the five-level conditioning ladder, built only from
case.json "look", objects.json descriptions and the g-buffers (visibility-aware wording).

Nothing here names a scene type, an object or a material: every noun in a prompt comes from the case.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ALL_LEVELS = ("L1", "L2", "L3", "L4", "L5")

# Visibility thresholds for the per-object prompt text.
SMALL_FRAC = 0.005  # visible area < 0.5% of the image -> "small in frame"
TINY_FRAC = 0.002   # visible area < 0.2% of the image -> not listed on its own (one summary line)
# Occlusion = share of the object's projected-bbox hull covered by NEARER non-structure objects.
HIDDEN_OCC = 0.50  # occluded share >= 0.50 -> "almost entirely hidden behind X"
PARTLY_OCC = 0.20  # occluded share >= 0.20 -> "partly hidden behind X"

OCCLUSION_SENTENCE = (
    " Objects that are partly hidden behind others in the render stay partly hidden exactly as they are; "
    "nothing is pulled into view, brought forward, added or moved."
)
PHOTO_SENTENCE = "The result must look like a real photograph, not a 3D render or CGI."


def category(o: dict) -> str:
    return str(o.get("category") or "").strip().lower()


def _label(name: str) -> str:
    return name.replace("_", " ").strip()


def _where(bbox, W: int, H: int) -> str:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
    h = "left" if cx < 0.36 else ("right" if cx > 0.64 else "centre")
    v = "upper " if cy < 0.36 else ("lower " if cy > 0.64 else "")
    return f"{v}{h}"


def visibility(objects: list, visible: dict, idm: np.ndarray, depth16: np.ndarray, cam: dict) -> dict:
    """Per non-structure object: image fraction, visible fraction of its unoccluded footprint
    (approximated by the convex hull of its projected 3D bbox), and its main occluder."""
    import cv2

    H, W = idm.shape
    w2c = np.array(cam["world_to_cam"], float)
    names = {o["id"]: o["name"] for o in objects}
    cats = {o["id"]: category(o) for o in objects}
    info = {}
    for o in objects:
        oid = o["id"]
        px = int((visible.get(str(oid)) or {}).get("pixels", 0))
        rec = {"pixels": px, "image_frac": round(px / (W * H), 5), "visible_frac": None,
               "occluded_frac": None, "occluder": None}
        if px and cats[oid] != "structure" and "bbox_min" in o and "bbox_max" in o:
            lo, hi = np.array(o["bbox_min"], float), np.array(o["bbox_max"], float)
            corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            P = (w2c @ corners.T).T
            if (P[:, 2] < -0.05).all():  # all corners in front of the camera
                u = cam["cx"] + cam["fx"] * P[:, 0] / (-P[:, 2])
                v = cam["cy"] - cam["fy"] * P[:, 1] / (-P[:, 2])
                hull = cv2.convexHull(np.stack([u, v], 1).astype(np.float32)).astype(np.int32)
                mask = np.zeros((H, W), np.uint8)
                cv2.fillConvexPoly(mask, hull, 1)
                mask = mask.astype(bool)
                if mask.any():
                    rec["visible_frac"] = round(px / max(int(mask.sum()), 1), 3)
                    own_depth = float(depth16[idm == oid].mean())
                    best, best_n, occ_px = None, 0, 0
                    for j in np.unique(idm[mask]):
                        j = int(j)
                        if j == oid or j == 0 or cats.get(j, "") == "structure":
                            continue
                        sel = mask & (idm == j)
                        if float(depth16[sel].mean()) > own_depth:  # inverse depth: near = white
                            occ_px += int(sel.sum())
                            if sel.sum() > best_n:
                                best, best_n = j, int(sel.sum())
                    rec["occluded_frac"] = round(occ_px / max(int(mask.sum()), 1), 3)
                    rec["occluder"] = names.get(best) if best is not None else None
        info[oid] = rec
    return info


def object_lines(objects: list, visible: dict, vis_info: dict, W: int, H: int) -> list:
    """Visibility-aware object list. Zero-pixel objects are omitted; largely hidden objects are
    described as hidden behind their occluder; small objects are described as small in frame."""
    lines, tiny = [], []
    for o in objects:
        v = visible.get(str(o["id"]))
        if not v or v.get("pixels", 0) == 0 or not v.get("bbox"):
            continue
        r = vis_info.get(o["id"], {})
        name = _label(o["name"])
        desc = str(o.get("description") or "").strip() or name
        where = _where(v["bbox"], W, H) + " of the frame"
        if category(o) == "exterior":
            where += ", outside, seen through an opening"
        of, occ = r.get("occluded_frac"), r.get("occluder")
        behind = f" behind the {_label(occ)}" if occ else ""
        if r.get("image_frac", 1) < TINY_FRAC:
            # A scene with dozens of small parts would otherwise bury the prompt in near-identical lines.
            tiny.append(name)
            continue
        if of is not None and occ and of >= HIDDEN_OCC:
            s = (f"- {name} ({where}, ALMOST ENTIRELY HIDDEN{behind}): {desc}. "
                 f"Only a small sliver of it (about {r['image_frac'] * 100:.1f}% of the image) is visible in the "
                 f"render; keep it hidden exactly there{behind}, do not bring it forward or reveal more of it")
        elif of is not None and occ and of >= PARTLY_OCC:
            s = (f"- {name} ({where}, partly hidden{behind}): {desc}; it stays "
                 f"partly hidden{behind} exactly as in the render")
        elif r.get("image_frac", 1) < SMALL_FRAC:
            s = (f"- {name} ({where}, small in frame, about {r['image_frac'] * 100:.1f}% of the image): "
                 f"{desc}; keep it the same small size and position")
        else:
            s = f"- {name} ({where}): {desc}"
        lines.append(s)
    if tiny:
        lines.append("- small details, each kept exactly where and as small as it is in the render: " + ", ".join(tiny))
    return lines


def build_level(level: str, inputs: dict, objs: list, look: str, ref: Path | None) -> tuple[str, list, dict]:
    """inputs: role -> path for 'beauty', 'depth', 'normal', 'idvis'. Returns (prompt, [(role, path)], extra args)."""
    obj_block = "\n".join(objs) if objs else "- (no object descriptions available)"
    tail = f"\n\nLook of the photograph: {look.strip()}\n{PHOTO_SENTENCE}"
    extra: dict = {"thinking_level": "minimal"}
    if level == "L1":
        imgs = [("depth map", inputs["depth"])]
        p = ("Image 1 is a depth map of a scene (near = white, far = black). Make a photograph whose layout "
             "roughly follows this depth map. What is in the scene, approximately:\n"
             f"{obj_block}{tail}")
    elif level == "L2":
        imgs = [("grey render", inputs["beauty"])]
        p = ("Image 1 is an untextured grey 3D render of a scene. Use it as a layout reference for a "
             "photograph of the same scene from roughly the same viewpoint; the objects may be restyled. "
             f"Objects in the scene:\n{obj_block}{tail}")
    elif level == "L3":
        imgs = [("grey render", inputs["beauty"])]
        p = ("Image 1 is an untextured grey 3D blockout render of a scene. Turn it into a real photograph "
             "of the same scene. Keep the camera, the layout and every object exactly where it is, with the same "
             "size, shape and silhouette; do not add, remove or move anything."
             f"{OCCLUSION_SENTENCE} Objects:\n{obj_block}{tail}")
    elif level == "L4":
        imgs = [("grey render", inputs["beauty"]), ("depth map", inputs["depth"]), ("normal map", inputs["normal"])]
        p = ("You are given three pixel-aligned images of the same view: image 1 is an untextured grey 3D "
             "blockout render of a scene, image 2 its depth map (near = white, far = black), image 3 its "
             "camera-space surface-normal map (colour encodes surface orientation). Turn image 1 into a real "
             "photograph of the same scene. Keep the camera, the layout and every object exactly where it is, "
             "with the same size, shape and silhouette, consistent with the depth and normal maps; do not add, "
             f"remove or move anything.{OCCLUSION_SENTENCE} Objects:\n{obj_block}{tail}")
    elif level == "L5":
        imgs = [("grey render", inputs["beauty"]), ("depth map", inputs["depth"]), ("normal map", inputs["normal"]),
                ("colour-region map", inputs["idvis"])]
        p = ("You are given four pixel-aligned images of the same view: image 1 is an untextured grey 3D "
             "blockout render of a scene, image 2 its depth map (near = white, far = black), image 3 its "
             "camera-space surface-normal map, image 4 an object map where every flat-coloured region is exactly "
             "one object (black = no geometry: empty background, sky or an opening). The colours in image 4 are "
             "arbitrary labels only: they are not paint or material colours and must not appear anywhere in the "
             "photograph; every surface gets the real material from its description below. Turn image 1 into a "
             "real photograph of the same scene with the exact same framing. Each coloured region of image 4 must "
             "become exactly one photographed object, and the object's outline in the photo must coincide with "
             "that region's edges pixel for pixel. Keep every object exactly where it is with the same size, shape "
             "and silhouette; do not add, remove, move, resize or crop anything, do not change the camera."
             f"{OCCLUSION_SENTENCE} Objects:\n{obj_block}{tail}")
        extra = {"thinking_level": "high"}
    else:
        raise ValueError(level)
    if ref is not None:
        imgs = imgs + [("previous target photo", ref)]
        p += (f"\n\nImage {len(imgs)} is the photograph of this scene from the previous iteration. Keep the same "
              "materials, colours, lighting and overall look as that photo, but follow the geometry of the new "
              "grey render (image 1 where given) exactly.")
    return p, imgs, extra


def id_colours(idm: np.ndarray) -> np.ndarray:
    """Flat arbitrary colour per object id (golden-ratio hues), black for id 0."""
    import colorsys

    out = np.zeros(idm.shape + (3,), np.uint8)
    for i in np.unique(idm):
        if i == 0:
            continue
        h = (int(i) * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.55 + 0.4 * ((int(i) * 7) % 3) / 2, 0.95)
        out[idm == i] = (int(r * 255), int(g * 255), int(b * 255))
    return out
