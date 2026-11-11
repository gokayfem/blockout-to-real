# Conventions

The data contracts every stage depends on. Read this before writing a new tool or changing a stage.
`scripts/b2r/common.py` is the code form of this page.

## A case

Everything about one scene lives in one folder, outside the repository (`cases/` is git-ignored).

```text
CASE/
  case.json            settings and state (below)
  CONTRACT.md          what will be judged, written before the first target exists
  acceptance.json      targets, never lowered after a failure
  input/               the input as given, copied with its sha256
  blockout/iter<k>/    blockout.blend, objects.json, preview.png, verification.json, (scene_spec.json, edits.json)
  gbuffers/iter<k>/    beauty, depth, position, normal, id, camera.json, objects_visible.json, selfcheck.json
  target/iter<k>/      sweep/, target.png, choice.json, prompt.txt
  decompose/iter<k>/   albedo, roughness, metallic, normal_cam, height, depth_moge, depth_delta, shading, lighting.json
  backproject/iter<k>/ textures/, scene.blend, scene_lit.blend, scene_lit_detail.blend, lights.json, fit.json
  residual/iter<k>/    rerender.png, lpips.json, per_object.json, error_map.png, contact_sheet.png, critique.json
  final/               scene_final.blend, summary.json, contact sheets, novel view, relights, judge.json
  logs/                one log per stage per iteration
  fal_calls.jsonl      every paid call with its estimated cost
```

`k` counts loop iterations from 0. A stage reads only earlier stages of the same iteration (and, for the
target, the previous iteration's target as a look reference).

## case.json

```json
{
  "name": "attic",
  "created": "2026-11-11T10:00:00",
  "input": {"name": "attic.blend", "kind": "blend", "sha256": "..."},
  "camera": "Camera",
  "size": {"width": 1152, "height": 928, "aspect_ratio": "5:4", "resolution": "1K", "size_verified": true},
  "look": "free text: what the photo should look like (style, materials, time of day)",
  "settings": {"budget_usd": 4.0, "target_level": null, "group_by": "object"},
  "iterations": {}
}
```

- `input.kind`: `spec` (JSON scene spec), `blend`, or `mesh` (glTF/GLB, OBJ, FBX, USD, PLY, STL).
- `size` is what every per-pixel map of the case has. It is the generator's exact output size for the
  case's aspect ratio, so the target needs no resampling. `size_verified` turns true once a target call
  has returned that size; if a call returns a different size the target stage records it, exits with code
  3, and `run` re-renders the g-buffers before trying again.
- `look` is the only place scene style lives. Nothing in `skills/` names a room, an object or a material.

## Objects

- One mesh object = one unit of everything downstream: one id, one UV map, one material, one row in every
  per-object table. `objects.json` lists `name`, `id`, `category`, `description`, bbox and counts.
- `id` = Blender `pass_index`, unique, >= 1, never reused across iterations. 0 means no object.
- `category` is free text. Three values are read by tools: `structure` (large enclosing surfaces: used
  when judging albedo flatness and filling surfaces the camera never saw), `exterior` (geometry outside
  an opening) and anything else (treated the same).
- `description` goes into the target prompt. Write it for an image model: material, colour, finish.
- UV map `b2r_uv`: non-overlapping islands in [0,1] with a margin, made at ingest. Existing UV maps are
  left in place and not used.

## Per-pixel maps

All are `size.width` x `size.height`, row 0 = top, pixel centres at +0.5, registered to each other.

| File | Type | Meaning |
|---|---|---|
| `id.png` | 16-bit | object `pass_index`, 0 = no object (open sky) |
| `position.exr` | float32 RGB | world XYZ of the first hit, metres |
| `depth.exr` | float32 | planar camera-space Z, metres, 0 where no object |
| `depth.png` | 16-bit | inverse depth, nearest = white (conditioning) |
| `normal.exr` | float32 RGB | world-space unit normal |
| `normal_cam.png` | 8-bit | camera-space normal, `n*0.5+0.5`, X right, Y up, Z towards the camera |
| `beauty.png` | 8-bit sRGB | grey render (conditioning and the LPIPS baseline) |
| `albedo.png` | 8-bit sRGB | base colour |
| `roughness.png`, `metallic.png` | 8-bit linear | 0..1 |
| `height.png` | 8/16-bit | relative height, mid-grey = no change |

The data passes are one sample at the pixel centre: no anti-aliasing across object edges, no denoising.
OpenCV returns colour files as BGR and needs `OPENCV_IO_ENABLE_OPENEXR=1` (`common.setup_model_env()`).

## Camera

`camera.json`: `width`, `height`, `fx`, `fy`, `cx`, `cy` in pixels, `fov_x_deg`, `fov_y_deg`,
`cam_to_world`, `world_to_cam` (4x4, Blender camera axes: -Z forward, +Y up, +X right).
Projection of a world point P: `p = world_to_cam @ P`, `x = fx * p.x / -p.z + cx`, `y = cy - fy * p.y / -p.z`.

## Colour

Renders that are compared with the target use the Standard view transform, exposure 0, no look, so
`sRGB(render) ~ target`. Light fitting linearises the target with the sRGB EOTF.

## Lights

`lights.json` lists every light of the fitted scene: `name`, `type`, `virtual`, `strength` with its
unit, colour and its share of image energy. `virtual: true` marks a light that stands for sources the
camera cannot see; it is hidden from camera and glossy rays and a person may delete it. Lights that were
already in the input scene are fit candidates like any other and keep their names.

## Exit codes

0 success. 3 from `target`: the generator's size for this aspect ratio was learned and differs from
the g-buffers; re-run `gbuffers`, then `target`. Any other non-zero code is a failure: the stage wrote
nothing that could be mistaken for a result.

## Honesty rules in code

- A failed model call or render stops the stage. No stage writes a stand-in.
- Nothing from the target is composited into a render. The target reaches the scene only as textures in
  object UV space and as numbers in a light fit.
- Texels the camera never saw are filled from the same object's seen texels, and the fractions are
  recorded in `coverage.json`.
