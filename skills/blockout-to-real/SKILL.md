---
name: blockout-to-real
description: >-
  Turns any grey 3D scene (a .blend with or without a camera, a glTF/GLB, OBJ, FBX, USD, PLY or STL file, a
  JSON scene spec of primitives, or an idea written as a spec first) into the same scene with ordinary PBR
  materials and fitted lights that re-render like a photo from its camera: a photoreal target of the fixed view
  (Nano Banana 2, conditioning ladder checked against the object-id mask), albedo, roughness, normals and metric
  depth from the target, back-projected into each object's own UV textures as Principled BSDF materials, lights
  fitted through Cycles basis renders, then a measured loop (LPIPS, per-object errors, bump detail, a fresh
  reviewer's list applied as geometry edits) and a blind judge. Ships a CLI (`scripts/b2r.py`) for case
  scaffolding, fal calls with a budget, every stage, contact sheets, relighting and an editable final .blend.
  Use when the user wants a blockout, greybox, clay render or LLM-built Blender scene made to look real, PBR
  materials and lighting for an existing scene "like this photo", or a check of how close a scene renders to a
  reference image of the same view. Not for single objects or characters (use image-to-sculpt), animation or
  several cameras, glass and mirrors, generating a scene from nothing without a layout, or a photo of a
  different viewpoint.
license: MIT
metadata:
  hermes:
    tags:
      - 3d
      - blender
      - pbr
      - materials
      - lighting
      - cycles
      - inverse-rendering
    category: graphics
---

# Blockout 2 Real

A grey scene in, the same scene out with real materials and lights: a photo of its fixed view is imagined
first, then written back into the scene as UV textures, Principled BSDF materials and fitted lights, and the
re-render is measured against that photo. The geometry stays editable and the lights stay ordinary; the target
never reaches the scene as an image.

## Input routing

Any input ends in the same case layout (`b2r.py new CASE --input ...`). Details: `references/inputs.md`.

| Input | Do this | Notes |
|---|---|---|
| A `.blend` with a camera | `new CASE --input scene.blend --camera Camera` | a grey copy is made; the source is untouched |
| A `.blend` without a camera | `new CASE --input scene.blend` | a camera `b2r_camera` frames every object: check the preview |
| glTF/GLB, OBJ, FBX, USD, PLY, STL | `new CASE --input scene.glb` | one object per mesh, or `--group-by collection|parent`; split merged meshes first |
| A JSON scene spec of primitives | `new CASE --input room.json` | metres, Z up, camera in the spec |
| Only an idea | write a spec, then `new` | primitives: furniture and architecture work, foliage does not |
| A photoreal image of this exact view | `target CASE --image PHOTO.png` | replaces the generated target; must be pixel-registered |
| No API key | not supported | `target` (unless `--image`) and `decompose` need fal; there is no offline path |

## Workflow

Post a labelled image at every checkpoint (preview, sweep sheet, decomposition sheet, contact sheet, final
evidence): images first, numbers second. `SKILL` below means `skills/blockout-to-real/scripts/b2r.py`.

1. **Check.** `python3 SKILL doctor`: Python packages, Blender, `FAL_KEY` (never printed).
2. **Scaffold.** `python3 SKILL new CASE --input SCENE [--look TEXT] [--camera NAME] [--budget USD]`. Then
   write, yourself, `case.json` `look` (place, style, time of day, light, lens) and every object's
   `description` (material, colour, finish) and `category` (`structure`, `exterior`, other) in
   `blockout/iter0/objects.json`. They are the whole target prompt; a file's object names say nothing. Look at
   `blockout/iter0/preview.png`, and fill the case's `CONTRACT.md` and `acceptance.json` before the first
   target (`references/inputs.md`).
3. **G-buffers.** `python3 SKILL gbuffers CASE --iter 0`: depth, normals, position, ids, grey beauty.
4. **Target.** `python3 SKILL target CASE --iter 0` runs the five-level ladder and keeps the weakest level
   whose edges match the id mask. **Look at the sweep sheet and the overlays yourself.** The rule cannot see an
   object that moved, grew, appeared or vanished while the outlines still match; when that happens, override:
   `target CASE --iter 0 --set-level L3 --reason "L2 enlarged the painting"` (`references/target.md`). Later
   iterations run the chosen level only, with the previous target as a look reference.
5. **Decompose.** `python3 SKILL decompose CASE --iter 0`: albedo, roughness, metallic, normals, metric depth,
   lighting estimate. Look at the map sheet: shading left in the albedo or a roughness map that looks like a
   light map is a failure (`references/decompose.md`).
6. **Back-project and relight.** `backproject`, `relight`, `relight-select` (each `CASE --iter 0`): UV textures
   and Principled BSDF materials, fills where the camera did not see, lights fitted through Cycles, the fit kept
   by LPIPS (`references/backproject.md`, `references/lighting.md`).
7. **Residual and detail.** `residual`, `detail`: LPIPS, per-object error map, contact sheet; small errors as
   bump detail, kept only where it helps. `python3 SKILL run CASE --iter K` runs steps 3 to 7 in order; to look
   at the sweep before spending on the rest, run `--to target`, check it, then `--from decompose`.
8. **Critique (agentic).** `python3 SKILL critique CASE --iter K` packs the target and the re-render and prints
   a prompt. Start a **fresh subagent** that may read only those two images, give it the printed prompt
   verbatim, and save its JSON with `critique CASE --iter K --save REVIEW.json` (validated, stored as
   `residual/iter<k>/critique.json`). Never write the review yourself (`references/loop.md`).
9. **Edit (agentic).** Turn the critique's geometry items (`missing_object`, `exterior`, wrong shapes) into an
   ops file, placing new geometry from the target's MoGe-3 depth, and apply it:
   `python3 SKILL edit CASE --iter K --ops OPS.json` writes blockout `iter K+1`. Material and lighting items go
   back to their stages, never to hand-painted textures or hand-placed lights. Then `run CASE --iter K+1`.
10. **Stop.** Loop until iteration 2 or until LPIPS stops improving on the scene-by-target matrix.
11. **Finalize and judge (agentic).** `python3 SKILL finalize CASE` writes `final/scene_final.blend`, the LPIPS
    matrix, a novel view, relights and the contact sheets. Then `python3 SKILL critique CASE --judge` packs
    three neutrally named images (blockout, target, final re-render) in `final/judge_in/` and prints the judge's
    prompt. Start another **fresh subagent** that may read only those three, give it the printed prompt, and save
    its JSON with `critique CASE --judge --save JUDGE.json` (stored as `final/judge.json`).
12. **Report and stop.** `python3 SKILL report CASE` writes `REPORT.md`. Present the contact sheets, the LPIPS
    table and matrix, the light list with the virtual share, the judge's tells, and what is weak. The user
    decides what next.

## Hard rules

- **Nothing from the target is composited into a render.** It reaches the scene only as textures in object UV
  space and as numbers in a light fit. No image planes, no backdrops, no emissive cards, no compositor, even
  when a reviewer suggests one (`b2r.py inspect` proves the final `.blend` is ordinary).
- **A failed stage stops.** A failed model call or render writes nothing that could pass for a result; never
  substitute a stand-in, and never claim a stage ran when it did not.
- **The weakest conditioning that holds geometry, checked by eye.** The edge rule picks; you look; you override
  with a recorded reason when an object moved or changed size.
- **Fresh reviewers only.** The critique reviewer and the final judge see only their images under neutral
  names, nothing about the method or scores. Do not review your own re-render.
- **Score the matrix, not one column.** The target is regenerated each iteration and drifts; report every scene
  against every target and pick the final scene on the mean.
- **Say what is virtual and what is filled.** Report the virtual lights' energy share (`lights.json`) and the
  fill fractions (`coverage.json`). A novel view will show the fills.
- **Geometry edits are real geometry.** New objects get new ids, a category and a description; ids are never
  reused. When primitives cannot make what the reviewer asks for, say so.
- **A fix is a re-run.** Fix a stage, then re-run it and everything after it; never patch an output.
- **Paid calls:** every fal call goes through the case budget (`settings.budget_usd`, estimated per call) and is
  logged in `fal_calls.jsonl`. `FAL_KEY` is never printed or written. The first run cost about $1.47.
- **Say what it is not.** Not photo-indistinguishable: from its own camera it reads as a good render, and a
  careful viewer calls it CG. The target comes from a closed API model.

## References

| File | Read when |
|---|---|
| `references/inputs.md` | routing an input; writing `look`, descriptions and categories; writing a spec from an idea |
| `references/target.md` | the ladder, the edge rule and its blind spot, overrides, your own photo, drift |
| `references/decompose.md` | which model gives which map, the checks, the dark-albedo and roughness rules |
| `references/backproject.md` | UV projection, visibility, fills, materials, the round-trip check |
| `references/lighting.md` | the light fit, candidates, pruning, virtual lights, relighting the result |
| `references/loop.md` | residual, detail, the critique and the blind judge, turning items into edits, stopping |
| `references/conventions.md` | case layout, maps, camera, ids, exit codes (for writing new tools) |
| `references/lessons.md` | what went wrong in the first run and why the rules above exist |

Templates: `templates/CONTRACT.md` and `templates/acceptance.json` (copied into every case by `new`),
`templates/spec.example.json` (a JSON scene spec to start from when the input is an idea).
