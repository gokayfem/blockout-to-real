# The loop

After the light fit, the re-render is compared with the target. Small errors become bump detail; large ones
go to a fresh reviewer whose list becomes geometry edits for the next iteration. The loop runs up to three
iterations (0, 1, 2) or until LPIPS stops improving.

## Residual

`b2r.py residual CASE --iter K` scores the Cycles re-render against the target, and the grey render against
the target as a baseline:

- LPIPS (AlexNet, the headline; VGG reported too), PSNR, SSIM in `lpips.json`;
- a spatial LPIPS error map on a fixed colour scale, so maps compare across iterations;
- per object (`per_object.json`): mean spatial LPIPS for re-render and blockout, the remaining-gap ratio
  (re-render over blockout), mean colour difference (dE76) and signed lightness difference, and a class:
  **large** when the gap ratio or the colour difference is above its threshold, **small** otherwise. Objects
  with too few pixels are always small, so a tiny object never drives a geometry edit;
- `contact_sheet.png`: blockout, target, re-render, error map and the per-object table. Look at it.

## Detail

`b2r.py detail CASE --iter K` turns the band-limited (about 2 to 12 px) luminance detail that the target has
and the re-render lacks into a height map, projects it into each small object's UV space, and adds it as a
Bump node after the existing normal map. It is a heuristic, not measured geometry: printed patterns, image
noise and small misregistrations become "relief" too, and it is fitted at this one camera. So it is kept per
object only where that object's LPIPS does not rise by more than 0.002, and switched off entirely if the total
does not improve. Dropped objects keep the node at strength 0. In the first run detail took about 0.013 off
LPIPS in each iteration and was always dropped on the side table and the framed picture.

## Critique: a fresh reviewer

```bash
python3 skills/blockout-to-real/scripts/b2r.py critique CASE --iter K
```

packs the target (A) and the iteration's best re-render (B: with detail if the detail stage improved LPIPS) in
`residual/iter<k>/critique_in/` and prints the reviewer's prompt. Then:

1. Start a **fresh subagent** that has seen nothing of this work: no method, no scores, no file names, no
   object list. It may read only the two packed images. Give it the printed prompt verbatim.
2. It returns JSON: `a_has_b_lacks` (what, kind, bbox, severity 1 to 5, fix_hint), `b_has_a_lacks` (what,
   bbox, severity) and `overall`.
3. Save it: `b2r.py critique CASE --iter K --save REVIEW.json` (validated, written to
   `residual/iter<k>/critique.json`).

A reviewer that has seen the pipeline grades the pipeline. If the agent cannot start a fresh subagent, ask the
user to run the prompt in a new session with only the two images; do not write the review yourself.

## From critique to edits

Sort the items by kind:

| Kind | Goes to |
|---|---|
| `missing_object`, `exterior`, shape items in `b_has_a_lacks` (an extra blob, a wrong silhouette) | geometry edits (below) |
| `material` | usually a decomposition problem (a wrong albedo or roughness source): fix the stage and re-run it; never paint a texture by hand |
| `lighting` | the light fit and `relight-select`; never add a light by hand to match the photo |

Geometry items become an ops file applied to the blockout:

```bash
python3 skills/blockout-to-real/scripts/b2r.py edit CASE --iter K --ops OPS.json   # or --script EDIT.py
```

which writes the next iteration's blockout (`blockout/iter<k+1>/`, with `edits.json`). Place new geometry from
the target's MoGe-3 depth (`decompose/iter<k>/depth_moge.exr` with `camera.json`): unproject the pixels the
reviewer's bbox covers and fit primitives to the points. New objects get new ids, a category and a
description; existing ids are never reused. Then `b2r.py run CASE --iter K+1`.

Refuse any fix hint that would put the target into the scene: an image backdrop behind a window, a photo on a
plane, an emissive card. The first reviewer suggested exactly that for the view through the window. Build
geometry instead, and if primitives cannot make it convincingly (foliage, hair, folds), say so in the report.

## When to stop

- After iteration 2, or
- when the new iteration's LPIPS is not below the last one's, judged on the matrix of every scene against every
  target (the targets drift), or
- when the remaining large items are ones geometry cannot fix.

`finalize` picks the final scene as the one with the lowest mean LPIPS over all targets. In the first run that
was iteration 1 (mean 0.145, against 0.171 and 0.150).

## The blind judge

After `b2r.py finalize CASE`, `b2r.py critique CASE --judge` packs three neutrally named images in
`final/judge_in/` (`reference_1`, `reference_2`, `candidate`: the grey blockout, the target and the final
re-render; the names do not say which is which) and prints the judge's prompt. Start another **fresh
subagent** that may read only those three, give it the printed prompt, and save its answer with
`b2r.py critique CASE --judge --save JUDGE.json` (validated, stored as `final/judge.json`). It answers which reference the candidate is closer to, a 0 to 10
score, whether it reads as a photo or CG, the CG giveaways with a bbox and severity, and what passes as real.
Report its tells as they are; the first judge's main tell (the blob tree, severity 5) is the headline of that
run's limitations.
