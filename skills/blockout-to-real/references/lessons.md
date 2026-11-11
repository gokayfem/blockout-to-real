# Lessons

What went wrong in the first full run of this method, and the rule each failure became. The run was one
interior room (17 objects in the first blockout, 20 in the final scene), one fixed camera, three loop
iterations, Nano Banana 2 for the target. Every number here is from that run's evidence (`summary.json`,
`choice.json`, `critique.json`, `judge.json`). The failures are not specific to rooms.

## Target

- **The edge rule cannot see an object that moved or grew while its outline stayed findable.** The rule
  checks that the target has edges where the object-id mask has boundaries. The first prompt let the
  generator pull the side table out from behind the armchair; at level L2 the wall painting came back about
  1.4 times taller than its blockout frame, its top edge about 70 px above the id outline. Both passed,
  because every boundary still had an edge nearby. Rules: look at the sweep overlays yourself and override the
  pick with `target --set-level` when an object moved, grew, appeared or vanished; record why. Objects that are
  partly hidden get an explicit "stays partly hidden behind X, exactly as in the render" clause in the prompt.
- **Re-editing with the previous target as a reference compounds.** Passing the last target in kept the look
  stable for one iteration; by the third, the target was visibly harsher and more textured (target 0 against
  target 2: LPIPS 0.086; 0 against 1: 0.029). Rule: scores are reported as a matrix, every scene against every
  target, and the final scene is the one with the lowest mean over all targets, not the lowest own-target
  number. In that run the own-target column was flat for iteration 1 (0.136 then 0.134) while the matrix showed
  a real gain (against target 0: 0.136 then 0.118; against target 1: 0.162 then 0.134).

## Decomposition

- **PATINA's roughness followed the lighting.** On a whole-room photo its roughness map tracked the shading
  inside the painted walls. Rule: every roughness source is checked before use (correlation with shading inside
  surfaces that should be uniform; fabric must not come out smoother than wood), and a failing one is replaced
  by Marigold-IID Appearance v1.1 run locally. That map is blurry: the glossy black side-table top still
  rendered grey. Metallic came out 0 everywhere, which was plausible for that room and is not evidence that
  metallic works.
- **A stale cache gave the wrong roughness.** The first pass of iteration 0 read a roughness map computed on an
  earlier provisional target. Rule: every cache is keyed on the content of the image it was computed from, not
  on a file name or an iteration number.
- **A correct dark albedo was "fixed".** A check that replaced near-black albedo with the other estimator's
  colour treated the black lacquered side table as an error, because its reflections made it look light on
  average. Rule: an albedo is replaced only when it is near-black and the object is light in the photo by a low
  percentile (the 25th), not the mean; a dark glossy object stays dark. Each decision is recorded per object.

## Lighting

- **A weak sun shaved 1% off the fit and painted hard patches.** The light fit minimises a linear error, and a
  sun carrying about 2% of the energy lowered it slightly while putting hard-edged diagonal patches on the rug
  that the soft-lit target did not have. The fresh reviewer named them; the fit could not see them. Rule: the
  fit is run at two pruning margins (1% and 2%) and the variant whose re-render has the lower LPIPS is kept
  (`relight-select`). In iterations 0 and 1 the stricter margin dropped the sun and won (0.149 against 0.151 and
  0.150).
- **The sky was pruned to black.** In the first pass the open sky outside the window was not part of the light
  fit (the fit used object pixels only), and it ended up black. Rule: the world (sky) is always a fit
  candidate, and the open-sky pixels are checked on the contact sheet after every fit.
- **The virtual fill carried more than half the light.** The window alone could not explain the target's even
  brightness, so the fit leaned on a hidden ceiling fill: 55% of image energy in the final scene, against 34%
  for the window and 11% for the sky. Shadows came out softer and contact shadows weaker than the target, and
  the judge named both. Rule: `lights.json` marks every virtual light and its energy share, and the report
  states the share. A virtual light is hidden from camera and glossy rays and can be deleted; `finalize`
  renders the scene without it so the dependency is visible.

## Detail

- **The detail bump is partly fitted to the view.** It comes from the target-versus-render residual at the one
  camera, so its gain (about 0.013 LPIPS) would not fully carry over to other views, and on the framed picture
  it made the result worse. Rule: bump detail is applied per object and kept only where that object's own
  LPIPS does not rise by more than 0.002; if the total does not improve, all of it is switched off. The node and
  texture stay in the `.blend` at strength 0 so a person can turn them back on. The side table and the wall
  picture were dropped in every iteration.

## Geometry

- **Blob trees: primitives have a limit.** The reviewer's first item was the missing view through the window;
  its hint was an image backdrop, which the honesty rules forbid (nothing from the target is pasted into the
  scene). A tree of ellipsoids was built instead, and the judge called it the main tell (severity 5). A finer
  tree of 100 to 200 small ellipsoids read as polka dots and made the score worse (iteration 2: 0.175). Rule:
  apply a reviewer's geometry items as real geometry and measure them; when an item needs geometry primitives
  cannot make (foliage, hair, cloth folds), say so in the report instead of adding more primitives.
- **Edits that the reviewer asked for helped when they were real.** The missing garden and a white blob behind
  the plant (a leaf the target did not have) were the top items of the first critique; turning them into meshes
  placed with MoGe-3 depth improved the scene against both fixed targets.

## Novel views

- **Fills on never-seen surfaces show from a moved camera.** Floor hidden by furniture is smeared, a wall the
  camera never saw is a flat mean colour, and the exterior only reads from the original viewpoint. Rule:
  `coverage.json` records per object how much was seen, filled from nearby seen texels, or filled by a mean;
  `finalize` renders a novel view so the fills are shown, not hidden. The result is a one-view reconstruction.

## Process

- **Re-run, do not patch.** When the stale cache, the dark-albedo fix and the weak sun were found, iteration 0
  was re-run on the fixed code and the first pass archived, so every iteration in the table came from the same
  code. Rule: a fix to a stage means re-running that stage and everything after it.
- **A fresh reviewer for every judgement.** The reviewer that writes the critique and the final judge see only
  the images they are given, with neutral names, and nothing about the method or the scores. A reviewer that
  has seen the pipeline grades the pipeline.
- **One number never stands for "real".** LPIPS against a drifting target, PSNR, the reviewer's list and the
  judge's tells answer different questions; report them side by side.
