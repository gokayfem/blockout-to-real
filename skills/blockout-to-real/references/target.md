# Target

`b2r.py target CASE --iter K` turns the grey render of the fixed camera into a photo of the same view. The
photo is the only source of the look: every material and every light later in the pipeline is fitted to it.
Nothing from it is ever pasted into the scene.

## The generator

Nano Banana 2 (`fal-ai/nano-banana-2/edit`, a closed model, about $0.08 per image estimated). It takes images
and a prompt and has no numeric conditioning strength, so "conditioning strength" is a ladder of how much
geometry it is given and how strictly the prompt binds it. Output size is fixed per aspect ratio: the case is
rendered at that exact size (`case.json` `size`), so the target needs no resampling. Measured at 1K: 5:4 is
1152x928 and 16:9 is 1376x768. Other aspect ratios are learned on first use: the stage records the returned
size, exits with code 3, and `run` re-renders the g-buffers and tries again. Nothing is cropped or padded.

## The ladder

| Level | Images given | Prompt |
|---|---|---|
| L1 | inverse depth only | a photo whose layout roughly follows the depth map |
| L2 | grey render | the render as a layout reference; furniture may be restyled |
| L3 | grey render | the same room; keep every object where it is, same size, shape and silhouette; add, remove or move nothing |
| L4 | grey render, depth, camera-space normals | L3, consistent with the depth and normal maps |
| L5 | as L4 plus the colour-coded id map | every coloured region becomes exactly one object, outline on the region's edges; more model thinking |

Every level's prompt lists the objects with their `description` and where they sit in the frame ("lower left
of the frame"), then the case's `look`. Objects that are partly hidden get an explicit clause ("stays partly
hidden behind the armchair exactly as in the render"); without it the first run's side table was pulled into
view.

## The rule, and its blind spot

Each level is scored by how well the target's edges line up with the object-id boundaries: Canny edges on
luminance (Gaussian sigma 2 px), recall of id-boundary pixels within 3 px, and mean distance. Two references
calibrate it: the grey render itself (an upper bound) and each image flipped horizontally (chance). A level
passes when

- its recall is at least 0.9 times the best level's,
- its mean distance is at most the grey render's plus 0.35 of the gap to chance,
- its recall beats its own flipped image by at least 0.2, and
- every non-structure object with at least 400 boundary pixels has a recall of at least 0.4.

The **weakest** passing level is kept: stronger conditioning holds geometry better but tends to keep the
blockout's flat, grey look. If none passes, the stage exits 2 and writes no target.

The rule only checks that outlines are found. It cannot see an object that moved, grew, was recoloured or
appeared while the other outlines still match: in the first run it passed an enlarged painting at L2 (about
1.4 times taller than its frame). **Look at the sweep sheet and the per-level overlays every time.** When the
rule's pick is wrong, override it and say why:

```bash
python3 skills/blockout-to-real/scripts/b2r.py target CASE --iter K --set-level L3 --reason "L2 enlarged the painting"
```

The override and its reason go into `target/iter<k>/choice.json` next to the rule's own pick.

## Iterations

Iteration 0 runs the whole ladder and records the chosen level in `case.json` (`settings.target_level`).
From iteration 1 on, only that level runs, and the previous target is passed as a look reference
(`--no-reference` turns that off). That kept the look stable for one
iteration and then compounded: by the third target the image was harsher and more textured (LPIPS 0.086 from
the first). Because the target moves, every scene is scored against every target (`finalize`), and the final
scene is chosen on the mean over all targets.

## Your own photo

`target CASE --image PHOTO.png` uses an existing photoreal image of exactly this view instead of generating
one: a render from another tool, a paint-over, a photo taken from the same camera. It must match the case's
`size` and line up with the id mask; look at the overlay before going on. Decomposition still needs fal.

## What to look for on the sweep sheet

| On the overlay | Means | Do |
|---|---|---|
| an object's photo edges sit off its id outline on one side | moved or resized | stronger level, or tighten that object's description |
| an object in the photo with no id region | added | stronger level; the prompt already forbids it |
| a region with no object in the photo | removed or hidden | describe it better; L5 binds regions to objects |
| everything lines up but the photo looks like a clean render | too strong | the weakest passing level is usually right; check `look` |
| the view through an opening is a flat colour | nothing outside the opening in the blockout | add exterior geometry (`category: exterior`) or describe the view in `look` |
