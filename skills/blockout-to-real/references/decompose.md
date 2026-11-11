# Decompose

`b2r.py decompose CASE --iter K` splits the target into maps registered to the g-buffers, plus a lighting
estimate. Raw model outputs are cached in `decompose/iter<k>/raw/`, keyed on the content of the target they
were computed from, so a re-run of the analysis costs nothing and a new target never reads an old cache.

## Sources

| Map | Source | Check before use |
|---|---|---|
| albedo | Marigold V2 (`fal-ai/marigold-v2`); PATINA base colour as the alternative | flatness of `structure` surfaces against the target, and how much of the target's shadow and sun-patch contrast survives (1 = lighting kept, 0 = removed); the lower score wins |
| normals | Marigold V2, MoGe-3, PATINA | agreement with the g-buffer normals (the first run: within about 2°) |
| metric depth | MoGe-3 (`microsoft/moge-3`) with the known field of view | `depth_delta`: where the photo's geometry differs from the blockout |
| roughness | PATINA (`fal-ai/patina`), else Marigold-IID Appearance v1.1 locally | must not track shading inside uniform surfaces; fabric must not be smoother than wood |
| metallic | PATINA, else Marigold-IID | plausibility per object |
| height | PATINA | used only as bump input |
| shading, lighting | target divided by albedo | emitters found in the image (lamp shades, openings) go to `lighting.json` as light candidates |

The first run called each fal model once or twice per iteration: 8 Marigold V2, 4 PATINA and 4 MoGe-3 calls for
three iterations, about $0.35 together (estimates).

## Albedo: the dark-object rule

After the albedo source is chosen, an object's albedo is replaced by the other source's only when the chosen
albedo is near-black (median linear luminance below 0.03) **and** the object is light in the photo by its 25th
percentile (at least 0.06), with a ratio above 4. A dark glossy object with bright reflections is light on
average and dark at the 25th percentile, so it stays dark. Every decision is recorded per object. The first
pass of the first run used the mean and replaced a correct black side table with the photo's colour.

## Roughness

PATINA's roughness followed the lighting on a whole-room photo: bright where the light fell. The check
measures the correlation between roughness and shading inside surfaces that should be uniform (painted walls)
and the order of materials that should not be inverted (fabric rougher than wood). A failing map is replaced
by Marigold-IID Appearance v1.1, run locally through `diffusers` (CUDA, MPS or CPU; install the optional block
of `requirements.txt`). That map is plausible and blurry: glossy small objects lose their gloss. Say so when
the target shows glossy things.

Metallic was 0 everywhere in the first run. That was plausible for its room; it is not evidence that metallic
estimation works on a scene with metal in it.

## Depth

MoGe-3 is given the camera's field of view, so its depth is in metres and comparable with `depth.exr`.
`depth_delta` shows where the target changed the geometry: in the first run it pinpointed the plant, whose
leaves the photo had redrawn, and the new leaves were placed from it. It is the input for geometry edits, not
for textures.

## Outputs

`albedo.png`, `roughness.png`, `metallic.png`, `normal_cam.png`, `height.png`, `depth_moge.exr`,
`depth_delta`, `shading`, `lighting.json`, and a contact sheet of all maps. Look at the sheet: an albedo that
still shows window light on the floor, or a roughness map that looks like a light map, is a failed estimate,
whatever the numbers say.

## Without fal

There is no offline path for this stage. A target you bring with `target --image` still goes through these fal
models.
