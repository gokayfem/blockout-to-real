# Lighting

`b2r.py relight CASE --iter K` replaces the scene's lights with lights fitted against the target, and
`b2r.py relight-select CASE --iter K` picks between two fits by LPIPS. The result is ordinary Blender lights
with ordinary units, which a person can move, retint or delete.

## Method

Cycles is the forward model. Each candidate light is rendered alone at unit white strength with the real
materials, at half resolution, into a linear basis image. Because rendering is linear in light intensity, the
target (linearised with the sRGB EOTF) is a non-negative sum of the bases:

```text
target_lin[c] ~= sum_i w_ic * basis_i[c]      per colour channel c, w_ic >= 0
```

The weights are solved with non-negative least squares inside an iteratively reweighted loop on the relative
error with a Huber weight, on object pixels whose 2x2 neighbourhood has one id and whose target is not
saturated. Then:

1. drop candidates whose basis is black;
2. solve with all candidates;
3. keep the strongest sun of each group (suns shining in through openings, suns that only light exterior
   geometry), drop the other suns, re-solve;
4. prune: repeatedly drop the light whose removal raises the error least, if that rise is below the pruning
   margin (1% by default) and the total rise stays below 3%; virtual lights obey the same rule;
5. clamp tints (no channel below 1/2.5 of the strongest), refit one scalar per light;
6. convert to Blender units (sun W/m², area and point W, world strength) and delete zero-weight candidates.

## Candidates

The world (sky); area lights in openings; a grid of sun directions; one point light per emitter found in the
target (`lighting.json`); the lights already in the input scene, which keep their names; and virtual fills.
A **virtual** light stands for sources the camera cannot see (a ceiling bounce, a lamp behind the camera). It
is hidden from camera and glossy rays, marked `virtual: true` in `lights.json` with its share of image energy,
and a person may delete it.

## Selection by LPIPS

The fit minimises a linear error. A weak sun that lowers it by 1% can still paint hard-edged patches the
target does not have; the fit cannot see that, LPIPS can. So the solve runs at two pruning margins (1% and 2%,
reusing the basis renders), both scenes are re-rendered at full quality, and the one with the lower LPIPS
against the target is kept. Both scores and the choice are in `relight_selection.json`.

## What to expect, and to report

- First run: R² about 0.84 against the target; final lights a window area light (34% of image energy), the
  sky (11%) and a virtual ceiling fill (55%).
- A large virtual share means the visible sources cannot explain the target's brightness. The re-render will
  have softer shadows and weaker contact shadows than the target. Report the share; `finalize` renders the
  scene without virtual lights so the dependency can be seen.
- Look at `fit_preview.png` and the open-sky pixels: a black sky or hard patches on the floor are failures
  that the fit's own error does not flag.

## Relighting the result

```bash
python3 skills/blockout-to-real/scripts/b2r.py render CASE/final/scene_final.blend dusk.png \
  --light-scale 'virtual_*=0.1' --light-color window_light=1,0.7,0.45
```

`render` takes `--cam-loc` / `--cam-target` for a moved camera and `--light-scale` / `--light-color` by light
name or pattern (`world` addresses the sky). Nothing is saved back into the `.blend`.
