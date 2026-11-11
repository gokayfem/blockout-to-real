---
applyTo: "**/*.py"
---

Copy this file into your repository's `.github/instructions/`. For agent mode, also install the skill with
`gh skill install gokayfem/blockout-to-real blockout-to-real`.

- Make grey scenes real with the `blockout-to-real` skill: `b2r.py new` → write `look` and descriptions →
  `gbuffers` → `target` (check the sweep) → `decompose` → `backproject` → `relight` → `relight-select` →
  `residual` → `detail` → `critique` (fresh reviewer) → `edit` → next iteration → `finalize` → `report`.
- Nothing from the target is composited into a render; it reaches the scene only as UV textures and a light fit.
- Paid calls go through the case budget and are logged; `FAL_KEY` is never printed.
