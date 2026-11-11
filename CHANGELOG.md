# Changelog

## [0.1.0] — 2026-11-11

First version.

- An agent skill that turns any grey 3D scene (a `.blend`, a glTF/GLB, OBJ, FBX, USD, PLY or STL file, a JSON
  scene spec, or an idea the agent writes as a spec) into the same scene with ordinary PBR materials and fitted
  lights: g-buffers of one fixed camera, a photoreal target from a five-level conditioning ladder, intrinsic
  maps, back-projection into each object's own UV textures, a Cycles light fit with pruning, and a residual
  loop (LPIPS, per-object errors, bump detail, a fresh reviewer's list applied as geometry edits).
- `b2r.py` CLI with case scaffolding, a fal budget ledger, every stage on its own or chained with `run`,
  `critique` packing for a fresh reviewer, `edit` for geometry edits, `finalize` (final `.blend`, scene-by-target
  LPIPS matrix, novel view, relights, contact sheets), `critique --judge` for the blind judge, `report`,
  `render` and `inspect`.
- 22 fast tests and one slow LPIPS test (`tests/run_tests.py`, no pytest); the fast ones run in CI
  on Linux without Blender, fal or a GPU.
- Adapters for Claude Code, Codex, Cursor, Copilot, Gemini CLI and chat assistants.
- Built from a first full run on one interior room (three iterations, LPIPS-AlexNet 0.463 for the grey
  blockout to 0.134 for the final re-render against its own target); its failures are in
  `skills/blockout-to-real/references/lessons.md`.
