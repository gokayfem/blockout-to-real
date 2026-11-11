# Changelog

## [0.2.0] — 2026-11-11

- `timelapse` command: a time-lapse video for any finished case. `blender/timelapse_frames.py` renders the
  final `.blend` with Cycles (materials added object by object, each fitted light alone, a camera orbit about
  the view pivot, the fitted rig fading to a warm evening) into `CASE/timelapse/frames/`, and `b2r/timelapse.py`
  composes `timelapse/timelapse.mp4` and `thumb.jpg`: the photo and its maps on the left, the scene on the
  right, with the step, the object or light being shown and the LPIPS. A step whose inputs are missing is left
  out and named; nothing stands in for it. Needs ffmpeg on PATH (`doctor` reports it).
- Input files are treated as untrusted:
  - object and light names are restricted to letters, digits, space, `_`, `.` and `-` (renamed at ingest,
    rejected in a JSON spec or an edit, checked again wherever a path is built from a name). Before, a name such
    as `../../x` in a scene file made later stages write texture files outside the case folder;
  - ingest removes text blocks and drivers, so scripts stored in an input `.blend` no longer travel into the
    blockout and the final `.blend`; `inspect` flags any that a file contains;
  - Blender is started with `--disable-autoexec` and every file is opened with `use_scripts=False`, so this no
    longer depends on `--factory-startup` alone;
  - free text taken from the input (material, image and object names) is cut to one printable line, and
    `SKILL.md` tells the agent to read it as labels, never as instructions.
- README section on security and privacy (what is uploaded to fal, untrusted inputs, `edit --script`);
  `.env` files are git-ignored.
- Seven more fast tests (the time-lapse composer on a synthetic case; names and input text).

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
