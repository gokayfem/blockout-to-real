# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, Copilot, Gemini) working on this repository. To use the
skill rather than develop it, see the README.

## Layout

```text
skills/blockout-to-real/        the skill (this is what gets installed)
  SKILL.md                      entry point: input routing, workflow, hard rules, reference routing
  references/                   inputs, target, decompose, backproject, lighting, loop, conventions, lessons
  scripts/b2r.py                CLI: doctor | new | gbuffers | target | decompose | backproject | relight |
                                relight-select | residual | detail | critique | edit | run | finalize | report |
                                render | inspect | status
  scripts/b2r/                  importable library (the user's Python: numpy, scipy, opencv, torch, lpips; fal-client)
    common      the data contract in code: case folders, stages, sizes, finding Blender (dependency-free)
    falio       every paid call: budget, log in fal_calls.jsonl, hard failure, no stand-ins
  scripts/b2r/ also holds       case (new), target, target_prompts, decompose, decompose_fetch, relight_solve,
                                select_relight, residual, critique, edit_tools, finalize, report
  scripts/blender/              scripts run inside Blender 5.2+ (blockout, ingest, gbuffers, backproject, relight,
                                detail, edit, render_view, inspect_blend)
templates/ (in the skill)       CONTRACT.md, acceptance.json, spec.example.json (copied into or used by cases)
tests/                          python3 tests/run_tests.py [--slow]; CI in .github/workflows/checks.yml
adapters/                       copy-in prompts for tools without skill support
gallery/                        README images from the first run
.claude-plugin/                 Claude Code plugin + marketplace manifests
```

## Commands

```bash
python3 tests/run_tests.py                                  # 22 fast tests, no Blender or fal
python3 tests/run_tests.py --slow                           # + the LPIPS test (torch)
python3 skills/blockout-to-real/scripts/b2r.py --help
python3 skills/blockout-to-real/scripts/b2r.py doctor
```

Blender stages run as `blender -b --factory-startup --python scripts/blender/<stage>.py -- --case CASE --iter K`;
Python stages as `python3 -m b2r.<stage> --case CASE --iter K`. `b2r.py <stage> CASE --iter K` does either.

## Rules for changes

- Input-agnostic: no hardcoded rooms, objects, materials, names, paths or camera poses in `skills/`. Scene style
  lives in a case's `case.json` `look` and its objects' descriptions, never in this repository.
- Keep the data contracts in `references/conventions.md` stable (case layout, map encodings, camera, ids, exit
  codes); `scripts/b2r/common.py` is their code form and every stage depends on it.
- Honesty in code: a failed model call or render stops the stage and writes nothing that could pass for a
  result. Nothing from the target is composited into a render; it reaches the scene only as UV textures and as
  numbers in a light fit. Unseen texels are filled from the same object and the fractions recorded.
- No outputs in the repository (cases, renders, `.blend` files, reports): cases live outside it (`cases/` is
  git-ignored). The README images in `gallery/` are the only images.
- Every new module gets a test in `tests/` against a closed form or a synthetic scene with known materials and
  lights. Tests run without fal.
- No secrets in files or logs. `FAL_KEY` is read from the environment and never printed.
