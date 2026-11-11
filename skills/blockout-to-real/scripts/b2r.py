#!/usr/bin/env python3
"""blockout-to-real CLI: any grey 3D scene -> the same scene with real PBR materials and lights.

  b2r.py doctor                                   check Python packages, Blender, FAL_KEY (never printed)
  b2r.py new CASE --input SCENE [--look TEXT]     scaffold a case from a .blend, a mesh file or a JSON scene spec
                  [--camera NAME] [--budget USD]  (the input is copied with its sha256; a grey blockout is built)
  b2r.py gbuffers CASE [--iter K]                 depth, normals, position, object ids, grey beauty (EXR + PNG)
  b2r.py target CASE [--iter K] [--image PNG]     photoreal image of the same view (conditioning ladder), or your own
  b2r.py decompose CASE [--iter K]                albedo, roughness, metallic, normals, depth check, lighting estimate
  b2r.py backproject CASE [--iter K]              maps -> each object's UV textures, Principled BSDF materials
  b2r.py relight CASE [--iter K]                  replace the lights by a fit against the target (Cycles basis renders)
  b2r.py relight-select CASE [--iter K]           keep the light fit whose re-render has the lower LPIPS
  b2r.py residual CASE [--iter K]                 LPIPS, per-object error map, contact sheet
  b2r.py detail CASE [--iter K]                   small errors -> bump detail, kept only where it helps
  b2r.py critique CASE [--iter K]                 pack target + re-render for a FRESH reviewer; prints its prompt
  b2r.py edit CASE --iter K --ops OPS.json        apply geometry edits -> the next iteration's blockout
                  (or --script EDIT.py)
  b2r.py run CASE [--iter K] [--from S] [--to S]  the looped stages of one iteration, in order
  b2r.py finalize CASE                            final .blend, LPIPS matrix, novel view, relight, contact sheets
  b2r.py report CASE                              REPORT.md
  b2r.py timelapse CASE [--skip-render]           time-lapse video: materials object by object, each light alone,
                  [--width PX] [--samples N]      a camera move, a relight (Cycles frames, then ffmpeg)
  b2r.py render BLEND OUT.png [...]               render any .blend (moved camera, scaled / tinted lights)
  b2r.py inspect BLEND OUT.json                   prove a .blend is ordinary meshes, materials and lights
  b2r.py status CASE
Stages S for --from/--to: gbuffers target decompose backproject relight relight-select residual detail
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from b2r import common  # noqa: E402

# name -> ("python", module) | ("blender", script file). Python modules expose main(argv) -> int.
STAGE_CMDS = {
    "blockout": ("blender", "blockout.py"),
    "ingest": ("blender", "ingest.py"),
    "gbuffers": ("blender", "gbuffers.py"),
    "target": ("python", "b2r.target"),
    "decompose": ("python", "b2r.decompose"),
    "backproject": ("blender", "backproject.py"),
    "relight": ("blender", "relight.py"),
    "relight-select": ("python", "b2r.select_relight"),
    "residual": ("python", "b2r.residual"),
    "detail": ("blender", "detail.py"),
    "critique": ("python", "b2r.critique"),
    "edit": ("blender", "edit.py"),
    "finalize": ("python", "b2r.finalize"),
    "report": ("python", "b2r.report"),
    "timelapse": ("python", "b2r.timelapse"),
}
RUN_ORDER = ["gbuffers", "target", "decompose", "backproject", "relight", "relight-select", "residual", "detail"]
EXIT_SIZE_CHANGED = 3  # the target stage learned the generator's real output size; g-buffers must be redone


def say(*a):
    print(*a, flush=True)


def child_env() -> dict:
    env = dict(os.environ)
    env["B2R_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def stage_command(name: str, args: list[str]) -> list[str]:
    kind, target = STAGE_CMDS[name]
    if kind == "python":
        return [sys.executable, "-m", target, *args]
    blender = common.find_blender()
    if not blender:
        raise SystemExit("Blender not found: set $BLENDER or put `blender` on PATH (b2r.py doctor)")
    return [blender, "-b", "--factory-startup", "--disable-autoexec", "--python", str(common.BLENDER_DIR / target), "--", *args]


def run_stage(name: str, case: str, extra: list[str], log: Path | None = None) -> int:
    cmd = stage_command(name, ["--case", case, *extra])
    if log is None:
        return subprocess.run(cmd, env=child_env()).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        return subprocess.run(cmd, env=child_env(), stdout=f, stderr=subprocess.STDOUT).returncode


def pop_opt(args: list[str], name: str, default=None):
    if name in args:
        i = args.index(name)
        val = args[i + 1]
        del args[i:i + 2]
        return val
    return default


def cmd_doctor(_args) -> int:
    ok = True
    for mod, need in (("numpy", True), ("scipy", True), ("cv2", True), ("PIL", True), ("skimage", True),
                      ("torch", True), ("lpips", True), ("fal_client", False), ("diffusers", False)):
        try:
            __import__(mod)
            say(f"  ok       {mod}")
        except ImportError:
            say(f"  {'MISSING ' if need else 'optional'} {mod}")
            ok &= not need
    b = common.find_blender()
    if b:
        v = subprocess.run([b, "-b", "--version"], capture_output=True, text=True).stdout.splitlines()
        say(f"  ok       blender: {v[0] if v else b}")
    else:
        say("  MISSING  blender (set $BLENDER): every geometry, render and material stage needs it")
        ok = False
    import shutil
    say(f"  {'ok      ' if shutil.which('ffmpeg') else 'optional'} ffmpeg {'' if shutil.which('ffmpeg') else '(only the timelapse command needs it)'}")
    from b2r import falio
    say(f"  {'ok      ' if falio.have_key() else 'missing '} FAL_KEY "
        f"{'(set; not shown)' if falio.have_key() else '(needed for target and decompose; `target --image` replaces the generated target)'}")
    return 0 if ok else 1


def cmd_status(args) -> int:
    from b2r import falio
    case = args[0]
    cfg = common.load_case(case)
    root = common.case_dir(case)
    say(f"case {root.name}: input {cfg.get('input', {}).get('name')} ({cfg.get('input', {}).get('kind')}), "
        f"size {cfg['size']['width']}x{cfg['size']['height']} ({cfg['size'].get('aspect_ratio')})")
    for k in range(common.MAX_ITERATIONS):
        done = [s for s in common.STAGES if (root / s / f"iter{k}").exists() and any((root / s / f"iter{k}").iterdir())]
        if done:
            say(f"  iter {k}: {', '.join(done)}")
    say(f"  fal: est. ${falio.spent(case):.2f} of ${cfg.get('settings', {}).get('budget_usd', falio.DEFAULT_BUDGET_USD):.2f} budget")
    return 0


def cmd_run(args) -> int:
    case = args.pop(0)
    k = pop_opt(args, "--iter", "0")
    first = pop_opt(args, "--from", RUN_ORDER[0])
    last = pop_opt(args, "--to", RUN_ORDER[-1])
    todo = RUN_ORDER[RUN_ORDER.index(first): RUN_ORDER.index(last) + 1]
    logs = common.case_dir(case) / "logs"
    i = 0
    while i < len(todo):
        name = todo[i]
        extra = ["--iter", k] + (args if name == "target" else [])
        log = logs / f"iter{k}_{name}.log"
        t = time.time()
        say(f"[run] {name} ...")
        rc = run_stage(name, case, extra, log)
        if name == "target" and rc == EXIT_SIZE_CHANGED:
            say("[run] target returned a different size for this aspect ratio: re-rendering g-buffers at that size")
            rc = run_stage("gbuffers", case, ["--iter", k], logs / f"iter{k}_gbuffers.log")
            if rc == 0:
                rc = run_stage("target", case, extra, log)
        if rc != 0:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
            say(f"[run] {name} FAILED (exit {rc}) after {time.time() - t:.0f}s; log: {log}\n{tail}")
            return rc
        say(f"[run] {name} ok in {time.time() - t:.0f}s")
        i += 1
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        say(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "doctor":
        return cmd_doctor(args)
    if cmd == "status":
        return cmd_status(args)
    if cmd == "run":
        return cmd_run(args)
    if cmd == "new":
        from b2r import case as case_mod
        return case_mod.cmd_new(args)
    if cmd in ("render", "inspect"):
        blender = common.find_blender()
        if not blender:
            raise SystemExit("Blender not found: set $BLENDER")
        script = "render_view.py" if cmd == "render" else "inspect_blend.py"
        if len(args) < 2:
            raise SystemExit(f"usage: b2r.py {cmd} BLEND OUT [...]")
        rest = ["--blend", args[0], "--out", args[1], *args[2:]]
        return subprocess.run([blender, "-b", "--factory-startup", "--disable-autoexec", "--python", str(common.BLENDER_DIR / script),
                               "--", *rest], env=child_env()).returncode
    if cmd in STAGE_CMDS:
        if not args:
            raise SystemExit(f"usage: b2r.py {cmd} CASE [--iter K] ...")
        return run_stage(cmd, args[0], args[1:])
    raise SystemExit(f"unknown command {cmd!r}; see b2r.py --help")


if __name__ == "__main__":
    sys.exit(main())
