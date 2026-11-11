"""Pick between two light-fit variants by what the loop actually measures: LPIPS.

    python -m b2r.select_relight --case CASE --iter K [--margin-b 0.02] [--final-samples N] [--rerender-scale F]

Why: the light fit minimises a linear relative error, which rewards a weak light that shaves
about 1% off the error while painting hard-edged patches a soft-lit target does not have. The
fit cannot see that; LPIPS can. So the same solve is run a second time with a stricter
parsimony margin (reusing the basis images already rendered), both scenes are re-rendered, and
the variant whose re-render is closer to the target (LPIPS-AlexNet) becomes the iteration's scene.

Variant m01 = what the relight stage already wrote (its default margin).
Variant m02 = the stricter margin, written to backproject/iter<K>/variant_m02/.
The winner is copied to the canonical names (scene_lit.blend, lights.json, fit.json,
fit_preview.png, residual/iter<K>/rerender.png and .exr). Both scores and the decision are
recorded in backproject/iter<K>/relight_selection.json. Nothing is deleted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from . import common

CANON = ["scene_lit.blend", "lights.json", "fit.json", "fit_preview.png"]


def _env() -> dict:
    env = dict(os.environ)
    env["B2R_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(common.SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run(cmd: list[str], log) -> None:
    with open(log, "a") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=_env())
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}), see {log}: {' '.join(cmd[:8])} ...")


def _score(case, k: int, rerender, tag: str, log) -> dict:
    _run([sys.executable, "-m", "b2r.residual", "--case", str(case), "--iter", str(k),
          "--rerender", str(rerender), "--tag", tag], log)
    d = common.read_json(common.stage_dir(case, "residual", k) / f"lpips_{tag}.json")
    return d["rerender_vs_target"]


def _light_names(path) -> list[str]:
    return [x["name"] for x in common.read_json(path)["lights"] if x.get("strength") is not None]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="b2r.select_relight")
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--margin-b", type=float, default=0.02)
    ap.add_argument("--final-samples", type=int, default=None, help="samples of variant B's re-render (default: the relight stage's)")
    ap.add_argument("--rerender-scale", type=float, default=None, help="must match the scale variant A was rendered at")
    args = ap.parse_args(argv)
    case, k = common.case_dir(args.case), args.iter

    bp = common.stage_dir(case, "backproject", k)
    res = common.stage_dir(case, "residual", k)
    for name in CANON:
        if not (bp / name).exists():
            raise SystemExit(f"missing {bp / name}: run the relight stage first")
    blender = common.find_blender()
    if not blender:
        raise SystemExit("Blender not found: set $BLENDER")
    logs = case / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"iter{k}_relight_select_detail.log"
    log.write_text("")

    # Variant A: keep a copy of what the relight stage produced.
    va = bp / "variant_m01"
    va.mkdir(exist_ok=True)
    for name in CANON:
        shutil.copyfile(bp / name, va / name)
    shutil.copyfile(res / "rerender.png", va / "rerender.png")
    if (res / "rerender.exr").exists():
        shutil.copyfile(res / "rerender.exr", va / "rerender.exr")

    # Variant B: stricter pruning, same basis images, its own output folder.
    vb = bp / "variant_m02"
    vb.mkdir(exist_ok=True)
    cmd = [blender, "-b", "--factory-startup", "--python", str(common.BLENDER_DIR / "relight.py"), "--",
           "--case", str(case), "--iter", str(k), "--prune-margin", str(args.margin_b), "--prune-total-cap", "0.05",
           "--reuse-basis", str(bp / "basis"), "--out-dir", str(vb), "--rerender-out", str(vb / "rerender.png")]
    if args.final_samples:
        cmd += ["--final-samples", str(args.final_samples)]
    if args.rerender_scale:
        cmd += ["--rerender-scale", str(args.rerender_scale)]
    _run(cmd, log)

    sa = _score(case, k, va / "rerender.png", "relight_m01", log)
    sb = _score(case, k, vb / "rerender.png", "relight_m02", log)
    la, lb = _light_names(va / "lights.json"), _light_names(vb / "lights.json")
    winner = "m02" if sb["lpips_alex"] < sa["lpips_alex"] else "m01"
    wdir = vb if winner == "m02" else va
    for name in CANON:
        shutil.copyfile(wdir / name, bp / name)
    shutil.copyfile(wdir / "rerender.png", res / "rerender.png")
    if (wdir / "rerender.exr").exists():
        shutil.copyfile(wdir / "rerender.exr", res / "rerender.exr")

    out = {
        "rule": "run the light fit at two prune margins; keep the variant whose re-render has the lower "
                "LPIPS-AlexNet against the target",
        "variant_m01": {"lights": la, **sa},
        "variant_m02": {"prune_margin": args.margin_b, "lights": lb, **sb},
        "same_light_set": la == lb,
        "winner": winner,
    }
    common.write_json(bp / "relight_selection.json", out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
