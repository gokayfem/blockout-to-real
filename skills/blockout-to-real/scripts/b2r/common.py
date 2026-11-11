"""Shared contract for every stage. Dependency-free: imported by Blender's bundled Python
(scripts/blender/*.py) and by the user's Python (scripts/b2r/*.py). See references/conventions.md.

A case is a folder. Nothing about a particular scene lives in this repository; everything a
stage needs comes from the case:

    CASE/case.json                 settings and state (see case.py)
    CASE/<stage>/iter<k>/...       one folder per stage per loop iteration, k = 0, 1, 2

Invocation
    Blender stages   $BLENDER -b --factory-startup --python scripts/blender/<stage>.py -- --case CASE --iter K
    Python stages    python3 scripts/b2r.py <stage> CASE --iter K     (or: python3 -m b2r.<stage> --case CASE --iter K)

Per-pixel maps are all <width> x <height> from case.json "size", row 0 = top:
    id.png          16-bit, value = object pass_index, 0 = no object (open sky)
    position.exr    float32 RGB, world XYZ of the first hit, metres
    depth.exr       float32, planar camera-space Z, metres, 0 where no object
    normal.exr      float32 RGB, world-space unit normal
    normal_cam.png  8-bit camera-space normal, n*0.5+0.5, X right / Y up / Z towards the camera
    camera.json     fx, fy, cx, cy (pixels), cam_to_world, world_to_cam (Blender camera axes)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent  # .../skills/blockout-to-real/scripts
BLENDER_DIR = SCRIPTS_DIR / "blender"

STAGES = ("blockout", "gbuffers", "target", "decompose", "backproject", "residual")
MAX_ITERATIONS = 3

# Aspect ratios the target generator accepts; a case uses the one nearest to its camera.
TARGET_ASPECTS = ("21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16")
DEFAULT_RESOLUTION = "1K"


# ---------------------------------------------------------------- tools on this machine
def find_blender() -> str | None:
    """$BLENDER, then `blender` on PATH, then the usual macOS / Linux locations."""
    cands = [os.environ.get("BLENDER"), os.environ.get("BLENDER_BIN"), shutil.which("blender"),
             "/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/snap/bin/blender"]
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    return None


def python_exe() -> str:
    """The user's Python (numpy, scipy, opencv, torch...). Under Blender, sys.executable is
    Blender's own Python, so the CLI passes the right one in $B2R_PYTHON."""
    return os.environ.get("B2R_PYTHON") or (sys.executable if "blender" not in Path(sys.executable).name.lower()
                                            else shutil.which("python3") or "python3")


def cache_dir() -> Path:
    p = Path(os.environ.get("B2R_CACHE") or Path.home() / ".cache" / "blockout-to-real")
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_model_env() -> None:
    """Call before importing torch / cv2 / huggingface_hub: model weights go to the cache dir,
    MPS falls back to CPU for missing ops, OpenCV may read EXR."""
    c = cache_dir()
    os.environ.setdefault("HF_HOME", str(c / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(c / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------- case folders
def case_dir(case: str | Path) -> Path:
    return Path(case).expanduser().resolve()


def case_json(case: str | Path) -> Path:
    return case_dir(case) / "case.json"


def load_case(case: str | Path) -> dict:
    p = case_json(case)
    if not p.exists():
        raise SystemExit(f"not a case folder (no case.json): {case_dir(case)}. Create one with `b2r.py new`.")
    return read_json(p)


def save_case(case: str | Path, cfg: dict) -> None:
    write_json(case_json(case), cfg)


def stage_dir(case: str | Path, stage: str, iteration: int, create: bool = True) -> Path:
    """CASE/<stage>/iter<k>/"""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    path = case_dir(case) / stage / f"iter{int(iteration)}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def render_size(case: str | Path) -> tuple[int, int]:
    """(width, height) every map of this case has. Set by `new`, corrected by the target stage
    when the generator returns a different size for the case's aspect ratio."""
    s = load_case(case)["size"]
    return int(s["width"]), int(s["height"])


def nearest_aspect(width: float, height: float) -> str:
    want = width / height

    def ratio(a: str) -> float:
        x, y = a.split(":")
        return float(x) / float(y)

    return min(TARGET_ASPECTS, key=lambda a: abs(ratio(a) - want) / max(ratio(a), want))


# ---------------------------------------------------------------- small helpers
def script_args(argv: list[str] | None = None) -> list[str]:
    """CLI args for a script: everything after `--` under Blender, else argv[1:]."""
    argv = list(sys.argv if argv is None else argv)
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    if Path(argv[0]).name.lower().startswith("blender"):
        return []
    return argv[1:]


def write_json(path: Path | str, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def read_json(path: Path | str):
    with open(path) as f:
        return json.load(f)
