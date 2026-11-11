"""`b2r.py new`: scaffold a case from any input scene, plus the case helpers the Blender stages share.

    b2r.py new CASE --input PATH [--look TEXT] [--camera NAME] [--budget USD]
                    [--aspect W:H] [--resolution 1K] [--group-by object|collection|parent]

What `new` does
  1. creates CASE/ (refuses if CASE/case.json already exists), copies the input to CASE/input/ with
     its sha256 (a .gltf brings its buffers and images, an .obj its .mtl and textures),
  2. detects the kind: `spec` (.json scene spec, see scripts/blender/blockout.py), `blend`, or `mesh`
     (.glb .gltf .obj .fbx .usd .usda .usdc .usdz .ply .stl),
  3. writes case.json (references/conventions.md) with "size": null unless --aspect was given,
  4. copies templates/CONTRACT.md and templates/acceptance.json into the case,
  5. builds blockout/iter0 with Blender: blockout.py for a spec, ingest.py for a blend or mesh file.
     The camera's aspect ratio is only known inside Blender, so that stage picks the camera and calls
     `decide_size` below, which writes case.json "size" before anything is rendered,
  6. prints what was built and what to do next.

Render size rule (decide_size): the generator aspect ratio is --aspect if given, else the one nearest
the camera's own aspect (common.nearest_aspect), else (no camera aspect is known: a spec without one,
or a mesh file without a camera) the first ratio whose generator size is already measured, so the
first target call needs no size correction. The size comes from nb2_sizes.json when that ratio was
measured ("size_verified": true); otherwise it is a provisional ~1 megapixel size with both sides
multiples of 16 and "size_verified": false; the target stage corrects it when the generator answers.

This module only uses the standard library: Blender's bundled Python imports it too.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import common

SIZES_FILE = Path(__file__).resolve().parent / "nb2_sizes.json"
TEMPLATES_DIR = common.SCRIPTS_DIR.parent / "templates"
CASE_TEMPLATES = ("CONTRACT.md", "acceptance.json")

MESH_EXTS = (".glb", ".gltf", ".obj", ".fbx", ".usd", ".usda", ".usdc", ".usdz", ".ply", ".stl")
GROUP_BY = ("object", "collection", "parent")
# pixel area of a provisional size per generator resolution ("1K" ~ one megapixel)
RESOLUTION_EDGE = {"0.5K": 512, "1K": 1024, "2K": 2048, "4K": 4096}


# ---------------------------------------------------------------- input kinds
def input_kind(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".json":
        return "spec"
    if ext == ".blend":
        return "blend"
    if ext in MESH_EXTS:
        return "mesh"
    raise SystemExit(f"unsupported input type {ext!r}: expected a .json scene spec, a .blend, or one of "
                     f"{' '.join(MESH_EXTS)}")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_path(case: str | Path, cfg: dict | None = None) -> Path:
    cfg = cfg or common.load_case(case)
    return common.case_dir(case) / "input" / cfg["input"]["name"]


def _companions(src: Path) -> list[Path]:
    """Files a mesh format references by relative path (copied next to the input)."""
    out: list[Path] = []
    ext = src.suffix.lower()
    try:
        if ext == ".gltf":
            g = json.loads(src.read_text())
            for key in ("buffers", "images"):
                for item in g.get(key, []):
                    uri = item.get("uri", "")
                    if uri and not uri.startswith("data:") and "://" not in uri:
                        out.append(src.parent / urllib_unquote(uri))
        elif ext == ".obj":
            for line in src.read_text(errors="replace").splitlines():
                if line.startswith("mtllib"):
                    mtl = src.parent / line.split(None, 1)[1].strip()
                    out.append(mtl)
                    if mtl.is_file():
                        for ml in mtl.read_text(errors="replace").splitlines():
                            parts = ml.strip().split()
                            if parts and parts[0].lower().startswith(("map_", "bump", "disp", "norm")):
                                out.append(mtl.parent / parts[-1])
    except (OSError, ValueError, IndexError):
        pass
    return [p for p in out if p.is_file()]


def urllib_unquote(s: str) -> str:
    from urllib.parse import unquote
    return unquote(s)


# ---------------------------------------------------------------- render size
def known_sizes() -> dict:
    return common.read_json(SIZES_FILE) if SIZES_FILE.is_file() else {}


def parse_aspect(a: str) -> float:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*", str(a))
    if not m:
        raise ValueError(f"aspect ratio must look like W:H, got {a!r}")
    return float(m.group(1)) / float(m.group(2))


def default_aspect(resolution: str = common.DEFAULT_RESOLUTION) -> str:
    """Used only when nothing states an aspect ratio: one whose generator size is measured."""
    table = known_sizes().get(resolution) or next(iter(known_sizes().values()), {})
    for a in table:
        if a in common.TARGET_ASPECTS:
            return a
    return "1:1"


def size_for(aspect: str, resolution: str = common.DEFAULT_RESOLUTION) -> dict:
    if aspect not in common.TARGET_ASPECTS:
        raise ValueError(f"aspect {aspect!r} is not one the generator accepts: {common.TARGET_ASPECTS}")
    known = known_sizes().get(resolution, {}).get(aspect)
    if known:
        w, h = int(known[0]), int(known[1])
        verified = True
    else:
        r = parse_aspect(aspect)
        edge = RESOLUTION_EDGE.get(resolution, 1024)
        w = int(round(math.sqrt(edge * edge * r) / 16.0)) * 16
        h = int(round(math.sqrt(edge * edge / r) / 16.0)) * 16
        verified = False
    return {"width": w, "height": h, "aspect_ratio": aspect, "resolution": resolution,
            "size_verified": verified}


def decide_size(cfg: dict, camera_wh: tuple[float, float] | None, source: str) -> dict:
    """The case's render size, chosen once. Returns cfg["size"] if already set; otherwise picks the
    aspect (settings.aspect, else nearest to the camera, else default_aspect), stores and returns it.
    The caller saves cfg."""
    if cfg.get("size") and cfg["size"].get("width"):
        return cfg["size"]
    st = cfg.setdefault("settings", {})
    resolution = st.get("resolution") or common.DEFAULT_RESOLUTION
    if st.get("aspect"):
        aspect, why = st["aspect"], "--aspect"
    elif camera_wh and camera_wh[0] > 0 and camera_wh[1] > 0:
        aspect = common.nearest_aspect(camera_wh[0], camera_wh[1])
        why = f"nearest to the camera aspect {camera_wh[0] / camera_wh[1]:.4f} ({source})"
    else:
        aspect = default_aspect(resolution)
        why = f"no camera aspect known ({source}); default = a ratio with a measured generator size"
    size = size_for(aspect, resolution)
    size["aspect_source"] = why
    if camera_wh:
        size["camera_aspect"] = round(camera_wh[0] / camera_wh[1], 6)
    cfg["size"] = size
    return size


# ---------------------------------------------------------------- running Blender stages
def blender_env() -> dict:
    env = dict(os.environ)
    env["B2R_PYTHON"] = common.python_exe()
    env["PYTHONPATH"] = str(common.SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_blender(script: str, args: list[str], log: Path) -> int:
    blender = common.find_blender()
    if not blender:
        raise SystemExit("Blender not found: set $BLENDER or put `blender` on PATH (b2r.py doctor)")
    cmd = [blender, "-b", "--factory-startup", "--python", str(common.BLENDER_DIR / script), "--", *args]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        return subprocess.run(cmd, env=blender_env(), stdout=f, stderr=subprocess.STDOUT).returncode


# ---------------------------------------------------------------- new
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="b2r.py new", description="scaffold a case from a scene")
    ap.add_argument("case", help="case folder to create")
    ap.add_argument("--input", required=True, help=".json spec, .blend, or mesh file")
    ap.add_argument("--look", default="", help="free text for the image prompt: style, materials, light")
    ap.add_argument("--camera", default=None, help="camera object to use (default: the scene camera)")
    ap.add_argument("--budget", type=float, default=None, help="fal budget in USD for this case")
    ap.add_argument("--aspect", default=None, help=f"generator aspect ratio, one of {' '.join(common.TARGET_ASPECTS)}")
    ap.add_argument("--resolution", default=common.DEFAULT_RESOLUTION, help="generator resolution (default 1K)")
    ap.add_argument("--group-by", default="object", choices=GROUP_BY,
                    help="blend/mesh input: one object per mesh object, per collection, or per top parent")
    ap.add_argument("--samples", type=int, default=None, help="preview render samples")
    return ap


def cmd_new(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    src = Path(args.input).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"input not found: {src}")
    kind = input_kind(src)
    if args.aspect and args.aspect not in common.TARGET_ASPECTS:
        raise SystemExit(f"--aspect must be one of {common.TARGET_ASPECTS}")
    root = common.case_dir(args.case)
    if (root / "case.json").exists():
        raise SystemExit(f"{root} already has a case.json; pick a new folder (cases are never overwritten)")
    root.mkdir(parents=True, exist_ok=True)
    (root / "input").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)

    dst = root / "input" / src.name
    shutil.copyfile(src, dst)
    companions = []
    for c in _companions(src):
        rel = c.resolve().relative_to(src.parent) if c.resolve().is_relative_to(src.parent) else Path(c.name)
        (root / "input" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c, root / "input" / rel)
        companions.append({"name": str(rel), "sha256": sha256_file(c)})

    from . import falio  # lazy: falio is not needed inside Blender
    cfg = {
        "name": root.name,
        "created": _dt.datetime.now().replace(microsecond=0).isoformat(),
        "input": {"name": src.name, "kind": kind, "sha256": sha256_file(dst)},
        "camera": args.camera,
        "size": None,
        "look": args.look or "",
        "settings": {"budget_usd": float(args.budget if args.budget is not None else falio.DEFAULT_BUDGET_USD),
                     "target_level": None, "group_by": args.group_by,
                     "aspect": args.aspect, "resolution": args.resolution},
        "iterations": {},
    }
    if companions:
        cfg["input"]["companions"] = companions
    if args.aspect:
        decide_size(cfg, None, "--aspect")
    common.save_case(root, cfg)

    for t in CASE_TEMPLATES:
        if (TEMPLATES_DIR / t).is_file() and not (root / t).exists():
            shutil.copyfile(TEMPLATES_DIR / t, root / t)

    script = "blockout.py" if kind == "spec" else "ingest.py"
    log = root / "logs" / f"iter0_{'blockout' if kind == 'spec' else 'ingest'}.log"
    extra = ["--case", str(root), "--iter", "0"]
    if args.samples:
        extra += ["--samples", str(args.samples)]
    print(f"[new] {kind} input copied to {dst.relative_to(root.parent)} (sha256 {cfg['input']['sha256'][:12]}...)")
    print(f"[new] building blockout/iter0 with {script} (log: {log.relative_to(root)}) ...", flush=True)
    rc = run_blender(script, extra, log)
    lines = log.read_text(errors="replace").splitlines()
    tag = "[blockout]" if kind == "spec" else "[ingest]"
    for line in lines:
        if line.startswith(tag) and ("NOTE" in line or "WARN" in line or "summary" in line):
            print("  " + line)
    if rc != 0:
        print(f"[new] {script} FAILED (exit {rc}); last lines of {log}:")
        print("\n".join(lines[-30:]))
        return rc
    cfg = common.load_case(root)
    if not cfg.get("size") or not cfg.get("camera"):
        print("[new] FAILED: the Blender stage did not record the size/camera in case.json")
        return 1
    s = cfg["size"]
    b0 = root / "blockout" / "iter0"
    objs = common.read_json(b0 / "objects.json")
    print(f"[new] case {root.name}: {len(objs)} objects, camera {cfg['camera']!r}, "
          f"size {s['width']}x{s['height']} ({s['aspect_ratio']}, {s['resolution']}, "
          f"size_verified={str(s['size_verified']).lower()})")
    print("Next:")
    print(f"  1. look at {b0 / 'preview.png'} and read {b0 / 'objects.json'}; edit each object's "
          "description (material, colour, finish) and category if needed")
    if not cfg["look"]:
        print(f"  2. fill case.json \"look\" (currently empty): what the photo should look like")
    else:
        print("  2. check case.json \"look\"")
    print(f"  3. write CONTRACT.md / acceptance.json in the case before the first target")
    print(f"  4. b2r.py gbuffers {root}    then    b2r.py run {root} --from target")
    return 0


if __name__ == "__main__":
    sys.exit(cmd_new(sys.argv[1:]))
