"""Decompose helper: run the fal estimators once per target and cache every raw output.

Cache layout (keyed on the target's content):
    decompose/iter<k>/raw/<sha12>/<name>.json   result JSON + {"files": {tag: file name}, "target_sha256"}
    decompose/iter<k>/raw/<sha12>/<name>_<tag>.<ext>
A flat legacy cache (raw/<name>.json, written before results were keyed) is accepted only when it
records the same target sha256, or when it records none and decompose.json next to it names the same
target sha256; anything else is refetched (or refused when fetching is disabled).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import common, falio


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs(W: int, H: int, fov_x: float) -> dict:
    mp = W * H / 1e6
    return {
        "marigold_albedo": (falio.MARIGOLD_V2, {"prediction_type": "albedo",
                            "image_size": {"width": W, "height": H}, "seed": 0}, 1.0),
        "marigold_normals": (falio.MARIGOLD_V2, {"prediction_type": "normals",
                             "image_size": {"width": W, "height": H}, "seed": 0}, 1.0),
        "patina": (falio.PATINA, {"maps": ["basecolor", "normal", "roughness", "metalness", "height"],
                   "seed": 0, "output_format": "png"}, mp),
        "moge3": (falio.MOGE_3, {"fov_x": float(fov_x), "num_refinement_steps": 2, "resolution_level": 9,
                  "apply_mask": False, "export_ply": True, "export_glb": True}, mp),
    }


def _save_result(d: Path, name: str, result: dict, target_sha: str) -> dict:
    """Download every URL in the result into d/<name>_<key>.<ext>; return {tag: absolute path}."""
    files = {}

    def walk(obj, key):
        if isinstance(obj, dict):
            if isinstance(obj.get("url"), str):
                url = obj["url"]
                tag = obj.get("map_type") or key or "file"
                ext = Path(url.split("?")[0]).suffix or ".bin"
                dest = d / f"{name}_{tag}{ext}"
                falio.download(url, dest)
                files[tag] = dest.name
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    walk(v, k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{key}{i}")

    walk(result, "")
    (d / f"{name}.json").write_text(json.dumps({"result": result, "files": files, "target_sha256": target_sha},
                                               indent=2))
    return {t: str(d / f) for t, f in files.items()}


def _load(cache: Path) -> dict:
    data = json.loads(cache.read_text())
    files = {t: str(cache.parent / Path(f).name) for t, f in data.get("files", {}).items()}
    missing = [f for f in files.values() if not Path(f).exists()]
    if missing:
        raise RuntimeError(f"cache {cache} lists files that are missing: {missing}")
    return {"result": data.get("result"), "files": files, "target_sha256": data.get("target_sha256")}


def legacy_sha(raw_root: Path) -> str | None:
    """target sha256 recorded by a decompose.json next to a flat legacy cache."""
    p = raw_root.parent / "decompose.json"
    if p.exists():
        try:
            return common.read_json(p).get("target_sha256")
        except Exception:  # noqa: BLE001
            return None
    return None


def fetch_all(case, target: Path, raw_root: Path, W: int, H: int, fov_x: float, refetch: bool = False,
              allow_fetch: bool = True) -> dict:
    """name -> {"files": {tag: path}, "cached": bool, "cache": how it was matched}.
    allow_fetch=False: read-only use of existing caches (raises if anything is missing)."""
    tsha = sha256(target)
    keyed = raw_root / tsha[:12]
    lsha = legacy_sha(raw_root)
    out, url = {}, None
    for name, (endpoint, args, units) in jobs(W, H, fov_x).items():
        hit = None
        if not refetch:
            kc = keyed / f"{name}.json"
            if kc.exists():
                d = _load(kc)
                if d["target_sha256"] not in (None, tsha):
                    raise RuntimeError(f"{kc} records target sha {str(d['target_sha256'])[:12]} != {tsha[:12]}")
                hit = {**d, "cached": True, "cache": f"keyed raw/{tsha[:12]}/{name}.json"}
            elif (raw_root / f"{name}.json").exists():
                d = _load(raw_root / f"{name}.json")
                if d["target_sha256"] == tsha:
                    hit = {**d, "cached": True, "cache": f"legacy raw/{name}.json (records the same target sha256)"}
                elif d["target_sha256"] is None and lsha == tsha:
                    hit = {**d, "cached": True, "cache": f"legacy raw/{name}.json (no sha recorded; decompose.json "
                                                         f"beside it names this target's sha256)"}
        if hit is not None:
            out[name] = hit
            continue
        if not allow_fetch:
            raise RuntimeError(f"no cached {name} for target sha {tsha[:12]} in {raw_root} and fetching is disabled "
                               f"(--out-dir reads the fal cache only)")
        keyed.mkdir(parents=True, exist_ok=True)
        if url is None:
            url = falio.upload(target)
        result = falio.call(case, endpoint, {"image_url": url, **args}, note=f"decompose {name}", units=units)
        files = _save_result(keyed, name, result, tsha)
        out[name] = {"result": result, "files": files, "cached": False, "cache": f"fresh raw/{tsha[:12]}/{name}.json"}
    return out
