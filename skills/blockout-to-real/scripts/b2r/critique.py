"""Pack images for a FRESH reviewer, print its prompt, and validate what it answers.

    python -m b2r.critique --case CASE [--iter K]                  pack residual/iterK/critique_in/ + print prompt
    python -m b2r.critique --case CASE [--iter K] --save FILE      validate the reviewer's JSON -> residual/iterK/critique.json
    python -m b2r.critique --case CASE --judge                     pack final/judge_in/ + print the blind-judge prompt
    python -m b2r.critique --case CASE --judge --save FILE         validate the judge's JSON -> final/judge.json

This tool does no reviewing. The calling agent starts a NEW subagent that has seen nothing of the case,
gives it only the packed files and the printed prompt, and saves its answer with --save.
Critique: A = the iteration's target, B = its best re-render. Judge (blind): reference_1 = grey blockout,
reference_2 = target, candidate = final re-render; the names do not say which is which.

The re-render packed for iteration K is the iteration's best: rerender_detail.png when the detail stage
improved LPIPS (residual/iterK/detail_result.json "improved"), else rerender.png.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import common

CRITIQUE_KINDS = ("missing_object", "shape", "exterior", "material", "lighting")

CRITIQUE_PROMPT = """You are shown two images of the same view, {w} x {h} pixels each:
  A_reference_photo.png  (A)
  B_scene_render.png     (B)
Look only at these two files. Compare B against A.

1. List what A has that B lacks. For each item give:
     "what":     one or two sentences describing the difference,
     "kind":     one of "missing_object", "shape", "exterior", "material", "lighting"
                 (missing_object = something present in A and absent in B; shape = an object is there
                 but its form or size differs; exterior = what is seen through openings or outside;
                 material = colour, texture, finish, gloss; lighting = light direction, softness,
                 brightness, shadows),
     "bbox":     [x0, y0, x1, y1] in pixels of A (x right, y down, origin top-left),
     "severity": integer 1-5 (5 = the most visible difference),
     "fix_hint": one sentence on what would change in the 3D scene to fix it.
2. List what B has that A lacks (artifacts, extra objects, wrong light patches), each with
   "what", "bbox" and "severity".
3. One short paragraph "overall".

Order each list by severity, highest first. Answer with JSON only, exactly this shape:
{{"a_has_b_lacks": [{{"what": "...", "kind": "...", "bbox": [x0, y0, x1, y1], "severity": 1, "fix_hint": "..."}}],
 "b_has_a_lacks": [{{"what": "...", "bbox": [x0, y0, x1, y1], "severity": 1}}],
 "overall": "..."}}"""

JUDGE_PROMPT = """You are shown three images of the same view, {w} x {h} pixels each:
  reference_1.png, reference_2.png, candidate.png
Look only at these three files.

1. Which reference is the candidate closer to? Give "closer_to" ("reference_1" or "reference_2") and
   "score_0_to_10": 0 = the candidate looks exactly like reference_1, 10 = exactly like reference_2.
   Explain in "reasons".
2. Does the candidate read as a photograph or as computer graphics? Answer in "photo_or_cg".
3. What gives the candidate away as CG? List "cg_giveaways", each with "what", "bbox"
   [x0, y0, x1, y1] in pixels (x right, y down, origin top-left) and "severity" 1-5 (5 = most obvious).
4. What in the candidate passes as real? List short phrases in "passes_as_real".
5. One-sentence "verdict": does the candidate keep the real feel of the closer reference?

Answer with JSON only, exactly this shape:
{{"closer_to": "reference_2", "score_0_to_10": 0.0, "reasons": "...", "photo_or_cg": "...",
 "cg_giveaways": [{{"what": "...", "bbox": [x0, y0, x1, y1], "severity": 1}}],
 "passes_as_real": ["..."], "verdict": "..."}}"""


# ---------------------------------------------------------------- validation (pure)
def _check_bbox(b, size, where, errs):
    if not (isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                         for x in b)):
        errs.append(f"{where}.bbox must be [x0, y0, x1, y1] numbers, got {b!r}")
        return
    x0, y0, x1, y1 = b
    if not (x1 > x0 and y1 > y0):
        errs.append(f"{where}.bbox needs x1 > x0 and y1 > y0, got {b}")
    if size is not None:
        w, h = size
        if min(x0, y0) < 0 or x1 > w or y1 > h:
            errs.append(f"{where}.bbox {b} lies outside the {w}x{h} image")


def _check_sev(s, where, errs):
    if not (isinstance(s, int) and not isinstance(s, bool) and 1 <= s <= 5):
        errs.append(f"{where}.severity must be an integer 1-5, got {s!r}")


def _check_text(d, key, where, errs):
    if not (isinstance(d.get(key), str) and d[key].strip()):
        errs.append(f"{where}.{key} must be a non-empty string")


def validate_critique(obj, size=None) -> list[str]:
    """Errors in a reviewer's critique JSON (empty list = valid). size = (width, height) for bbox checks."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["critique must be a JSON object"]
    extra = set(obj) - {"a_has_b_lacks", "b_has_a_lacks", "overall"}
    if extra:
        errs.append(f"unexpected keys {sorted(extra)}")
    for key in ("a_has_b_lacks", "b_has_a_lacks"):
        if not isinstance(obj.get(key), list):
            errs.append(f"{key} must be a list")
            continue
        for i, it in enumerate(obj[key]):
            where = f"{key}[{i}]"
            if not isinstance(it, dict):
                errs.append(f"{where} must be an object")
                continue
            _check_text(it, "what", where, errs)
            _check_bbox(it.get("bbox"), size, where, errs)
            _check_sev(it.get("severity"), where, errs)
            if key == "a_has_b_lacks":
                if it.get("kind") not in CRITIQUE_KINDS:
                    errs.append(f"{where}.kind must be one of {CRITIQUE_KINDS}, got {it.get('kind')!r}")
                _check_text(it, "fix_hint", where, errs)
    _check_text(obj, "overall", "critique", errs)
    return errs


def validate_judge(obj, size=None) -> list[str]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["judge answer must be a JSON object"]
    need = {"closer_to", "score_0_to_10", "reasons", "photo_or_cg", "cg_giveaways", "passes_as_real", "verdict"}
    missing = need - set(obj)
    if missing:
        errs.append(f"missing keys {sorted(missing)}")
    if obj.get("closer_to") not in ("reference_1", "reference_2"):
        errs.append(f"closer_to must be 'reference_1' or 'reference_2', got {obj.get('closer_to')!r}")
    s = obj.get("score_0_to_10")
    if not (isinstance(s, (int, float)) and not isinstance(s, bool) and 0 <= s <= 10):
        errs.append(f"score_0_to_10 must be a number in 0..10, got {s!r}")
    for key in ("reasons", "photo_or_cg", "verdict"):
        _check_text(obj, key, "judge", errs)
    g = obj.get("cg_giveaways")
    if not isinstance(g, list):
        errs.append("cg_giveaways must be a list")
    else:
        for i, it in enumerate(g):
            where = f"cg_giveaways[{i}]"
            if not isinstance(it, dict):
                errs.append(f"{where} must be an object")
                continue
            _check_text(it, "what", where, errs)
            _check_bbox(it.get("bbox"), size, where, errs)
            _check_sev(it.get("severity"), where, errs)
    p = obj.get("passes_as_real")
    if not (isinstance(p, list) and all(isinstance(x, str) and x.strip() for x in p)):
        errs.append("passes_as_real must be a list of non-empty strings")
    return errs


# ---------------------------------------------------------------- packing
def best_rerender(case: Path, k: int) -> Path:
    res = common.stage_dir(case, "residual", k, create=False)
    det = res / "detail_result.json"
    if det.exists() and common.read_json(det).get("improved") and (res / "rerender_detail.png").exists():
        return res / "rerender_detail.png"
    return res / "rerender.png"


def latest_iter(case: Path) -> int:
    ks = [k for k in range(common.MAX_ITERATIONS) if (case / "residual" / f"iter{k}" / "rerender.png").exists()]
    if not ks:
        raise SystemExit(f"no residual/iter*/rerender.png in {case}")
    return max(ks)


def _image_size(p: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SystemExit(f"missing input: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def pack_critique(case: Path, k: int) -> tuple[Path, str]:
    out = common.stage_dir(case, "residual", k) / "critique_in"
    if out.exists():
        shutil.rmtree(out)
    _copy(common.stage_dir(case, "target", k, create=False) / "target.png", out / "A_reference_photo.png")
    _copy(best_rerender(case, k), out / "B_scene_render.png")
    w, h = _image_size(out / "A_reference_photo.png")
    return out, CRITIQUE_PROMPT.format(w=w, h=h)


def final_iter(case: Path) -> int:
    p = case / "final" / "summary.json"
    if not p.exists():
        raise SystemExit("final/summary.json is missing: run `b2r.py finalize CASE` before the judge")
    return int(common.read_json(p)["final_iteration"])


def pack_judge(case: Path) -> tuple[Path, str]:
    fk = final_iter(case)
    out = case / "final" / "judge_in"
    if out.exists():
        shutil.rmtree(out)
    _copy(common.stage_dir(case, "gbuffers", fk, create=False) / "beauty.png", out / "reference_1.png")
    _copy(common.stage_dir(case, "target", fk, create=False) / "target.png", out / "reference_2.png")
    _copy(case / "final" / "render_final.png", out / "candidate.png")
    w, h = _image_size(out / "candidate.png")
    return out, JUDGE_PROMPT.format(w=w, h=h)


def _load_answer(path: str):
    text = Path(path).read_text().strip()
    if text.startswith("```"):  # tolerate a fenced answer
        text = text.strip("`")
        text = text[text.index("{"):]
    return json.loads(text)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, default=None, help="default: the latest iteration with a re-render")
    ap.add_argument("--judge", action="store_true", help="the blind final judge instead of the per-iteration critique")
    ap.add_argument("--save", default=None, help="validate this JSON answer and store it")
    a = ap.parse_args(argv)
    case = common.case_dir(a.case)
    common.load_case(case)

    if a.judge:
        if a.save:
            size = _image_size(case / "final" / "judge_in" / "candidate.png")
            obj = _load_answer(a.save)
            errs = validate_judge(obj, size)
            if errs:
                print("[critique] judge answer rejected:\n  " + "\n  ".join(errs), file=sys.stderr)
                return 1
            common.write_json(case / "final" / "judge.json", obj)
            print(f"[critique] saved {case / 'final' / 'judge.json'}")
            return 0
        out, prompt = pack_judge(case)
        print(f"[critique] packed {out}: {', '.join(sorted(p.name for p in out.iterdir()))}")
        print("[critique] start a FRESH subagent that may see only these files; give it this prompt:\n")
        print(prompt)
        return 0

    k = latest_iter(case) if a.iter is None else a.iter
    if a.save:
        inp = common.stage_dir(case, "residual", k, create=False) / "critique_in" / "A_reference_photo.png"
        size = _image_size(inp) if inp.exists() else common.render_size(case)
        obj = _load_answer(a.save)
        errs = validate_critique(obj, size)
        if errs:
            print("[critique] critique rejected:\n  " + "\n  ".join(errs), file=sys.stderr)
            return 1
        dst = common.stage_dir(case, "residual", k) / "critique.json"
        common.write_json(dst, obj)
        print(f"[critique] saved {dst} ({len(obj['a_has_b_lacks'])} + {len(obj['b_has_a_lacks'])} items)")
        return 0
    out, prompt = pack_critique(case, k)
    print(f"[critique] iter {k}: packed {out}: A_reference_photo.png (target), B_scene_render.png "
          f"({best_rerender(case, k).name})")
    print("[critique] start a FRESH subagent that may see only these two files; give it this prompt:\n")
    print(prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
