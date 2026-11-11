"""Collect the evidence for a finished case and package the final editable scene.

    python -m b2r.finalize --case CASE [--iters 0 1 2] [--final-iter K] [--samples 256]
                           [--scale 1.0] [--orbit-deg 30] [--dolly 0.9]

--iters defaults to every iteration that has residual/iter<k>/lpips.json.

Writes CASE/final/:
  summary.json                LPIPS per iteration (own target), the scene x target LPIPS matrix, target
                              drift, lights with the virtual share, detail results, the final choice and
                              why, what the final .blend contains, fal spend
  scene_final.blend           copy of the chosen iteration's lit scene (with its detail bump only if the
                              detail stage improved that iteration); inspect.json describes it
  render_final.png            that scene's re-render from the case camera (copied from the residual stage)
  novel_view.png              scene_final.blend from a camera orbited about the view pivot (no coordinates:
                              render_view.py --orbit-deg / --dolly); no target exists for this view
  relight_dusk.png            virtual lights x0.12, the strongest real light x2 tinted warm, world dimmed
                              and tinted; lights are chosen by kind from lights.json, not by name
  relight_no_virtual_fill.png every virtual light off (only when the fit has virtual lights)
  contact_all_iterations.png  one row per iteration: blockout | target | re-render | error map
  contact_final_evidence.png  target | re-render | novel view | relight

Every iteration is scored against ITS OWN target and the target is regenerated each iteration, so no single
column is a fair yardstick. The final iteration is the scene with the lowest LPIPS-Alex averaged over all
iterations' targets (ties -> the later iteration), unless --final-iter is given; summary.json says which.
Everything shown is rendered from the .blend; nothing from the target is composited into a render.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import common, falio

common.setup_model_env()

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

DUSK = {"virtual_scale": 0.12, "key_scale": 2.0, "key_tint": (1.0, 0.5, 0.25),
        "world_scale": 0.45, "world_tint": (1.0, 0.62, 0.45)}


def font(size: int):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rel(case: Path, p) -> str:
    try:
        return str(Path(p).resolve().relative_to(case))
    except ValueError:
        return str(p)


class LPIPS:
    def __init__(self):
        import warnings
        warnings.filterwarnings("ignore", message=".*(pretrained|weight enum).*")
        import lpips
        import torch
        self.torch = torch
        self.nets = {n: lpips.LPIPS(net=n, verbose=False).eval() for n in ("alex", "vgg")}

    def _t(self, path: Path):
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return self.torch.from_numpy(a).permute(2, 0, 1)[None]

    def __call__(self, a: Path, b: Path) -> dict:
        ta, tb = self._t(a), self._t(b)
        if ta.shape != tb.shape:
            raise SystemExit(f"{a} and {b} differ in size; refusing to resample")
        with self.torch.no_grad():
            return {f"lpips_{n}": round(float(m(ta, tb).item()), 4) for n, m in self.nets.items()}


def blender(case: Path, script: str, *args: str, log_name: str) -> None:
    exe = common.find_blender()
    if not exe:
        raise SystemExit("Blender not found: set $BLENDER")
    env = dict(os.environ)
    env.setdefault("B2R_PYTHON", sys.executable)
    env["PYTHONPATH"] = str(common.SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [exe, "-b", "--factory-startup", "--disable-autoexec", "--python", str(common.BLENDER_DIR / script), "--", *args]
    log = case / "logs" / f"finalize_{log_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
        raise SystemExit(f"{script} failed (exit {rc}); log {log}:\n{tail}")


def row_sheet(rows, title: str, out: Path, cell_w: int = 576) -> None:
    """rows of (label, image path); every cell is scaled to cell_w wide (keeps its aspect)."""
    f_title, f_lab = font(26), font(17)
    pad, lab_h, title_h = 10, 26, 44
    cells = []
    for r in rows:
        ims = []
        for label, p in r:
            im = Image.open(p).convert("RGB")
            ims.append((label, im.resize((cell_w, round(im.height * cell_w / im.width)), Image.LANCZOS)))
        cells.append(ims)
    ncol = max(len(r) for r in cells)
    row_h = [max(im.height for _, im in r) + lab_h for r in cells]
    sheet = Image.new("RGB", (pad + ncol * (cell_w + pad), title_h + sum(row_h) + pad * (len(cells) + 1)),
                      (250, 250, 248))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), title, fill=(20, 20, 20), font=f_title)
    y = title_h
    for r, rh in zip(cells, row_h):
        x = pad
        for label, im in r:
            d.text((x, y), label, fill=(30, 30, 30), font=f_lab)
            sheet.paste(im, (x, y + lab_h))
            x += cell_w + pad
        y += rh + pad
    sheet.save(out)


# ---------------------------------------------------------------- lights by kind
def _is_world(L: dict) -> bool:
    return "WORLD" in str(L.get("type", "")).upper() or L.get("kind") == "world"


def _colour(L: dict):
    c = L.get("colour_rgb_linear") or L.get("color") or L.get("colour") or [1.0, 1.0, 1.0]
    return [float(x) for x in c[:3]]


def _share(L: dict) -> float:
    v = L.get("energy_share_of_image", L.get("energy_share"))
    return float(v) if v is not None else 0.0


def emitting(lights: list[dict]) -> list[dict]:
    """Lights that carry energy (portals and other helpers have no strength)."""
    return [L for L in lights if L.get("strength") is not None]


def relight_plan(lights: list[dict]) -> dict:
    """render_view.py arguments for the dusk relight and the no-virtual render, chosen by kind."""
    em = emitting(lights)
    virtual = [L["name"] for L in em if L.get("virtual") and not _is_world(L)]
    real = [L for L in em if not L.get("virtual") and not _is_world(L)]
    has_world = any(_is_world(L) for L in em)
    key = max(real, key=_share) if real else None
    scale, colour, notes = [], [], []
    for n in virtual:
        scale.append(f"{n}={DUSK['virtual_scale']}")
    if virtual:
        notes.append(f"virtual lights x{DUSK['virtual_scale']} ({', '.join(virtual)})")
    if key is not None:
        c = [round(a * b, 4) for a, b in zip(_colour(key), DUSK["key_tint"])]
        scale.append(f"{key['name']}={DUSK['key_scale']}")
        colour.append(f"{key['name']}={c[0]},{c[1]},{c[2]}")
        notes.append(f"strongest real light '{key['name']}' (share {_share(key):.2f}) x{DUSK['key_scale']}, tinted warm")
    if has_world:
        scale.append(f"world={DUSK['world_scale']}")
        w = DUSK["world_tint"]
        colour.append(f"world={w[0]},{w[1]},{w[2]}")
        notes.append(f"world x{DUSK['world_scale']}, tinted warm")
    return {"dusk_scale": scale, "dusk_colour": colour, "dusk_note": "; ".join(notes) or "no lights to change",
            "virtual": virtual, "key_light": key["name"] if key else None}


# ---------------------------------------------------------------- main
def available_iters(case: Path) -> list[int]:
    return [k for k in range(common.MAX_ITERATIONS) if (case / "residual" / f"iter{k}" / "lpips.json").exists()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--iters", type=int, nargs="+", default=None)
    ap.add_argument("--final-iter", type=int, default=None)
    ap.add_argument("--samples", type=int, default=256, help="samples of the novel view and relight renders")
    ap.add_argument("--scale", type=float, default=1.0, help="size factor of the novel view and relight renders "
                    "(1 = case size; use e.g. 0.5 with --samples 48 for a quick check)")
    ap.add_argument("--orbit-deg", type=float, default=30.0, help="novel view: camera yaw about the view pivot")
    ap.add_argument("--dolly", type=float, default=0.9, help="novel view: camera-pivot distance factor")
    ap.add_argument("--iter", type=int, default=None, help=argparse.SUPPRESS)  # passed by `b2r.py run`; unused
    a = ap.parse_args(argv)

    case = common.case_dir(a.case)
    cfg = common.load_case(case)
    iters = sorted(a.iters) if a.iters else available_iters(case)
    if not iters:
        print("[finalize] no iteration has residual/iter<k>/lpips.json", file=sys.stderr)
        return 2
    if a.final_iter is not None and a.final_iter not in iters:
        print(f"[finalize] --final-iter {a.final_iter} is not among --iters {iters}", file=sys.stderr)
        return 2
    final = case / "final"
    final.mkdir(parents=True, exist_ok=True)
    lp = LPIPS()

    per_iter, scene_img, target_img, blend_path, lights_by_iter = {}, {}, {}, {}, {}
    for k in iters:
        res = common.stage_dir(case, "residual", k, create=False)
        bp = common.stage_dir(case, "backproject", k, create=False)
        tg = common.stage_dir(case, "target", k, create=False)
        base = common.read_json(res / "lpips.json")
        det = common.read_json(res / "detail_result.json") if (res / "detail_result.json").exists() else None
        use_detail = bool(det and det.get("improved"))
        scene_img[k] = res / ("rerender_detail.png" if use_detail else "rerender.png")
        blend_path[k] = bp / ("scene_lit_detail.blend" if use_detail else "scene_lit.blend")
        target_img[k] = tg / "target.png"
        for p in (scene_img[k], blend_path[k], target_img[k], bp / "lights.json"):
            if not p.exists():
                print(f"[finalize] iteration {k}: missing {p}", file=sys.stderr)
                return 2
        lj = common.read_json(bp / "lights.json")
        lights = lj["lights"] if isinstance(lj, dict) else lj
        lights_by_iter[k] = lights
        choice = common.read_json(tg / "choice.json") if (tg / "choice.json").exists() else {}
        sel_p = bp / "relight_selection.json"
        em = emitting(lights)
        per_iter[k] = {
            "target_level": choice.get("chosen_level", choice.get("level")),
            "blockout_vs_target": base["blockout_vs_target"],
            "rerender_vs_target_no_detail": base["rerender_vs_target"],
            "rerender_vs_target_with_detail": det.get("selected") if det else None,
            "detail_improved": use_detail,
            "detail_kept_objects": det.get("kept") if det else None,
            "detail_dropped_objects": det.get("dropped") if det else None,
            "headline_lpips_alex": round(float((det["selected"] if use_detail else base["rerender_vs_target"])
                                               ["lpips_alex"]), 4),
            "relight_selection": common.read_json(sel_p) if sel_p.exists() else None,
            "lights": [{"name": L["name"], "type": L.get("type"), "virtual": bool(L.get("virtual")),
                        "strength": L.get("strength"), "unit": L.get("strength_unit", L.get("unit")),
                        "energy_share": _share(L) if L.get("strength") is not None else None} for L in lights],
            "virtual_energy_share": round(sum(_share(L) for L in em if L.get("virtual")), 4),
            "scene_image": rel(case, scene_img[k]),
            "scene_blend": rel(case, blend_path[k]),
            "critique": rel(case, res / "critique.json") if (res / "critique.json").exists() else None,
        }

    print("[finalize] LPIPS matrix (every scene against every target) ...", flush=True)
    matrix = {f"scene_iter{i}": {f"target_iter{j}": lp(scene_img[i], target_img[j]) for j in iters} for i in iters}
    drift = {f"target_iter{x}_vs_target_iter{y}": lp(target_img[x], target_img[y])
             for x in iters for y in iters if x < y}
    mean_over_targets = {i: round(float(np.mean([matrix[f"scene_iter{i}"][f"target_iter{j}"]["lpips_alex"]
                                                  for j in iters])), 4) for i in iters}
    rule = ("lowest LPIPS-Alex averaged over every iteration's target (targets are regenerated each iteration "
            "and drift, so no single column is a fair yardstick); ties go to the later iteration")
    if a.final_iter is not None:
        fk, why = a.final_iter, f"set by --final-iter (the rule would pick "\
                                f"{min(iters, key=lambda i: (mean_over_targets[i], -i))})"
    else:
        fk = min(iters, key=lambda i: (mean_over_targets[i], -i))
        why = "chosen by the rule"

    shutil.copyfile(blend_path[fk], final / "scene_final.blend")
    shutil.copyfile(scene_img[fk], final / "render_final.png")
    print(f"[finalize] final iteration {fk} ({why}); inspecting and rendering scene_final.blend ...", flush=True)
    blender(case, "inspect_blend.py", "--blend", str(final / "scene_final.blend"),
            "--out", str(final / "inspect.json"), log_name="inspect")
    insp = common.read_json(final / "inspect.json")

    plan = relight_plan(lights_by_iter[fk])
    s = ["--blend", str(final / "scene_final.blend"), "--samples", str(a.samples), "--scale", str(a.scale)]
    blender(case, "render_view.py", *s, "--out", str(final / "novel_view.png"),
            "--orbit-deg", str(a.orbit_deg), "--dolly", str(a.dolly), log_name="novel_view")
    dusk = [*s, "--out", str(final / "relight_dusk.png")]
    if plan["dusk_scale"]:
        dusk += ["--light-scale", *plan["dusk_scale"]]
    if plan["dusk_colour"]:
        dusk += ["--light-color", *plan["dusk_colour"]]
    blender(case, "render_view.py", *dusk, log_name="relight_dusk")
    renders = {"novel_view.png": f"camera orbited {a.orbit_deg:g} deg about the view pivot, dolly x{a.dolly:g}",
               "relight_dusk.png": plan["dusk_note"]}
    nv = final / "relight_no_virtual_fill.png"
    renders["settings"] = f"{a.samples} spp, scale {a.scale:g} of the case size"
    if plan["virtual"]:
        blender(case, "render_view.py", *s, "--out", str(nv),
                "--light-scale", *[f"{n}=0.0" for n in plan["virtual"]], log_name="relight_no_virtual")
        renders["relight_no_virtual_fill.png"] = f"virtual lights off ({', '.join(plan['virtual'])})"
    elif nv.exists():
        nv.unlink()

    rows = []
    for k in iters:
        res = common.stage_dir(case, "residual", k, create=False)
        p = per_iter[k]
        err = res / ("error_map_detail.png" if p["detail_improved"] and (res / "error_map_detail.png").exists()
                     else "error_map.png")
        lvl = f" (level {p['target_level']})" if p["target_level"] else ""
        rows.append([
            (f"iter {k}: blockout (LPIPS to target {p['blockout_vs_target']['lpips_alex']:.3f})",
             common.stage_dir(case, "gbuffers", k, create=False) / "beauty.png"),
            (f"iter {k}: target{lvl}", target_img[k]),
            (f"iter {k}: Cycles re-render  LPIPS {p['headline_lpips_alex']:.3f}"
             f" (no detail {p['rerender_vs_target_no_detail']['lpips_alex']:.3f})", scene_img[k]),
            (f"iter {k}: error (spatial LPIPS, fixed scale)", err),
        ])
    row_sheet(rows, f"{case.name}: blockout | target | re-render | error, per iteration "
                    f"(LPIPS-Alex, each vs its own target)", final / "contact_all_iterations.png")
    ev = [[("target (final iteration)", target_img[fk]), ("Cycles re-render of scene_final.blend", final / "render_final.png")],
          [("same .blend, camera moved (no target exists for this view)", final / "novel_view.png"),
           ("same .blend, relit at dusk: virtual x0.12, key light warm x2, world dimmed", final / "relight_dusk.png")]]
    row_sheet(ev, f"{case.name}: final scene (iteration {fk}): re-render, novel view, relight",
              final / "contact_final_evidence.png", cell_w=760)

    fal = falio.summary(case)
    keys = ("mesh_objects", "total_faces", "images", "images_packed", "images_unpacked", "lights", "world",
            "view_transform", "flags_not_ordinary")
    summary = {
        "case": case.name,
        "look": cfg.get("look"),
        "iterations_scored": iters,
        "iterations": {str(k): v for k, v in per_iter.items()},
        "lpips_matrix_scene_vs_target": matrix,
        "mean_lpips_alex_over_targets": {f"scene_iter{i}": v for i, v in mean_over_targets.items()},
        "target_drift": drift,
        "final_iteration": fk,
        "final_choice_rule": rule,
        "final_choice_reason": why,
        "final_blend": "final/scene_final.blend",
        "final_blend_source": rel(case, blend_path[fk]),
        "final_blend_inspect": {k: insp.get(k) for k in keys},
        "final_lights": per_iter[fk]["lights"],
        "final_virtual_energy_share": per_iter[fk]["virtual_energy_share"],
        "evidence_renders": renders,
        "judge_inputs": {"reference_1.png": "grey blockout", "reference_2.png": "target",
                         "candidate.png": "final re-render"},
        "fal_calls": fal,
        "fal_est_cost_usd": round(sum(v["est_cost_usd"] for v in fal.values()), 2),
    }
    common.write_json(final / "summary.json", summary)

    print(f"[finalize] final iteration {fk}: {why}; mean over targets {mean_over_targets}")
    for k in iters:
        p = per_iter[k]
        wd = p["rerender_vs_target_with_detail"]
        print(f"  iter {k}: blockout {p['blockout_vs_target']['lpips_alex']:.4f} | re-render "
              f"{p['rerender_vs_target_no_detail']['lpips_alex']:.4f} | with detail "
              f"{wd['lpips_alex'] if wd else float('nan'):.4f}")
    print("  matrix (LPIPS-Alex), rows = scene, cols = target:")
    for i in iters:
        print(f"    scene {i}: " + "  ".join(f"{matrix[f'scene_iter{i}'][f'target_iter{j}']['lpips_alex']:.4f}"
                                          for j in iters))
    print(f"  fal est. USD {summary['fal_est_cost_usd']}; wrote {final}")
    print("[finalize] next: `b2r.py critique CASE --judge` and give its files to a FRESH subagent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
