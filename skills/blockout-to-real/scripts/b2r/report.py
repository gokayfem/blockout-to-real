"""Write CASE/REPORT.md from the case's own records: facts and numbers only.

    python -m b2r.report --case CASE

Reads final/summary.json (from `finalize`), final/judge.json (the blind judge, saved with
`critique --judge --save`) and residual/iter<k>/critique.json. Writes the LPIPS table per iteration,
the scene-by-target matrix and the final choice, the lights of the final scene with the virtual share,
what the final .blend contains, the critiques, the judge's answer and the fal spend.

The "Verdict" section is left for the agent to write in its own words: this tool never invents one.
If REPORT.md already has a written verdict (text between the VERDICT markers), it is kept.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import common

VERDICT_BEGIN = "<!-- VERDICT:BEGIN (written by the agent, not by report.py) -->"
VERDICT_END = "<!-- VERDICT:END -->"
VERDICT_TODO = ("_Not written yet. The agent writes its own verdict here after looking at the evidence sheets "
                "and the judge's answer: did the re-render keep the real feel, and what gives it away._")


def f4(x) -> str:
    return "n/a" if x is None else f"{float(x):.4f}"


def f3(x) -> str:
    return "n/a" if x is None else f"{float(x):.3f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def existing_verdict(path: Path) -> str | None:
    if not path.exists():
        return None
    m = re.search(re.escape(VERDICT_BEGIN) + r"\n(.*?)\n" + re.escape(VERDICT_END), path.read_text(), re.S)
    if m and m.group(1).strip() and m.group(1).strip() != VERDICT_TODO:
        return m.group(1).strip()
    return None


def build(case: Path) -> str:
    cfg = common.load_case(case)
    sp = case / "final" / "summary.json"
    if not sp.exists():
        raise SystemExit("final/summary.json is missing: run `b2r.py finalize CASE` first")
    S = common.read_json(sp)
    iters = [int(k) for k in S["iterations_scored"]]
    its = S["iterations"]
    fk = int(S["final_iteration"])
    L: list[str] = [f"# {case.name}: blockout to real", ""]
    inp = cfg.get("input", {})
    size = cfg.get("size", {})
    L += [f"Input `{inp.get('name')}` ({inp.get('kind')}), rendered at {size.get('width')} x {size.get('height')} "
          f"({size.get('aspect_ratio')}). Look: {cfg.get('look') or 'not set'}.", ""]

    L += ["## Verdict", "", VERDICT_BEGIN, existing_verdict(case / "REPORT.md") or VERDICT_TODO, VERDICT_END, ""]

    # ---- LPIPS per iteration
    L += ["## LPIPS per iteration", "",
          "LPIPS-AlexNet, lower is better. Each iteration is scored against its own target.", ""]
    rows = []
    for k in iters:
        p = its[str(k)]
        wd = p.get("rerender_vs_target_with_detail")
        rows.append([str(k), f3(p["blockout_vs_target"]["lpips_alex"]),
                     f3(p["rerender_vs_target_no_detail"]["lpips_alex"]),
                     f3(wd["lpips_alex"]) if wd else "n/a",
                     "kept" if p.get("detail_improved") else ("not kept" if wd else "not run"),
                     f3(p["headline_lpips_alex"])])
    L += table(["Iteration", "Blockout", "Re-render", "+ detail bump", "Detail", "Headline"], rows) + [""]

    # ---- matrix
    M = S["lpips_matrix_scene_vs_target"]
    mean = S.get("mean_lpips_alex_over_targets", {})
    L += ["## Every scene against every target", "",
          "Rows are the scene of an iteration, columns the target of an iteration (LPIPS-AlexNet).", ""]
    rows = [[f"scene {i}"] + [f3(M[f"scene_iter{i}"][f"target_iter{j}"]["lpips_alex"]) for j in iters]
            + [f3(mean.get(f"scene_iter{i}"))] for i in iters]
    L += table([""] + [f"target {j}" for j in iters] + ["mean"], rows) + [""]
    L += [f"The final scene is iteration {fk}. Rule: {S.get('final_choice_rule', '')}. "
          f"This run: {S.get('final_choice_reason', '')}.", ""]
    drift = S.get("target_drift") or {}
    if drift:
        L += ["Target drift between iterations (LPIPS-AlexNet): "
              + "; ".join(f"{k.replace('_vs_', ' vs ').replace('_iter', ' ')} {f3(v['lpips_alex'])}"
                          for k, v in drift.items()) + ".", ""]

    # ---- lights
    L += ["## Lights of the final scene", ""]
    rows = []
    for x in S.get("final_lights", []):
        rows.append([x["name"], str(x.get("type")), "yes" if x.get("virtual") else "no",
                     "n/a" if x.get("strength") is None else f"{float(x['strength']):.4g} {x.get('unit') or ''}".strip(),
                     "n/a" if x.get("energy_share") is None else f"{100 * float(x['energy_share']):.0f}%"])
    L += table(["Light", "Type", "Virtual", "Strength", "Share of image energy"], rows) + [""]
    L += [f"Virtual lights carry {100 * float(S.get('final_virtual_energy_share', 0)):.0f}% of the fitted image "
          "energy. A virtual light stands for a source the camera cannot see; it is hidden from camera and "
          "glossy rays and can be deleted.", ""]

    # ---- the .blend
    I = S.get("final_blend_inspect", {})
    flags = I.get("flags_not_ordinary") or []
    world = I.get("world") or {}
    L += ["## What the final .blend contains", "",
          f"`{S.get('final_blend')}` (copied from `{S.get('final_blend_source')}`): {I.get('mesh_objects')} mesh "
          f"objects with {I.get('total_faces')} faces, {I.get('images')} images of which {I.get('images_packed')} "
          f"are packed, {len(I.get('lights') or [])} light objects"
          + (f", a world background of strength {world.get('strength')}" if world else "")
          + f", view transform {I.get('view_transform')}. "
          + ("Nothing was flagged as non-ordinary." if not flags else "Flagged as non-ordinary: " + "; ".join(flags) + "."),
          ""]
    ev = S.get("evidence_renders") or {}
    if ev:
        L += ["Renders made from it (nothing composited):", ""]
        L += [f"- `final/{k}`: {v}" for k, v in ev.items()] + [""]

    # ---- critiques
    crit = [(k, case / "residual" / f"iter{k}" / "critique.json") for k in iters]
    crit = [(k, p) for k, p in crit if p.exists()]
    if crit:
        L += ["## Critiques (fresh reviewer, target vs re-render)", ""]
        for k, p in crit:
            c = common.read_json(p)
            L.append(f"Iteration {k}: {len(c['a_has_b_lacks'])} things the target has and the scene lacks, "
                     f"{len(c['b_has_a_lacks'])} the other way.")
            L.append("")
            for it in sorted(c["a_has_b_lacks"], key=lambda t: -t["severity"]):
                L.append(f"- severity {it['severity']}, {it['kind']}: {it['what']}")
            for it in sorted(c["b_has_a_lacks"], key=lambda t: -t["severity"]):
                L.append(f"- severity {it['severity']}, only in the scene: {it['what']}")
            L.append("")
    else:
        L += ["## Critiques", "", "No critique.json was saved for any iteration.", ""]

    # ---- judge
    jp = case / "final" / "judge.json"
    L += ["## Blind judge", ""]
    if jp.exists():
        J = common.read_json(jp)
        L += [f"A fresh reviewer saw only the grey blockout (reference_1), the target (reference_2) and the final "
              f"re-render (candidate). It placed the candidate closer to {J['closer_to']}, at "
              f"{J['score_0_to_10']}/10 (0 = reference_1, 10 = reference_2).", "",
              f"Photo or CG, in its words: {J['photo_or_cg']}", "",
              "What gives it away as CG:", ""]
        L += [f"- severity {g['severity']}: {g['what']}" for g in sorted(J["cg_giveaways"], key=lambda g: -g["severity"])]
        L += ["", "What passes as real: " + "; ".join(J["passes_as_real"]) + ".", "",
              f"The judge's one-sentence verdict: {J['verdict']}", ""]
    else:
        L += ["Not judged yet: run `b2r.py critique CASE --judge`, give the packed files to a fresh subagent and "
              "save its answer with `--save`.", ""]

    # ---- spend
    fal = S.get("fal_calls") or {}
    L += ["## fal spend", ""]
    if fal:
        L += [f"About ${S.get('fal_est_cost_usd', 0):.2f} estimated: "
              + ", ".join(f"{v['calls']} calls to {k} (${v['est_cost_usd']:.2f})" for k, v in fal.items())
              + ". Every call is logged in `fal_calls.jsonl`.", ""]
    else:
        L += ["No paid calls were logged.", ""]

    L += ["## Files", "",
          "- `final/contact_all_iterations.png`: blockout, target, re-render and error map per iteration",
          "- `final/contact_final_evidence.png`: target, re-render, novel view, relight",
          "- `residual/iter<k>/contact_sheet.png`: per-iteration sheets with the per-object table",
          "- `final/summary.json`: every number above", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--iter", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    case = common.case_dir(a.case)
    text = build(case)
    (case / "REPORT.md").write_text(text)
    print(f"[report] wrote {case / 'REPORT.md'} ({len(text.splitlines())} lines); the Verdict section is "
          + ("kept from the previous REPORT.md" if VERDICT_TODO not in text else "for the agent to write"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
