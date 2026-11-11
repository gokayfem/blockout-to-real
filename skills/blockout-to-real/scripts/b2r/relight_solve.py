"""Light-fit solver for the relight stage (user's Python: numpy, scipy, opencv).

Importable:  solve_lights(bases, target, kinds, ...) -> dict      (pure numpy, no files)
CLI:         python -m b2r.relight_solve --work DIR [--prune-margin 0.01] [--prune-total-cap 0.03]

DIR must contain basis/manifest.json written by scripts/blender/relight.py:
    {"scale", "target", "id", "objects": {id: {"name", "category"}},
     "candidates": [{"name", "kind" (world|area|sun|point|spot), "virtual", "sun_group", "exr"}]}
and gets basis/solution.json, fit.json and fit_preview.png.

Model:  target_lin[c] ~= sum_i w_ic * basis_i[c]   per channel c, w_ic >= 0
        (Cycles is linear in light intensity; bases are white unit-strength lights.)
Pixels: a basis-resolution pixel is used if all its full-resolution ids are equal and either
        > 0 with every full-resolution target sRGB channel < SAT_SRGB, or == 0 (open sky: the
        world basis carries its own radiance there; never dropped, target clipped at 1.0).
Loss:   IRLS on the relative residual (pred - t) / (t + EPS) with a Huber weight, solved with
        scipy.optimize.nnls on a fixed random pixel subset.
Steps:  1) drop black bases; 2) solve with every candidate; 3) keep the brightest sun of each
        sun group, drop the other suns, re-solve; 3b) parsimony pruning (greedy, world exempt);
        4) clamp tints; 5) refit one scalar per light with the tint locked; world floor if no sky.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from b2r import common

import warnings

# numpy 2.x + Accelerate on Apple silicon raises spurious FP warnings in matmul
warnings.filterwarnings("ignore", message=".*encountered in matmul")
EPS = 0.02            # linear; floor of the relative-error denominator
HUBER = 0.15          # relative-error Huber threshold
IRLS_ITERS = 8
SUBSET = 60000
SAT_SRGB = 250 / 255.0
TINT_MAX_RATIO = 2.5
WORLD_MIN_STRENGTH = 0.1  # basis units; applied only when no open-sky pixel is visible
BLACK_LUM = 1e-3      # a basis pixel counts as lit above this linear luminance
BLACK_FRAC = 0.002    # a basis is "black" if fewer than this fraction of pixels are lit
LUMA = np.array([0.2126, 0.7152, 0.0722])


# --------------------------------------------------------------------------- numerics
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def solve_channel(A, t, iters=IRLS_ITERS):
    """Non-negative IRLS for the relative Huber loss. A: N x K, t: N. Returns w (K,)."""
    K = A.shape[1]
    if K == 0:
        return np.zeros(0)
    base = 1.0 / (t + EPS)
    hub = np.ones_like(t)
    norms = np.linalg.norm(A * base[:, None], axis=0)
    norms[norms == 0] = 1.0
    w = np.zeros(K)
    for _ in range(iters):
        rw = np.sqrt(hub) * base
        As = A * rw[:, None] / norms[None]
        x, _ = nnls(As, t * rw, maxiter=50 * K)
        w = x / norms
        a = np.abs((A @ w - t) * base)
        hub = np.where(a <= HUBER, 1.0, HUBER / np.maximum(a, 1e-12))
    return w


def _solve(B, T, active, idx):
    """B: K x N x 3, T: N x 3, active: basis indices. Returns W: K x 3."""
    W = np.zeros((B.shape[0], 3))
    for c in range(3):
        if active:
            W[active, c] = solve_channel(B[active][:, idx, c].T, T[idx, c])
    return W


def fit_metrics(P, T):
    """Relative RMSE, NRMSE, R^2 per channel and overall. P, T: N x 3 linear."""
    out = {}
    rel = (P - T) / (T + EPS)
    for k, c in (("R", 0), ("G", 1), ("B", 2)):
        ss_res = np.sum((P[:, c] - T[:, c]) ** 2)
        ss_tot = np.sum((T[:, c] - T[:, c].mean()) ** 2)
        out[k] = {"rel_rmse": float(np.sqrt(np.mean(rel[:, c] ** 2))),
                  "nrmse": float(np.sqrt(ss_res / max(np.sum(T[:, c] ** 2), 1e-30))),
                  "r2": float(1 - ss_res / max(ss_tot, 1e-30)),
                  "mean_ratio_fit_over_target": float(P[:, c].mean() / max(T[:, c].mean(), 1e-30))}
    ss_res = np.sum((P - T) ** 2)
    ss_tot = np.sum((T - T.mean(0)) ** 2)
    lum_tot = np.sum((T @ LUMA - (T @ LUMA).mean()) ** 2)
    out["overall"] = {"rel_rmse": float(np.sqrt(np.mean(rel ** 2))),
                      "median_abs_rel_err": float(np.median(np.abs(rel))),
                      "nrmse": float(np.sqrt(ss_res / max(np.sum(T ** 2), 1e-30))),
                      "r2": float(1 - ss_res / max(ss_tot, 1e-30)),
                      "r2_luminance": float(1 - np.sum(((P - T) @ LUMA) ** 2) / max(lum_tot, 1e-30))}
    return out


def _energies(E, W):
    e = np.array([float(np.sum((E[i] * W[i]) @ LUMA)) for i in range(E.shape[0])])
    tot = e.sum()
    return e, (e / tot if tot > 0 else e)


def solve_lights(bases, target, kinds, groups=None, is_sky=None, energy_bases=None,
                 prune_margin=0.01, prune_total_cap=0.03, subset=SUBSET, seed=0, names=None) -> dict:
    """Fit non-negative RGB weights of K unit-light basis images to a linear target.

    bases         K x N x 3 linear basis pixels (the pixels used in the fit)
    target        N x 3 linear target pixels
    kinds         K strings: "world" (never pruned, at most one), "sun", or anything else
    groups        K sun-group labels (None entries for non-suns); the brightest sun of each group
                  survives step 3, all other suns are dropped
    is_sky        N bools, pixels where the camera sees open sky (default none)
    energy_bases  K x M x 3 pixels over which energy shares and the black test are computed
                  (default: bases at the non-sky pixels)
    Returns a dict with "weights" (K x 3 final), the per-stage weights, "metrics" per stage
    (fit_metrics on the fit pixels), "energy_share" (K, final), "survivors", "pruned",
    "black", "best_sun", "tint_clamped", "world_floor".
    """
    B = np.asarray(bases, np.float64)
    T = np.asarray(target, np.float64)
    K, N = B.shape[0], B.shape[1]
    if T.shape != (N, 3):
        raise ValueError(f"target {T.shape} does not match bases {B.shape}")
    names = list(names) if names is not None else [f"light{i}" for i in range(K)]
    groups = list(groups) if groups is not None else [None] * K
    is_sky = np.zeros(N, bool) if is_sky is None else np.asarray(is_sky, bool)
    E = B[:, ~is_sky] if energy_bases is None else np.asarray(energy_bases, np.float64)
    n_sky = int(is_sky.sum())
    worlds = [i for i in range(K) if kinds[i] == "world"]
    if len(worlds) > 1:
        raise ValueError("at most one world candidate")
    world_i = worlds[0] if worlds else None
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N, size=min(subset, N), replace=False))

    # 1 black bases
    lum = E @ LUMA
    lit_frac = (lum > BLACK_LUM).mean(axis=1) if lum.shape[1] else np.zeros(K)
    black = [i for i in range(K) if lit_frac[i] < BLACK_FRAC]
    active = [i for i in range(K) if i not in black]

    # 2 every candidate
    W1 = _solve(B, T, active, idx)
    m1 = fit_metrics(np.einsum("knc,kc->nc", B, W1), T)
    e1, sh1 = _energies(E, W1)

    # 3 one sun per group
    best_sun = {}
    for g in sorted({str(groups[i] or "default") for i in active if kinds[i] == "sun"}):
        members = [i for i in active if kinds[i] == "sun" and str(groups[i] or "default") == g]
        b = max(members, key=lambda i: e1[i])
        best_sun[g] = b if e1[b] > 0 else None
    keep_suns = {b for b in best_sun.values() if b is not None}
    active2 = [i for i in active if kinds[i] != "sun" or i in keep_suns]
    W2 = _solve(B, T, active2, idx)
    m2 = fit_metrics(np.einsum("knc,kc->nc", B, W2), T)

    # 3b parsimony pruning (greedy, least useful light first; the world is exempt)
    def rel_rmse_of(act):
        Wt = _solve(B, T, act, idx)
        return fit_metrics(np.einsum("knc,kc->nc", B, Wt), T)["overall"]["rel_rmse"]

    cur = [i for i in active2 if W2[i].max() > 0]
    if world_i is not None and world_i in active2 and world_i not in cur:
        cur.append(world_i)
    r_unpruned = m2["overall"]["rel_rmse"]
    r_cur = r_unpruned
    rounds, pruned = [], []
    while len([i for i in cur if i != world_i]) > 1:
        prunable = [i for i in cur if i != world_i]
        trial = {i: rel_rmse_of([j for j in cur if j != i]) for i in prunable}
        i_min = min(prunable, key=lambda i: trial[i])
        step = (trial[i_min] - r_cur) / r_cur
        total = (trial[i_min] - r_unpruned) / r_unpruned
        rounds.append({"rel_rmse_with_all_current": r_cur, "lights": [names[i] for i in cur],
                       "relative_increase_without": {names[i]: (trial[i] - r_cur) / r_cur for i in prunable}})
        if step < prune_margin and total <= prune_total_cap:
            pruned.append({"light": names[i_min], "rel_rmse_with": r_cur, "rel_rmse_without": trial[i_min],
                           "relative_increase": step, "cumulative_relative_increase": total})
            cur = [j for j in cur if j != i_min]
            r_cur = trial[i_min]
        else:
            break
    W2p = _solve(B, T, cur, idx)
    m2p = fit_metrics(np.einsum("knc,kc->nc", B, W2p), T)

    # 4 tint clamp
    clamped = []
    W3 = W2p.copy()
    for i in cur:
        mx = W3[i].max()
        if mx > 0 and W3[i].min() < mx / TINT_MAX_RATIO:
            before = W3[i].tolist()
            W3[i] = np.maximum(W3[i], mx / TINT_MAX_RATIO)
            clamped.append({"light": names[i], "rgb_before": before, "rgb_after": W3[i].tolist()})
    # 5 one scalar per light with the tint locked
    live = [i for i in cur if W3[i].max() > 0]
    if live:
        tint = {i: W3[i] / W3[i].max() for i in live}
        A = np.concatenate([np.stack([B[i][idx, c] * tint[i][c] for i in live], 1) for c in range(3)], 0)
        tt = np.concatenate([T[idx, c] for c in range(3)])
        sc = solve_channel(A, tt)
        for k, i in enumerate(live):
            W3[i] = tint[i] * sc[k]
    W3[[i for i in range(K) if i not in live]] = 0
    world_floor = None
    if world_i is not None and n_sky == 0 and W3[world_i].max() < WORLD_MIN_STRENGTH:
        before = W3[world_i].tolist()
        W3[world_i] = (W3[world_i] / W3[world_i].max() * WORLD_MIN_STRENGTH if W3[world_i].max() > 0
                       else np.full(3, WORLD_MIN_STRENGTH))
        world_floor = {"rgb_before": before, "rgb_after": W3[world_i].tolist()}
    P3 = np.einsum("knc,kc->nc", B, W3)
    m3 = fit_metrics(P3, T)
    _, sh3 = _energies(E, W3)
    return {
        "weights": W3, "weights_stage1": W1, "weights_one_sun": W2, "weights_pruned": W2p,
        "metrics": {"stage1_all": m1, "stage2_one_sun": m2, "stage2b_pruned": m2p, "final": m3,
                    "final_interior_only": fit_metrics(P3[~is_sky], T[~is_sky]) if (~is_sky).any() else None,
                    "final_open_sky_only": fit_metrics(P3[is_sky], T[is_sky]) if n_sky else None},
        "energy_share_stage1": sh1, "energy_share": sh3,
        "lit_fraction": lit_frac, "black": black, "best_sun": best_sun,
        "survivors": cur, "live": live, "pruned": pruned, "prune_rounds": rounds,
        "rel_rmse_unpruned": r_unpruned, "tint_clamped": clamped, "world_floor": world_floor,
        "n_active_stage1": len(active), "n_active_stage2": len(active2), "subset": int(len(idx)),
    }


# --------------------------------------------------------------------------- files
def read_exr(p):
    import cv2
    a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"cannot read {p}")
    return a[..., :3][..., ::-1].astype(np.float64)  # BGR -> RGB


def block(a, s):
    """Average s x s blocks (a: H x W x C or H x W), cropping the remainder."""
    H, W = a.shape[:2]
    a = a[: H // s * s, : W // s * s]
    return a.reshape((H // s, s, W // s, s) + a.shape[2:]).mean(axis=(1, 3))


def fit_files(work: Path, prune_margin: float, prune_total_cap: float) -> dict:
    import cv2
    t0 = time.time()
    man = common.read_json(work / "basis" / "manifest.json")
    s = int(man["block"])
    cands = man["candidates"]
    tgt8 = cv2.imread(man["target"], cv2.IMREAD_UNCHANGED)
    ids = cv2.imread(man["id"], cv2.IMREAD_UNCHANGED)
    if tgt8 is None or ids is None:
        raise RuntimeError("cannot read the target or the id map")
    if ids.ndim == 3:
        ids = ids[..., 0]
    ids = ids.astype(np.int64)
    tgt = tgt8[..., :3][..., ::-1].astype(np.float64) / (65535.0 if tgt8.dtype == np.uint16 else 255.0)
    H, W_ = ids.shape
    if tgt.shape[:2] != (H, W_):
        raise ValueError(f"target {tgt.shape[:2]} and id map {(H, W_)} differ in size")
    tlin = block(srgb_to_lin(tgt), s)
    h, w = tlin.shape[:2]
    idb = ids[: h * s, : w * s].reshape(h, s, w, s)
    id_lo = idb[:, 0, :, 0]
    id_same = np.all(idb == id_lo[:, None, :, None], axis=(1, 3))
    sat = np.any(tgt[: h * s, : w * s].reshape(h, s, w, s, 3) >= SAT_SRGB, axis=(1, 3, 4))
    valid = id_same & (id_lo > 0)
    sky = id_same & (id_lo == 0)
    tlin = np.where(sky[..., None], np.minimum(tlin, 1.0), tlin)
    mask = (valid & ~sat) | sky
    objs = {int(k): v for k, v in man.get("objects", {}).items()}

    B = np.stack([read_exr(c["exr"]) for c in cands])
    if B.shape[1:3] != (h, w):
        raise ValueError(f"basis images are {B.shape[1:3]}, expected {(h, w)} (block {s})")
    K = len(cands)
    Bf = B.reshape(K, -1, 3)
    mflat, vflat = mask.reshape(-1), valid.reshape(-1)
    Tm = tlin.reshape(-1, 3)[mflat]
    is_sky = sky.reshape(-1)[mflat]
    names = [c["name"] for c in cands]
    r = solve_lights(Bf[:, mflat], Tm, [c["kind"] for c in cands], [c.get("sun_group") for c in cands],
                     is_sky=is_sky, energy_bases=Bf[:, vflat], prune_margin=prune_margin,
                     prune_total_cap=prune_total_cap, names=names)
    W3 = r["weights"]
    full = np.einsum("knc,kc->nc", Bf, W3).reshape(h, w, 3)

    per_obj = {}
    for oid in sorted(np.unique(id_lo[mask]).tolist()):
        mo = mask & (id_lo == oid)
        pf, tf = full[mo], tlin[mo]
        o = objs.get(oid, {"name": "open sky (id 0)" if oid == 0 else str(oid), "category": ""})
        ratio = float((pf @ LUMA).mean() / max((tf @ LUMA).mean(), 1e-9))
        per_obj[o["name"]] = {"id": oid, "category": o.get("category", ""), "pixels_basis_res": int(mo.sum()),
                              "mean_ratio_luminance": ratio,
                              "mean_ratio_rgb": [float(pf[:, c].mean() / max(tf[:, c].mean(), 1e-9)) for c in range(3)],
                              "verdict": "too dark" if ratio < 0.85 else "too bright" if ratio > 1.15
                              else "ok (within 15%)"}

    pruned_names = {p["light"] for p in r["pruned"]}
    sol = [{**c, "black_skipped": i in r["black"],
            "lit_fraction_valid_px": float(r["lit_fraction"][i]),
            "rgb_weight_stage1_all": r["weights_stage1"][i].tolist(),
            "rgb_weight_stage2_one_sun": r["weights_one_sun"][i].tolist(),
            "rgb_weight_after_pruning": r["weights_pruned"][i].tolist(),
            "pruned": c["name"] in pruned_names,
            "rgb_weight_final": W3[i].tolist(),
            "energy_share_stage1": float(r["energy_share_stage1"][i]),
            "energy_share_final": float(r["energy_share"][i]),
            "keep": bool(W3[i].max() > 0)} for i, c in enumerate(cands)]
    common.write_json(work / "basis" / "solution.json", {"candidates": sol})

    m = r["metrics"]
    fit = {
        "model": "target_lin[c] ~= sum_i w_ic basis_i[c], w >= 0, per channel; bases = Cycles renders of "
                 "unit-strength white lights with the real materials",
        "basis_block": s, "basis_resolution": [w, h],
        "pixel_rule": f"basis pixel used if its {s}x{s} full-res ids are equal and either > 0 with every full-res "
                      f"target sRGB channel < {SAT_SRGB:.4f}, or == 0 (open sky; kept even if saturated, target "
                      "clipped at 1.0)",
        "pixels_open_sky": int(is_sky.sum()),
        "world_min_strength_rule": f"only if no open-sky pixels: world raised to strength {WORLD_MIN_STRENGTH}",
        "world_floor_applied": r["world_floor"],
        "pixels_used": int(mflat.sum()), "pixels_valid": int(valid.sum()),
        "pixels_excluded_saturated": int((valid & sat).sum()), "solver_subset_pixels": r["subset"],
        "loss": f"IRLS relative error (pred-t)/(t+{EPS}), Huber delta {HUBER}, {IRLS_ITERS} iterations, "
                "scipy.optimize.nnls",
        "metric_definitions": {"rel_rmse": f"sqrt(mean(((fit-t)/(t+{EPS}))^2)) over pixels used",
                               "nrmse": "||fit-t|| / ||t||",
                               "r2": "1 - SS_res/SS_tot (per channel, or all channels about per-channel means)"},
        "black_bases_skipped": [names[i] for i in r["black"]],
        "stage1_all_candidates": {"n_active": r["n_active_stage1"], "metrics": m["stage1_all"]},
        "best_sun": {g: (names[b] if b is not None else None) for g, b in r["best_sun"].items()},
        "stage2_one_sun_per_group": {"n_active": r["n_active_stage2"], "metrics": m["stage2_one_sun"]},
        "pruning": {"rule": f"greedy: drop the light whose removal (re-solving the rest) raises rel_rmse least, "
                            f"if that rise < {prune_margin:.3%} of the current value and the cumulative rise <= "
                            f"{prune_total_cap:.3%} of the unpruned value; repeat; the world is exempt",
                    "margin": prune_margin, "total_cap": prune_total_cap,
                    "rel_rmse_before": r["rel_rmse_unpruned"],
                    "rel_rmse_after": m["stage2b_pruned"]["overall"]["rel_rmse"],
                    "pruned": r["pruned"], "rounds": r["prune_rounds"],
                    "survivors": [names[i] for i in r["survivors"]]},
        "stage2b_after_pruning": {"n_active": len(r["survivors"]), "metrics": m["stage2b_pruned"]},
        "tint_clamp_rule": f"max/min channel ratio <= {TINT_MAX_RATIO}: channels below max/{TINT_MAX_RATIO} are "
                           "raised to it, then one scalar per light is re-solved with the tint locked",
        "tint_clamped": r["tint_clamped"],
        "final": {"n_lights": len(r["live"]), "metrics": m["final"],
                  "metrics_interior_only": m["final_interior_only"], "metrics_open_sky_only": m["final_open_sky_only"]},
        "per_object_mean_ratio_fit_over_target": per_obj,
        "seconds": round(time.time() - t0, 2),
    }
    common.write_json(work / "fit.json", fit)

    # preview: fit | target | log2 ratio
    fit_s = (lin_to_srgb(full) * 255 + 0.5).astype(np.uint8)
    tgt_s = (lin_to_srgb(tlin) * 255 + 0.5).astype(np.uint8)
    lr = np.clip(np.log2((full @ LUMA + 1e-3) / (tlin @ LUMA + 1e-3)), -1, 1)  # +-1 stop full scale
    ratio = np.zeros((h, w, 3), np.uint8)
    ratio[..., 0] = (255 * np.clip(lr, 0, 1)).astype(np.uint8)
    ratio[..., 2] = (255 * np.clip(-lr, 0, 1)).astype(np.uint8)
    ratio[..., 1] = (60 * (1 - np.abs(lr))).astype(np.uint8)
    ratio[~mask] = (40, 40, 40)
    fs = max(0.35, w / 1150.0)
    bar = int(34 * fs / 0.5)
    tiles = []
    for img, lab in ((fit_s, "fitted light combination (Cycles bases)"), (tgt_s, "target"),
                     (ratio, "log2 fit/target: red too bright, blue too dark (+-1 stop), grey unused")):
        t = np.zeros((h + bar, w, 3), np.uint8)
        t[bar:] = img
        cv2.putText(t, lab, (8, int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    sep = np.full((h + bar, 6, 3), 255, np.uint8)
    sheet = np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], 1)
    foot = np.zeros((bar, sheet.shape[1], 3), np.uint8)
    o = m["final"]["overall"]
    cv2.putText(foot, f"final: {len(r['live'])} lights, rel RMSE {o['rel_rmse']:.3f}, R2 {o['r2']:.3f}, "
                f"NRMSE {o['nrmse']:.3f}, {int(mflat.sum())} px", (8, int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(work / "fit_preview.png"), np.concatenate([sheet, foot], 0)[..., ::-1])
    print(f"[solve] stage1 {m['stage1_all']['overall']}\n[solve] stage2 {m['stage2_one_sun']['overall']}\n"
          f"[solve] final {o}\n[solve] best sun {fit['best_sun']}, pruned {[p['light'] for p in r['pruned']]}, "
          f"survivors {fit['pruning']['survivors']}, black {fit['black_bases_skipped']}, "
          f"clamped {len(r['tint_clamped'])}", flush=True)
    return fit


def main(argv: list[str] | None = None) -> int:
    common.setup_model_env()
    ap = argparse.ArgumentParser(prog="b2r.relight_solve")
    ap.add_argument("--work", required=True, help="directory holding basis/manifest.json")
    ap.add_argument("--prune-margin", type=float, default=0.01)
    ap.add_argument("--prune-total-cap", type=float, default=0.03)
    args = ap.parse_args(argv)
    fit_files(Path(args.work).resolve(), args.prune_margin, args.prune_total_cap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
