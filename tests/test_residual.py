"""The residual's small / large rule on synthetic images where the damage is controlled."""
import types

import numpy as np

from _util import Skip, need, slow


def _ev(**arrays):
    return types.SimpleNamespace(**arrays)


def test_classify_rule():
    R = need("b2r.residual")
    assert R.classify(R.MIN_PIXELS, R.GAP_RATIO_MAX + 0.01, 0.0)[0] == "large"
    assert R.classify(R.MIN_PIXELS, 0.0, R.DELTA_E_MAX + 0.1)[0] == "large"
    assert R.classify(R.MIN_PIXELS, R.GAP_RATIO_MAX, R.DELTA_E_MAX)[0] == "small"  # strict inequalities
    cls, why = R.classify(R.MIN_PIXELS - 1, 5.0, 50.0)  # tiny objects never drive edits
    assert cls == "small" and why == []


def test_per_object_rows_and_open_sky():
    """Synthetic spatial maps: object 2 keeps the blockout's error (large), object 1 is nearly fixed
    (small), id 0 (open sky) gets its own row."""
    R = need("b2r.residual")
    H, W = 120, 160
    ids = np.zeros((H, W), np.int32)
    ids[:, :60] = 1
    ids[:, 60:130] = 2  # columns 130.. stay 0 = open sky
    bl = np.full((H, W), 0.5)
    re = np.where(ids == 2, 0.45, 0.05)
    re[ids == 0] = 0.4
    zero = np.zeros((H, W))
    ev = _ev(sp=re, sp_c=re, dE=zero + 3, dE_c=zero + 3, dL=zero - 5)
    ev_bl = _ev(sp=bl, sp_c=bl)
    objs = R.per_object(ids, {1: {"name": "a", "category": ""}, 2: {"name": "b", "category": ""}}, ev, ev_bl)
    assert set(objs) == {"0", "1", "2"}
    assert objs["1"]["class"] == "small" and objs["2"]["class"] == "large"
    assert objs["0"]["is_open_sky"] and objs["0"]["class"] == "large" and objs["0"]["pixels"] == H * 30
    assert abs(objs["2"]["coarse_gap_ratio"] - 0.9) < 1e-9
    assert objs["1"]["brightness"] == "too dark"


def test_detail_height_band_and_mask():
    R = need("b2r.residual")
    rng = np.random.default_rng(0)
    H, W = 96, 128
    base = np.full((H, W, 3), 128, np.uint8)
    target = base.copy()
    target[..., :] = np.clip(128 + 40 * rng.standard_normal((H, W, 1)), 0, 255).astype(np.uint8)
    ids = np.zeros((H, W), np.int32)
    ids[:, :64] = 1
    ids[:, 64:] = 2
    objs = {"0": {"class": "large", "is_open_sky": True, "name": "sky"},
            "1": {"class": "small", "name": "a"}, "2": {"class": "large", "name": "b"}}
    h16, info = R.detail_height(target, base, ids, objs)
    h = h16.astype(float) / 65535
    assert info["small_object_ids"] == [1]
    assert np.allclose(h[:, 70:], 0.5, atol=1e-4)  # large object: neutral
    assert h[10:-10, 10:54].std() > 0.01           # small object: relief
    h2, _ = R.detail_height(base, base, ids, objs)  # no residual -> neutral everywhere
    assert np.abs(h2.astype(float) / 65535 - 0.5).max() < 1e-4


@slow
def test_lpips_classification_on_controlled_damage():
    """Real LPIPS: a target, a re-render equal to it except one object whose colour is wrong, and a grey
    blockout. The damaged object must be large, the untouched one small."""
    need("torch")
    need("lpips")
    R = need("b2r.residual")
    rng = np.random.default_rng(0)
    H, W = 192, 256
    yy, xx = np.mgrid[0:H, 0:W]
    tex = (np.sin(xx / 3.0) * np.cos(yy / 4.0) * 40 + rng.normal(0, 8, (H, W)))[..., None]
    target = np.clip(np.array([150, 110, 80]) + tex, 0, 255).astype(np.uint8)
    ids = np.ones((H, W), np.int32)
    ids[:, W // 2:] = 2
    rer = target.copy()
    rer[:, W // 2:] = np.clip(target[:, W // 2:].astype(int) + np.array([-80, 40, 90]), 0, 255).astype(np.uint8)
    blockout = np.full((H, W, 3), 128, np.uint8)
    try:
        m = R.Metrics(R.pick_device("cpu"))
    except Exception as e:  # weights not downloadable here
        raise Skip(f"LPIPS weights unavailable: {e}")
    ev, ev_bl = R.Eval(m, rer, target), R.Eval(m, blockout, target)
    objs = R.per_object(ids, {}, ev, ev_bl)
    assert objs["1"]["class"] == "small" and objs["2"]["class"] == "large", objs
    assert ev.g["lpips_alex"] < ev_bl.g["lpips_alex"]
    return f"lpips re {ev.g['lpips_alex']:.3f} bl {ev_bl.g['lpips_alex']:.3f}"
