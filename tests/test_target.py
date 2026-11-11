"""b2r.target.edge_scores: the conditioning ladder's edge metric on a synthetic id map and image."""
import numpy as np

from _util import Skip, need


def _scene(H=200, W=260):
    ids = np.zeros((H, W), np.int64)
    ids[40:160, 30:120] = 1
    ids[60:180, 150:240] = 2
    img = np.full((H, W, 3), 90, np.uint8)
    img[ids == 1] = (200, 60, 40)
    img[ids == 2] = (40, 180, 220)
    return ids, img


def test_aligned_edges_score_high():
    T = need("b2r.target")
    if not hasattr(T, "edge_scores"):
        raise Skip("b2r.target has no edge_scores")
    ids, img = _scene()
    s = T.edge_scores(ids, img)
    assert s["recall3"] > 0.95, s
    assert s["mean_dist_px"] < 1.5
    assert set(s["per_object"]) == {0, 1, 2}


def test_misaligned_or_blank_scores_low():
    T = need("b2r.target")
    ids, img = _scene()
    shifted = np.roll(img, 12, axis=1)   # every edge 12 px off
    s = T.edge_scores(ids, shifted)
    assert s["recall3"] < 0.5 and s["mean_dist_px"] > 5, s
    blank = np.full_like(img, 128)
    b = T.edge_scores(ids, blank)
    assert b["recall3"] == 0.0 and b["edge_density"] == 0.0
    try:
        T.edge_scores(ids[:-1], img)
    except ValueError:
        pass
    else:
        raise AssertionError("size mismatch accepted")
