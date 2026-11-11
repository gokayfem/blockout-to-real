"""b2r.relight_solve: recover known non-negative light weights from synthetic basis images."""
import numpy as np

from _util import Skip, need


def _bases(rng, K, N):
    """Smooth, distinct, non-negative basis images (like unit-light renders)."""
    x = np.linspace(0, 1, N)
    B = np.empty((K, N, 3))
    for k in range(K):
        prof = 0.05 + np.exp(-((x - (k + 0.5) / K) ** 2) / 0.02) + 0.2 * rng.random(N)
        B[k] = prof[:, None] * (0.7 + 0.3 * rng.random(3))[None]
    return B


def test_recovers_known_weights():
    S = need("b2r.relight_solve")
    if not hasattr(S, "solve_lights"):
        raise Skip("b2r.relight_solve has no solve_lights")
    rng = np.random.default_rng(0)
    K, N = 4, 6000
    B = _bases(rng, K, N)
    W_true = np.array([[1.0, 0.95, 0.9], [3.0, 3.0, 2.7], [0.5, 0.5, 0.5], [2.0, 1.8, 1.6]])
    T = np.einsum("knc,kc->nc", B, W_true)
    out = S.solve_lights(B, T, ["world", "area", "area", "point"], subset=N, names=list("wabc"))
    W = np.asarray(out["weights"])
    assert np.all(W >= 0)
    assert np.allclose(W, W_true, rtol=0.03), W
    assert out["pruned"] == []


def test_prunes_a_useless_light():
    S = need("b2r.relight_solve")
    rng = np.random.default_rng(1)
    K, N = 4, 6000
    B = _bases(rng, K, N)
    B[3] = 0.5 * B[1] + 0.5 * B[2] + 1e-3 * rng.random((N, 3))  # nearly a mix of two others
    W_true = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.5, 1.5, 1.5], [0.0, 0.0, 0.0]])
    T = np.einsum("knc,kc->nc", B, W_true)
    out = S.solve_lights(B, T, ["world", "area", "area", "area"], subset=N, names=list("wabc"))
    W = np.asarray(out["weights"])
    assert np.all(W >= 0)
    assert W[3].max() < 0.05 * W[1].max(), W
    assert np.allclose(W[:3], W_true[:3], rtol=0.05), W
    assert out["metrics"]["final"]["overall"]["rel_rmse"] < 0.02
