"""edit_tools: camera projection against closed forms, ellipsoid fits on synthetic clouds."""
import math

import numpy as np

from _util import look_at_camera
from b2r import edit_tools as E


def _cam():
    return E.camera_from_dict(look_at_camera([0.5, -3.0, 1.6], [0.2, 1.0, 0.9]))


def test_project_unproject_round_trip():
    cam = _cam()
    rng = np.random.default_rng(0)
    P = np.array([0.2, 1.0, 0.9]) + rng.normal(0, 0.6, (500, 3))
    u, v, z = E.project(cam, P)
    assert (z > 0).all()
    Q = E.unproject(cam, u, v, z)
    assert np.abs(Q - P).max() < 1e-9


def test_projection_closed_form():
    cam = _cam()
    u, v, z = E.project(cam, np.array([[0.2, 1.0, 0.9]]))  # the look-at point is the principal point
    assert abs(u[0] - cam.cx) < 1e-9 and abs(v[0] - cam.cy) < 1e-9
    assert abs(z[0] - np.linalg.norm(np.array([0.2, 1.0, 0.9]) - cam.origin)) < 1e-9
    # a point one unit right of the axis at depth d lands fx/d pixels right of cx; up is image-up
    c2w = cam.c2w
    d = 2.5
    p = cam.origin + d * cam.forward + 1.0 * c2w[:3, 0] + 0.5 * c2w[:3, 1]
    u, v, _ = E.project(cam, p[None])
    assert abs(u[0] - (cam.cx + cam.fx / d)) < 1e-9 and abs(v[0] - (cam.cy - 0.5 * cam.fy / d)) < 1e-9


def test_unproject_mask_with_depth_map():
    cam = _cam()
    depth = np.full((cam.height, cam.width), 3.0, np.float32)
    depth[:10] = 0  # no object
    m = np.zeros(depth.shape, bool)
    m[5:20, 100:110] = True
    pts, u, v = E.unproject_mask(cam, m, depth)
    assert len(pts) == 10 * 10  # rows 5..9 dropped (depth 0)
    assert np.allclose((pts - cam.origin) @ cam.forward, 3.0)


def test_euler_round_trip():
    for e in ([10, 20, 30], [-75, 40, 170], [0, 0, 0], [5, -80, -20]):
        R = E.rotmat_from_euler_xyz_deg(e)
        assert np.allclose(R.T @ R, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-12
        assert np.allclose(E.rotmat_from_euler_xyz_deg(E.euler_xyz_deg(R)), R, atol=1e-9)


def _ellipsoid_cloud(c, R, semi, n, rng):
    x = rng.normal(size=(n * 3, 3))
    x = x[np.linalg.norm(x, axis=1) > 0]
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * rng.random((len(x), 1)) ** (1 / 3)  # uniform in ball
    with np.errstate(all="ignore"):  # spurious Accelerate matmul flags on macOS
        return c + (x[:n] * semi) @ R.T


def test_fit_ellipsoid_pca_recovers_axes():
    rng = np.random.default_rng(1)
    R = E.rotmat_from_euler_xyz_deg([20, -35, 60])
    semi = np.array([1.2, 0.6, 0.25])
    c = np.array([1.0, 2.0, 0.5])
    P = _ellipsoid_cloud(c, R, semi, 40000, rng)
    # uniform solid ellipsoid: std along an axis = semi / sqrt(5), so semi = sqrt(5) * std
    C, V, s = E.fit_ellipsoid_pca(P, sigma_scale=math.sqrt(5))
    assert np.linalg.norm(C - c) < 0.01
    assert np.allclose(s, semi, rtol=0.02), s
    for i in range(3):
        assert abs(abs(V[:, i] @ R[:, i]) - 1) < 1e-3
    assert np.linalg.det(V) > 0


def test_cluster_ellipsoids_separates_blobs():
    rng = np.random.default_rng(2)
    cents = [np.array([0.0, 0, 0]), np.array([3.0, 0, 0]), np.array([0.0, 4, 1])]
    P = np.vstack([_ellipsoid_cloud(c, np.eye(3), np.array([0.5, 0.4, 0.3]), 3000 + 1000 * i, rng)
                   for i, c in enumerate(cents)])
    out = E.cluster_ellipsoids(P, 3, seed=0, sigma_scale=math.sqrt(5))
    assert len(out) == 3 and [o[3] for o in out] == [5000, 4000, 3000]
    for (C, R, semi, n), c in zip(out, cents[::-1]):
        assert np.linalg.norm(C - c) < 0.03
        assert np.allclose(sorted(semi), [0.3, 0.4, 0.5], rtol=0.05)


def test_ellipsoid_part_silhouette():
    cam = _cam()
    R = E.rotmat_from_euler_xyz_deg([0, 0, 30])
    part = E.ellipsoid_part([0.2, 1.0, 0.9], R, [0.4, 0.3, 0.2], obj_location=[0.2, 1.0, 0.0])
    assert part["type"] == "sphere" and np.allclose(part["location"], [0, 0, 0.9])
    lo, hi = E.part_world_extent(part, [0.2, 1.0, 0.0])
    assert np.all(lo < [0.2, 1.0, 0.9]) and np.all(hi > [0.2, 1.0, 0.9])
    m = E.rasterize_parts(cam, [part], [0.2, 1.0, 0.0])
    assert m[int(cam.cy), int(cam.cx)] and m.sum() > 100 and not m[0, 0]
    cyl = E.cylinder_between([0, 0, 0], [0, 0, 2], 0.1)
    assert abs(cyl["depth"] - 2) < 1e-9 and np.allclose(cyl["location"], [0, 0, 1])
    assert E.iou(m, m) == 1.0 and E.iou(m, ~m) == 0.0
