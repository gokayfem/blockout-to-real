"""Pure helpers for turning a critique or a residual into blockout geometry edits.

numpy + scipy + opencv only; no Blender, no scene names. Every input comes from the case
(gbuffers/iter<k>/camera.json, depth.exr, id.png, decompose/iter<k>/depth_moge.exr ...).

Camera conventions (references/conventions.md): Blender camera axes (+X right, +Y up, -Z forward),
pixel centre of column i / row j is (i+0.5, j+0.5), u = cx + fx*Xc/(-Zc), v = cy - fy*Yc/(-Zc).
"Depth" means planar camera depth (-Zc), as in depth.exr.

Entry points
------------
load_camera(path) / camera_from_dict(d) -> Camera
read_depth(path) / read_ids(path) / read_position(path)
unproject(cam, u, v, depth)             -> world points (N, 3)
unproject_mask(cam, mask, depth_map, clamp=(lo, hi)) -> points, u, v
unproject_box(cam, box, depth_map)      -> points of the box pixels
project(cam, pts)                       -> u, v, depth
rays_hit_z(cam, u, v, z0) / rays_hit_plane(cam, u, v, p0, n)
kmeans(points, k, seed)                 -> labels, centres (deterministic)
fit_ellipsoid_pca(points)               -> centre, R (columns = axes, major first), semi-axes
fit_flat_ellipsoid(cam, u, v, pts, ...) -> thin ellipsoid whose silhouette matches a pixel cluster
fit_blobs_on_plane(cam, mask, p0, n, k) -> volumetric ellipsoids for a far region of unknown depth
cluster_ellipsoids(points, k)           -> one PCA ellipsoid per k-means cluster
ellipsoid_part(centre, R, semi, obj_location) -> spec part dict ("sphere" + size + rotation)
cylinder_between(p0, p1, r, obj_location)     -> spec cylinder part along p0 -> p1
part_world_extent(part, obj_location)          -> (min xyz, max xyz) of a spec part
rasterize_parts(cam, parts, obj_location, obj_rotation, shape) -> bool silhouette mask
euler_xyz_deg(R) / rotmat_from_euler_xyz_deg(e)  Blender "XYZ" euler (R = Rz @ Ry @ Rx)
iou(a, b)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import common

common.setup_model_env()  # OPENCV_IO_ENABLE_OPENEXR before cv2 import
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.cluster.vq import kmeans2  # noqa: E402


# --------------------------------------------------------------------------- io
@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    c2w: np.ndarray  # 4x4
    w2c: np.ndarray  # 4x4

    @property
    def origin(self) -> np.ndarray:
        return self.c2w[:3, 3].copy()

    @property
    def forward(self) -> np.ndarray:
        return -self.c2w[:3, 2].copy()


def camera_from_dict(c: dict) -> Camera:
    return Camera(float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"]),
                  int(c["width"]), int(c["height"]),
                  np.array(c["cam_to_world"], dtype=np.float64),
                  np.array(c["world_to_cam"], dtype=np.float64))


def load_camera(path) -> Camera:
    with open(path) as f:
        return camera_from_dict(json.load(f))


def read_depth(path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    if d.ndim == 3:
        d = d[..., 2]  # BGR -> take R channel (all equal for a depth EXR)
    return d.astype(np.float32)


def read_ids(path) -> np.ndarray:
    ids = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if ids is None:
        raise FileNotFoundError(path)
    if ids.ndim == 3:
        ids = ids[..., 0]
    return ids


def read_position(path) -> np.ndarray:
    p = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return p[..., ::-1].astype(np.float32)  # BGR -> XYZ


# --------------------------------------------------------------------------- projection
def camera_dirs(cam: Camera, u, v) -> np.ndarray:
    """World-space ray directions scaled so that planar depth 1 = 1 unit along it."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    xc = (u - cam.cx) / cam.fx
    yc = -(v - cam.cy) / cam.fy
    dc = np.stack([xc, yc, -np.ones_like(xc)], axis=-1)
    with np.errstate(all="ignore"):  # spurious Accelerate matmul warnings on macOS
        return dc @ cam.c2w[:3, :3].T


def unproject(cam: Camera, u, v, depth) -> np.ndarray:
    """Pixel coords (continuous; pixel centre = i+0.5) + planar depth -> world XYZ."""
    d = camera_dirs(cam, u, v)
    return cam.origin + d * np.asarray(depth, dtype=np.float64)[..., None]


def pixel_centres(mask: np.ndarray):
    rows, cols = np.nonzero(mask)
    return cols + 0.5, rows + 0.5, rows, cols


def unproject_mask(cam: Camera, mask: np.ndarray, depth_map: np.ndarray, clamp=None):
    """Mask -> world points using a depth map. clamp=(lo, hi) clips depth; pixels
    with depth <= 0 or non-finite are dropped. Returns points, u, v."""
    u, v, r, c = pixel_centres(mask)
    d = depth_map[r, c].astype(np.float64)
    ok = np.isfinite(d) & (d > 0)
    if clamp is not None:
        d = np.clip(d, clamp[0], clamp[1])
    u, v, d = u[ok], v[ok], d[ok]
    return unproject(cam, u, v, d), u, v


def unproject_box(cam: Camera, box, depth_map: np.ndarray, clamp=None):
    """box = [x0, y0, x1, y1] in pixels (inclusive-exclusive)."""
    m = np.zeros(depth_map.shape, bool)
    x0, y0, x1, y1 = [int(round(b)) for b in box]
    m[y0:y1, x0:x1] = True
    return unproject_mask(cam, m, depth_map, clamp)


def project(cam: Camera, pts: np.ndarray):
    p = np.asarray(pts, dtype=np.float64)
    with np.errstate(all="ignore"):  # spurious Accelerate matmul warnings on macOS
        pc = p @ cam.w2c[:3, :3].T + cam.w2c[:3, 3]
    z = -pc[..., 2]
    u = cam.cx + cam.fx * pc[..., 0] / z
    v = cam.cy - cam.fy * pc[..., 1] / z
    return u, v, z


def rays_hit_z(cam: Camera, u, v, z0: float) -> np.ndarray:
    """Where the pixel rays meet the horizontal plane z = z0 (NaN if parallel/behind)."""
    d = camera_dirs(cam, u, v)
    t = (z0 - cam.origin[2]) / d[..., 2]
    t = np.where(t > 0, t, np.nan)
    return cam.origin + d * t[..., None]


def rays_hit_plane(cam: Camera, u, v, p0, n) -> np.ndarray:
    d = camera_dirs(cam, u, v)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(all="ignore"):  # spurious Accelerate matmul warnings on macOS
        t = ((np.asarray(p0) - cam.origin) @ n) / (d @ n)
    return cam.origin + d * t[..., None]


# --------------------------------------------------------------------------- rotations
def rotmat_from_euler_xyz_deg(e) -> np.ndarray:
    rx, ry, rz = [math.radians(float(a)) for a in (e if e is not None else (0, 0, 0))]
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # Blender Euler 'XYZ'


def euler_xyz_deg(R: np.ndarray) -> list[float]:
    """Inverse of rotmat_from_euler_xyz_deg (Blender 'XYZ')."""
    R = np.asarray(R, dtype=np.float64)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def rot_z_to(direction) -> np.ndarray:
    """Rotation taking local +Z to `direction` (for cylinders/cones)."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    s, c = np.linalg.norm(v), float(z @ d)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


# --------------------------------------------------------------------------- clustering / fitting
def kmeans(points: np.ndarray, k: int, seed: int = 0, iters: int = 50):
    pts = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    # k-means++ style deterministic init
    cent = [pts[rng.integers(len(pts))]]
    for _ in range(1, k):
        d2 = np.min(((pts[:, None, :] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
        cent.append(pts[rng.choice(len(pts), p=d2 / d2.sum())])
    cent, labels = kmeans2(pts, np.array(cent), iter=iters, minit="matrix")
    return labels, cent


def fit_ellipsoid_pca(points: np.ndarray, sigma_scale: float = 2.0):
    """Centre, rotation (columns = principal axes, major first, det +1) and
    semi-axes (= sigma_scale * std, 2 = uniformly filled ellipse)."""
    p = np.asarray(points, dtype=np.float64)
    c = p.mean(0)
    w, V = np.linalg.eigh(np.cov((p - c).T))
    order = np.argsort(w)[::-1]
    w, V = np.maximum(w[order], 0), V[:, order]
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return c, V, sigma_scale * np.sqrt(w)


def fit_flat_ellipsoid(cam: Camera, u, v, pts, thickness: float = 0.02, max_tilt_deg: float = 55.0,
             sigma_scale: float = 2.0, depth=None):
    """Flattened ellipsoid for one thin, roughly planar pixel cluster (a panel, a sheet, a flat cluster).

    In-plane axes come from the 2D pixel distribution (so the silhouette matches
    the photo); the plane passes through the cluster's median-depth point and
    has the normal of the 3D point cloud's PCA, tilted at most max_tilt_deg away
    from facing the camera (noisy monocular depth can otherwise make it edge-on).
    Returns centre, R (columns: major, minor, normal), semi-axes.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    pts = np.asarray(pts, float)
    uv = np.stack([u, v], 1)
    c2 = uv.mean(0)
    w2, V2 = np.linalg.eigh(np.cov((uv - c2).T))
    order = np.argsort(w2)[::-1]
    w2, V2 = np.maximum(w2[order], 1e-6), V2[:, order]
    a_px = sigma_scale * np.sqrt(w2)
    # planar depth of the cluster centre = median planar depth of its points
    d0 = float(np.median((pts - cam.origin) @ cam.forward)) if depth is None else float(depth)
    C = unproject(cam, c2[0], c2[1], d0)
    # plane normal from 3D PCA
    _, V3, _ = fit_ellipsoid_pca(pts)
    n = V3[:, 2]
    view = cam.origin - C
    view /= np.linalg.norm(view)
    if n @ view < 0:
        n = -n
    ang = math.degrees(math.acos(float(np.clip(n @ view, -1, 1))))
    if ang > max_tilt_deg:
        # rotate n toward view so the angle is max_tilt_deg
        axis = np.cross(view, n)
        axis /= np.linalg.norm(axis)
        th = math.radians(max_tilt_deg)
        n = view * math.cos(th) + np.cross(axis, view) * math.sin(th)
    n /= np.linalg.norm(n)
    ends = []
    for i in range(2):
        e = V2[:, i] * a_px[i]
        P = rays_hit_plane(cam, np.array([c2[0] + e[0], c2[0] - e[0]]),
                           np.array([c2[1] + e[1], c2[1] - e[1]]), C, n)
        ends.append((P[0] - P[1]) / 2.0)
    A1, A2 = ends
    u1 = A1 - (A1 @ n) * n
    s1 = np.linalg.norm(u1)
    u1 /= s1
    u2 = np.cross(n, u1)
    s2 = abs(A2 @ u2)
    R = np.stack([u1, u2, n], 1)
    return C, R, np.array([s1, s2, thickness / 2.0])


def cluster_ellipsoids(points: np.ndarray, k: int, seed: int = 0, sigma_scale: float = 2.0,
                       min_points: int = 30):
    """k-means the points and fit one PCA ellipsoid per cluster. Returns a list of
    (centre, R, semi, n_points), largest cluster first; clusters under min_points are dropped."""
    pts = np.asarray(points, dtype=np.float64)
    labels, _ = kmeans(pts, k, seed=seed)
    out = []
    for c in range(k):
        sel = labels == c
        if sel.sum() < max(min_points, 4):
            continue
        C, R, semi = fit_ellipsoid_pca(pts[sel], sigma_scale)
        out.append((C, R, semi, int(sel.sum())))
    return sorted(out, key=lambda t: -t[3])


def fit_blobs_on_plane(cam: Camera, mask: np.ndarray, plane_point, plane_normal, k: int,
                       seed: int = 0, sigma_scale: float = 2.0, depth_ratio: float = 0.8):
    """Volumetric blobs for a far-field region whose depth is unknown:
    intersect the mask's pixel rays with a chosen plane, k-means the hit points and fit
    one ellipsoid per cluster (in-plane axes from PCA, depth semi-axis =
    depth_ratio * mean in-plane semi-axis). Returns a list of (centre, R, semi, n_px)."""
    u, v, _, _ = pixel_centres(mask)
    P = rays_hit_plane(cam, u, v, plane_point, plane_normal)
    labels, _ = kmeans(P, k, seed=seed)
    n = np.asarray(plane_normal, float)
    n = n / np.linalg.norm(n)
    out = []
    for c in range(k):
        sel = labels == c
        if sel.sum() < 30:
            continue
        C, R, semi = fit_ellipsoid_pca(P[sel], sigma_scale)
        a1 = R[:, 0] - (R[:, 0] @ n) * n
        a1 /= np.linalg.norm(a1)
        a2 = np.cross(n, a1)
        s1 = sigma_scale * np.std((P[sel] - C) @ a1)
        s2 = sigma_scale * np.std((P[sel] - C) @ a2)
        Rn = np.stack([a1, a2, n], 1)
        out.append((C, Rn, np.array([s1, s2, depth_ratio * 0.5 * (s1 + s2)]), int(sel.sum())))
    return out


def ellipsoid_world_half_extent(R, semi) -> np.ndarray:
    R = np.asarray(R, float)
    return np.sqrt(((R * np.asarray(semi, float)[None, :]) ** 2).sum(1))


# --------------------------------------------------------------------------- spec parts
def _r(x, nd=4):
    return [round(float(a), nd) for a in x]


def ellipsoid_part(centre, R, semi, obj_location=(0, 0, 0), segments=12, rings=6) -> dict:
    return {"type": "sphere",
            "size": _r(2 * np.asarray(semi)),
            "location": _r(np.asarray(centre) - np.asarray(obj_location)),
            "rotation": _r(euler_xyz_deg(R), 2),
            "segments": int(segments), "rings": int(rings)}


def cylinder_between(p0, p1, radius, obj_location=(0, 0, 0), segments=8) -> dict:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    return {"type": "cylinder", "radius": round(float(radius), 4),
            "depth": round(float(np.linalg.norm(d)), 4),
            "location": _r((p0 + p1) / 2 - np.asarray(obj_location)),
            "rotation": _r(euler_xyz_deg(rot_z_to(d)), 2),
            "segments": int(segments)}


def part_local_verts(part: dict, n_seg: int = 32) -> np.ndarray:
    """Surface sample points of a spec part in object-local coordinates
    (part rotation + location applied; object rotation not applied)."""
    t = part["type"]
    if t == "box":
        sx, sy, sz = part["size"]
        g = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        v = g * [sx, sy, sz]
    elif t in ("cylinder", "cone"):
        r1 = r2 = part.get("radius")
        if t == "cone":
            r1, r2 = part["radius_bottom"], part["radius_top"]
        h = part["depth"]
        a = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        bot = np.stack([r1 * np.cos(a), r1 * np.sin(a), np.full_like(a, -h / 2)], 1)
        top = np.stack([r2 * np.cos(a), r2 * np.sin(a), np.full_like(a, h / 2)], 1)
        v = np.vstack([bot, top])
    elif t == "sphere":
        dx, dy, dz = part["size"] if "size" in part else [2 * part["radius"]] * 3
        th = np.linspace(0, np.pi, 13)
        ph = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        T, P = np.meshgrid(th, ph)
        v = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
        v = v * [dx / 2, dy / 2, dz / 2]
    elif t == "plane":
        sx, sy = part["size"]
        v = np.array([[x * sx, y * sy, 0] for x in (-.5, .5) for y in (-.5, .5)])
    else:
        raise ValueError(t)
    R = rotmat_from_euler_xyz_deg(part.get("rotation"))
    return v @ R.T + np.asarray(part.get("location", (0, 0, 0)), float)


def part_world_verts(part, obj_location=(0, 0, 0), obj_rotation=None) -> np.ndarray:
    Ro = rotmat_from_euler_xyz_deg(obj_rotation)
    return part_local_verts(part) @ Ro.T + np.asarray(obj_location, float)


def part_world_extent(part, obj_location=(0, 0, 0), obj_rotation=None):
    w = part_world_verts(part, obj_location, obj_rotation)
    return w.min(0), w.max(0)


def rasterize_parts(cam: Camera, parts, obj_location=(0, 0, 0), obj_rotation=None,
                    shape=None) -> np.ndarray:
    """Silhouette of convex primitives from the camera (no occlusion by other objects)."""
    H, W = shape or (cam.height, cam.width)
    m = np.zeros((H, W), np.uint8)
    for p in parts:
        w = part_world_verts(p, obj_location, obj_rotation)
        u, v, z = project(cam, w)
        if (z <= 0).any():
            continue
        hull = cv2.convexHull(np.stack([u - 0.5, v - 0.5], 1).astype(np.float32))
        cv2.fillConvexPoly(m, np.round(hull[:, 0, :] * 16).astype(np.int32), 1,
                           lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def iou(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def save_mask(path, mask):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.asarray(mask, bool) * 255).astype(np.uint8))

