"""Shared bits of the test runner: Skip, @slow, optional imports, a synthetic camera."""
import importlib

import numpy as np


class Skip(Exception):
    pass


def slow(fn):
    fn.slow = True
    return fn


def need(module: str):
    """Import a module or skip the test with the reason printed."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise Skip(f"{module} not importable ({e})")


def look_at_camera(eye, target, W=320, H=240, f=300.0):
    """A camera.json dict (conventions.md) looking from eye to target, world Z up."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return {"width": W, "height": H, "fx": f, "fy": f, "cx": W / 2, "cy": H / 2,
            "cam_to_world": c2w.tolist(), "world_to_cam": np.linalg.inv(c2w).tolist()}
