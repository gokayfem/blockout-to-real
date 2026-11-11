"""The time-lapse composer (b2r/timelapse.py) on a tiny synthetic case: no Blender, no models."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from _util import Skip, need


def _img(path: Path, color, size=(64, 48)):
    Image = need("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _case(root: Path, frames: bool):
    root.mkdir(parents=True, exist_ok=True)
    (root / "case.json").write_text(json.dumps({"name": "tiny", "size": {"width": 64, "height": 48}}))
    for k in (0, 1):
        _img(root / "gbuffers" / f"iter{k}" / "beauty.png", (120 + 40 * k,) * 3)
        _img(root / "target" / f"iter{k}" / "target.png", (200, 120, 60))
    for name, col in (("albedo", (220, 150, 90)), ("roughness", (128,) * 3), ("normal_cam", (128, 128, 255))):
        _img(root / "decompose" / "iter1" / f"{name}.png", col)
    (root / "final").mkdir(parents=True, exist_ok=True)
    (root / "final" / "summary.json").write_text(json.dumps({"final_iteration": 1, "iterations": {"1": {
        "blockout_vs_target": {"lpips_alex": 0.5}, "headline_lpips_alex": 0.2}}}))
    _img(root / "final" / "render_final.png", (190, 120, 70))
    if not frames:
        return
    fr = root / "timelapse" / "frames"
    for i in range(4):
        _img(fr / "reveal" / f"f_{i:04d}.png", (60 * i, 90, 90))
    for i in (1, 2):
        _img(fr / "lights" / f"f_{i:04d}.png", (40 * i, 40 * i, 20))
    for i in range(1, 6):
        _img(fr / "orbit" / f"f_{i:04d}.png", (30 * i, 100, 160))
    for i in range(1, 4):
        _img(fr / "relight" / f"f_{i:04d}.png", (200, 60 * i, 40))
    (fr / "frames.json").write_text(json.dumps({
        "iteration": 1, "reveal": {"count": 4, "order": ["floor", "desk", "lamp"]},
        "lights": {"count": 2, "order": [{"name": "key", "virtual": False, "energy_share": 0.7},
                                         {"name": "virtual_fill", "virtual": True, "energy_share": 0.3}]},
        "orbit": {"count": 5, "amplitude_deg": 6.0}, "relight": {"count": 3, "look": "warm evening"}}))


def _compose(frames: bool):
    if not shutil.which("ffmpeg"):
        raise Skip("ffmpeg not on PATH")
    need("numpy")
    from b2r import timelapse
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "tiny"
        _case(root, frames)
        out = timelapse.compose(root, fps=12, width=320)
        info = json.loads((root / "timelapse" / "timelapse.json").read_text())
        assert out.exists() and out.stat().st_size > 2000 and (root / "timelapse" / "thumb.jpg").exists()
        if shutil.which("ffprobe"):
            n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames,width,height", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True).stdout.strip().split(",")
            assert [int(n[0]), int(n[1])] == info["size"] and int(n[2]) == info["frames"], (n, info)
        return info


def test_all_steps():
    info = _compose(True)
    assert info["steps"] == ["Blockout", "Photo", "Maps", "Edits", "Materials", "Lights", "Compare", "Camera", "Relight"]
    assert info["left_out"] == [] and info["final_iteration"] == 1
    return f"{info['frames']} frames, {info['seconds']} s"


def test_missing_frames_are_left_out_not_faked():
    info = _compose(False)
    assert info["steps"] == ["Blockout", "Photo", "Maps", "Edits", "Compare"]
    assert len(info["left_out"]) == 4 and all("no timelapse/frames" in s for s in info["left_out"])
