"""The case contract in b2r/common.py."""
import tempfile
from pathlib import Path

from b2r import common


def test_nearest_aspect():
    assert common.nearest_aspect(1152, 928) == "5:4"
    assert common.nearest_aspect(1920, 1080) == "16:9"
    assert common.nearest_aspect(1000, 1000) == "1:1"
    assert common.nearest_aspect(1080, 1920) == "9:16"
    assert common.nearest_aspect(2520, 1080) == "21:9"
    assert common.nearest_aspect(1024, 770) == "4:3"
    for a in common.TARGET_ASPECTS:  # every accepted ratio maps to itself
        x, y = (float(v) for v in a.split(":"))
        assert common.nearest_aspect(x * 100, y * 100) == a


def test_stage_dir():
    with tempfile.TemporaryDirectory() as d:
        p = common.stage_dir(d, "residual", 2)
        assert p == Path(d).resolve() / "residual" / "iter2" and p.is_dir()
        q = common.stage_dir(d, "target", 1, create=False)
        assert not q.exists()
        try:
            common.stage_dir(d, "nonsense", 0)
        except ValueError:
            pass
        else:
            raise AssertionError("unknown stage accepted")


def test_case_json_round_trip():
    cfg = {"name": "t", "input": {"name": "x.blend", "kind": "blend", "sha256": "0" * 64}, "camera": "Camera",
           "size": {"width": 640, "height": 480, "aspect_ratio": "4:3", "resolution": "1K", "size_verified": False},
           "look": "a photo", "settings": {"budget_usd": 1.0, "target_level": None, "group_by": "object"},
           "iterations": {}}
    with tempfile.TemporaryDirectory() as d:
        common.save_case(d, cfg)
        assert common.load_case(d) == cfg
        assert common.render_size(d) == (640, 480)
        assert not list(Path(d).glob("*.tmp"))  # atomic write leaves nothing behind
        try:
            common.load_case(Path(d) / "missing")
        except SystemExit:
            pass
        else:
            raise AssertionError("load_case accepted a folder without case.json")


def test_script_args():
    assert common.script_args(["blender", "-b", "--python", "x.py", "--", "--case", "C"]) == ["--case", "C"]
    assert common.script_args(["prog.py", "--case", "C"]) == ["--case", "C"]
    assert common.script_args(["/Applications/Blender", "-b"]) == []
