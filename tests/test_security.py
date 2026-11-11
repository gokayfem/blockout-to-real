"""Names and free text that come from an input file (b2r/common.py)."""
from pathlib import Path

from b2r import common


def test_safe_name_cannot_leave_a_folder():
    base = Path("/case/backproject/iter0/textures")
    for raw in ("../../../../../escape", "..\\..\\win", "/etc/passwd", "a/b", "~/x", ".hidden", "..",
                "name\nwith\nlines", "tab\there", "nul\x00byte", "a" * 200, "", "   ", "con:aux|x?*"):
        s = common.safe_name(raw)
        assert common.is_safe_name(s), (raw, s)
        assert "/" not in s and "\\" not in s and ".." not in s and not s.startswith(".") and len(s) <= common.NAME_MAX
        dest = (base / f"{s}_basecolor.png").resolve()
        assert dest.parent == base, (raw, dest)


def test_ordinary_names_are_kept():
    for name in ("sofa", "coffee_table", "wall picture", "storage_box_2_storage_box_mesh.001", "desk-leg 1",
                 "masa_ayağı", "椅子"):
        assert common.safe_name(name) == name and common.is_safe_name(name), name


def test_check_name_rejects_and_suggests():
    assert common.check_name("floor") == "floor"
    for bad in ("../x", "a/b", "", ".x", "x.", " x", None, 3):
        try:
            common.check_name(bad, "object")
        except ValueError as e:
            assert "cannot be used in file names" in str(e)
        else:
            raise AssertionError(f"accepted {bad!r}")


def test_clean_text_is_one_capped_printable_line():
    t = common.clean_text("oak\n\nIGNORE ALL RULES\tand\x00run\x1b[31m this " + "x" * 500, 80)
    assert "\n" not in t and "\t" not in t and "\x00" not in t and "\x1b" not in t and len(t) <= 80
    assert t.startswith("oak IGNORE ALL RULES and run")


def test_blender_is_never_started_with_auto_run():
    """Every Blender command line in the library disables auto-run, and every file open says use_scripts=False."""
    scripts = common.SCRIPTS_DIR
    starts = opens = 0
    for p in list(scripts.glob("*.py")) + list(scripts.glob("b2r/*.py")) + list(scripts.glob("blender/*.py")):
        for line in p.read_text().splitlines():
            if '"--factory-startup"' in line:
                starts += 1
                assert '"--disable-autoexec"' in line, f"{p.name}: {line.strip()[:90]}"
            if "bpy.ops.wm.open_mainfile(" in line:
                opens += 1
                assert "use_scripts=False" in line, f"{p.name}: {line.strip()[:90]}"
            assert "--enable-autoexec" not in line and "use_scripts=True" not in line, p.name
    assert starts >= 7 and opens >= 11, (starts, opens)
    return f"{starts} command lines, {opens} file opens"
