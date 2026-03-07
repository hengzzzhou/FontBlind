# tests/test_fonts.py
"""Tests for font registry."""
from fontbench.fonts import FontRegistry


def test_registry_has_latin_fonts():
    registry = FontRegistry()
    latin = registry.get_fonts_by_script("latin")
    assert len(latin) >= 8
    assert all(f.script == "latin" for f in latin)


def test_registry_has_cjk_fonts():
    registry = FontRegistry()
    cjk = registry.get_fonts_by_script("cjk")
    assert len(cjk) >= 2
    assert all(f.script == "cjk" for f in cjk)


def test_registry_has_other_fonts():
    registry = FontRegistry()
    other = registry.get_fonts_by_script("other")
    assert len(other) >= 2


def test_font_has_required_attributes():
    registry = FontRegistry()
    font = registry.get_fonts_by_script("latin")[0]
    assert hasattr(font, "name")
    assert hasattr(font, "path")
    assert hasattr(font, "script")
    assert hasattr(font, "sub_script")
    assert hasattr(font, "styles")


def test_font_sub_script_values():
    registry = FontRegistry()
    for font in registry.all_fonts():
        assert font.sub_script in ("latin", "chinese", "arabic", "devanagari"), (
            f"Unexpected sub_script '{font.sub_script}' for font '{font.name}'"
        )


def test_get_fonts_by_sub_script():
    registry = FontRegistry()
    chinese = registry.get_fonts_by_sub_script("chinese")
    assert len(chinese) >= 2
    assert all(f.sub_script == "chinese" for f in chinese)


def test_font_path_exists():
    registry = FontRegistry()
    for font in registry.all_fonts():
        assert font.path.exists(), f"Font file not found: {font.path}"
