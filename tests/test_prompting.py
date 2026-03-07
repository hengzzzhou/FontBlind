# tests/test_prompting.py
"""Tests for prompt engineering strategies."""
from fontbench.prompting import ZeroShotStrategy, FewShotStrategy, CoTStrategy


def test_zero_shot_formats_mc():
    s = ZeroShotStrategy()
    prompt = s.format_mc("What font is this?", ["Arial", "Georgia", "Verdana", "Courier"])
    assert "What font is this?" in prompt
    assert "A)" in prompt
    assert "D)" in prompt


def test_few_shot_formats_mc():
    s = FewShotStrategy()
    prompt = s.format_mc("What font is this?", ["Arial", "Georgia", "Verdana", "Courier"])
    assert "Example" in prompt or "example" in prompt


def test_cot_formats_mc():
    s = CoTStrategy()
    prompt = s.format_mc("What font is this?", ["Arial", "Georgia", "Verdana", "Courier"])
    assert "step" in prompt.lower() or "think" in prompt.lower()


def test_zero_shot_formats_open_ended():
    s = ZeroShotStrategy()
    prompt = s.format_open_ended("Describe the font.")
    assert "Describe the font." in prompt


def test_strategy_names():
    assert ZeroShotStrategy().name == "zero_shot"
    assert FewShotStrategy().name == "few_shot"
    assert CoTStrategy().name == "cot"
