# tests/test_generator.py
"""Tests for synthetic image generator."""
from pathlib import Path
from fontbench.generator import SyntheticGenerator


def test_generate_single_image():
    gen = SyntheticGenerator()
    result = gen.generate_one(
        text="Hello World",
        font_name="Arial",
        font_size=24,
        font_color="black",
        font_style="regular",
        background="white",
        difficulty="easy",
    )
    assert result["image"] is not None
    assert result["image"].size[0] > 0
    assert result["metadata"]["font_family"] == "Arial"
    assert result["metadata"]["font_size"] == 24
    assert result["metadata"]["font_color"] == "black"
    assert result["metadata"]["font_style"] == "regular"
    assert result["metadata"]["script"] == "latin"
    assert result["metadata"]["difficulty"] == "easy"


def test_generate_cjk_image():
    gen = SyntheticGenerator()
    result = gen.generate_one(
        text="你好世界",
        font_name="STHeiti",
        font_size=32,
        font_color="red",
        font_style="regular",
        background="white",
        difficulty="easy",
    )
    assert result["image"] is not None
    assert result["metadata"]["script"] == "cjk"


def test_generate_colored_background():
    gen = SyntheticGenerator()
    result = gen.generate_one(
        text="Test",
        font_name="Helvetica",
        font_size=20,
        font_color="blue",
        font_style="regular",
        background="gradient",
        difficulty="medium",
    )
    assert result["image"] is not None
    assert result["metadata"]["difficulty"] == "medium"


def test_generate_multi_text_image():
    gen = SyntheticGenerator()
    text_specs = [
        {"text": "Title", "font_name": "Arial", "font_size": 48, "font_color": "black", "font_style": "bold"},
        {"text": "Body text", "font_name": "Georgia", "font_size": 16, "font_color": "gray", "font_style": "regular"},
    ]
    result = gen.generate_multi(text_specs, background="white", difficulty="hard")
    assert result["image"] is not None
    assert len(result["regions"]) == 2
    assert result["regions"][0]["metadata"]["font_family"] == "Arial"
    assert result["regions"][1]["metadata"]["font_family"] == "Georgia"
