# tests/test_cv_baseline.py
"""Tests for traditional CV baseline evaluator."""
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
from fontbench.cv_baseline import CVBaselineEvaluator, _detect_text_color, _detect_font_size


def _make_colored_text_image(text_color_rgb, bg_color=(255, 255, 255)):
    """Create a synthetic image with a colored block simulating text on a background."""
    img = Image.new("RGB", (200, 80), bg_color)
    draw = ImageDraw.Draw(img)
    # Draw a solid rectangle to simulate text — avoids antialiasing issues with small fonts
    draw.rectangle([40, 20, 160, 60], fill=text_color_rgb)
    return img


def test_color_detection_red():
    img = _make_colored_text_image((220, 50, 50))
    detected = _detect_text_color(img)
    assert detected == "red", f"Expected 'red', got '{detected}'"


def test_color_detection_blue():
    img = _make_colored_text_image((50, 50, 220))
    detected = _detect_text_color(img)
    assert detected == "blue", f"Expected 'blue', got '{detected}'"


def test_color_detection_black():
    img = _make_colored_text_image((0, 0, 0))
    detected = _detect_text_color(img)
    assert detected == "black", f"Expected 'black', got '{detected}'"


def test_size_detection_returns_valid_option():
    img = _make_colored_text_image((0, 0, 0))
    detected = _detect_font_size(img)
    assert detected in ("small", "medium", "large", "xlarge"), (
        f"Unexpected size bucket: {detected}"
    )


def test_evaluate_mc_returns_valid_answer():
    img = _make_colored_text_image((220, 50, 50))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        evaluator = CVBaselineEvaluator()
        result = evaluator.evaluate_mc(
            f.name,
            "What is the color of the text in this image?",
            ["red", "blue", "green", "black"],
        )
        assert "parsed_answer" in result
        assert "response" in result
        assert result["parsed_answer"] in ["red", "blue", "green", "black"]


def test_evaluate_mc_font_family_returns_option():
    img = _make_colored_text_image((0, 0, 0))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        evaluator = CVBaselineEvaluator()
        options = ["Arial", "Helvetica", "Times New Roman", "Georgia"]
        result = evaluator.evaluate_mc(
            f.name,
            "What is the font family of the text in this image?",
            options,
        )
        assert result["parsed_answer"] in options
