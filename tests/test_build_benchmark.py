# tests/test_build_benchmark.py
"""Tests for benchmark builder."""
import json
from pathlib import Path
from fontbench.build_benchmark import build_synthetic_subset


def test_build_synthetic_subset(tmp_path):
    output_dir = tmp_path / "synthetic"
    output_dir.mkdir()
    samples = build_synthetic_subset(output_dir, num_samples=5, seed=42)
    assert len(samples) == 5
    # Check files created
    assert (output_dir / "images").exists()
    assert len(list((output_dir / "images").glob("*.png"))) == 5
    # Check metadata
    assert (output_dir / "metadata.json").exists()
    with open(output_dir / "metadata.json") as f:
        data = json.load(f)
    assert len(data) == 5
    # Check each sample has questions
    for sample in data:
        assert "mc_questions" in sample
        assert "open_ended_question" in sample
        assert "image_path" in sample
        assert len(sample["mc_questions"]) == 4
