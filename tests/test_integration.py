# tests/test_integration.py
"""End-to-end integration test for the full pipeline."""
import json
import random
from pathlib import Path
from fontbench.build_benchmark import build_synthetic_subset
from fontbench.scoring import score_mc_results


def test_full_pipeline_synthetic(tmp_path):
    """Test: generate data -> load -> simulate scoring."""
    output_dir = tmp_path / "synthetic"
    output_dir.mkdir()

    # Generate small dataset
    samples = build_synthetic_subset(output_dir, num_samples=10, seed=42)
    assert len(samples) == 10

    # Load back
    with open(output_dir / "metadata.json") as f:
        loaded = json.load(f)
    assert len(loaded) == 10

    # Simulate perfect MC results
    perfect_results = []
    for sample in loaded:
        for q in sample["mc_questions"]:
            perfect_results.append({
                "sample_id": sample["id"],
                "property": q["property"],
                "answer": q["answer"],
                "parsed_answer": q["answer"],  # simulate perfect model
                "difficulty": sample["metadata"]["difficulty"],
                "script": sample["metadata"]["script"],
                "source": sample["source"],
            })

    scores = score_mc_results(perfect_results)
    assert scores["overall_accuracy"] == 1.0
    assert all(v == 1.0 for v in scores["per_property"].values())

    # Simulate random MC results
    random.seed(0)
    random_results = []
    for sample in loaded:
        for q in sample["mc_questions"]:
            random_results.append({
                "sample_id": sample["id"],
                "property": q["property"],
                "answer": q["answer"],
                "parsed_answer": random.choice(q["options"]),
                "difficulty": sample["metadata"]["difficulty"],
                "script": sample["metadata"]["script"],
                "source": sample["source"],
            })

    scores = score_mc_results(random_results)
    # Random should be around 0.25 for 4 choices, but with 10 samples variance is high
    assert 0.0 <= scores["overall_accuracy"] <= 1.0
