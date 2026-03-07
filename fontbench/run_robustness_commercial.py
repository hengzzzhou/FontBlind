"""Run robustness experiments for GPT-5.2 and Gemini-3-Flash.

Evaluates 8 degradation transforms (matching existing Qwen3-VL-8B data)
and merges results into transform_results.json without losing existing data.
"""
import json
from pathlib import Path
from fontbench.config import MODELS, SYNTHETIC_DIR, RESULTS_DIR
from fontbench.evaluator import VLMEvaluator
from fontbench.scoring import score_mc_results
from fontbench.transforms import (
    GaussianNoise, GaussianBlur, JPEGCompression, Rotation,
    apply_transform_to_dataset,
)
from fontbench.run_eval import load_benchmark, run_mc_evaluation

# Models to evaluate
TARGET_MODELS = [m for m in MODELS if m["name"] in ("GPT-5.2", "Gemini-3-Flash")]

# Transforms matching existing Qwen3-VL-8B data
TRANSFORMS_TO_RUN = {
    "gaussian_noise": [GaussianNoise(sigma=10), GaussianNoise(sigma=50)],
    "gaussian_blur": [GaussianBlur(radius=1), GaussianBlur(radius=4)],
    "jpeg_compression": [JPEGCompression(quality=75), JPEGCompression(quality=10)],
    "rotation": [Rotation(angle=5), Rotation(angle=45)],
}

TRANSFORMS_DIR = SYNTHETIC_DIR.parent / "transformed"


def main():
    # Load existing results
    results_path = RESULTS_DIR / "transform_results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    total_runs = sum(len(ts) for ts in TRANSFORMS_TO_RUN.values()) * len(TARGET_MODELS)
    done = 0

    for category, transforms in TRANSFORMS_TO_RUN.items():
        if category not in all_results:
            all_results[category] = {}

        for transform in transforms:
            if transform.name not in all_results[category]:
                all_results[category][transform.name] = {}

            # Apply transform (reuse if already exists)
            t_dir = TRANSFORMS_DIR / transform.name
            if not (t_dir / "metadata.json").exists():
                print(f"\nApplying transform: {transform.name}")
                apply_transform_to_dataset(SYNTHETIC_DIR, t_dir, transform)
            else:
                print(f"\nReusing existing transform: {transform.name}")

            samples = load_benchmark(t_dir)

            for model_cfg in TARGET_MODELS:
                model_name = model_cfg["name"]

                # Skip if already done
                if model_name in all_results[category][transform.name]:
                    done += 1
                    print(f"  [{done}/{total_runs}] {model_name} on {transform.name}: SKIPPED (exists)")
                    continue

                done += 1
                print(f"  [{done}/{total_runs}] Evaluating {model_name} on {transform.name}...")
                evaluator = VLMEvaluator(model_id=model_cfg["id"], model_name=model_name)
                mc_results = run_mc_evaluation(evaluator, samples, t_dir)
                mc_scores = score_mc_results(mc_results)

                all_results[category][transform.name][model_name] = {
                    "overall_accuracy": mc_scores["overall_accuracy"],
                    "per_property": mc_scores["per_property"],
                }
                print(f"    Accuracy: {mc_scores['overall_accuracy']:.3f}")
                for prop, acc in mc_scores["per_property"].items():
                    print(f"      {prop}: {acc:.3f}")

                # Save after each model to avoid losing progress
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll done! Results saved to {results_path}")


if __name__ == "__main__":
    main()
