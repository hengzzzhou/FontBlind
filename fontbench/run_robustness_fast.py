"""Fast robustness experiments using concurrent API calls.

Runs 8 degradation transforms for GPT-5.2 and Gemini-3-Flash concurrently.
Merges results into existing transform_results.json.
"""
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from fontbench.config import MODELS, SYNTHETIC_DIR, RESULTS_DIR, API_BASE_URL, API_KEY
from fontbench.evaluator import VLMEvaluator
from fontbench.scoring import score_mc_results
from fontbench.transforms import (
    GaussianNoise, GaussianBlur, JPEGCompression, Rotation,
    apply_transform_to_dataset,
)

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
RESULTS_PATH = RESULTS_DIR / "transform_results.json"
MAX_WORKERS = 8  # concurrent API calls


def load_benchmark(data_dir):
    with open(data_dir / "metadata.json") as f:
        return json.load(f)


def eval_single_sample(model_cfg, sample, data_dir):
    """Evaluate all 4 MC questions for one sample. Returns list of result dicts."""
    evaluator = VLMEvaluator(model_id=model_cfg["id"], model_name=model_cfg["name"])
    image_path = str(data_dir / sample["image_path"])
    results = []
    for q in sample["mc_questions"]:
        result = evaluator.evaluate_mc(image_path, q["question"], q["options"])
        results.append({
            "sample_id": sample["id"],
            "property": q["property"],
            "answer": q["answer"],
            "parsed_answer": result["parsed_answer"],
            "response": result["response"],
            "difficulty": sample["metadata"]["difficulty"],
            "script": sample["metadata"]["script"],
            "source": sample["source"],
        })
    return results


def run_concurrent_eval(model_cfg, samples, data_dir):
    """Evaluate a model on all samples using thread pool."""
    all_results = []
    total = len(samples)
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(eval_single_sample, model_cfg, sample, data_dir): sample["id"]
            for sample in samples
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"    Error on sample {sample_id}: {e}")
            done += 1
            if done % 50 == 0 or done == total:
                print(f"    {done}/{total} samples done")

    return all_results


def save_results(all_results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def main():
    # Load existing results
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Flatten all (category, transform) pairs to run
    jobs = []
    for category, transforms in TRANSFORMS_TO_RUN.items():
        if category not in all_results:
            all_results[category] = {}
        for transform in transforms:
            if transform.name not in all_results[category]:
                all_results[category][transform.name] = {}
            for model_cfg in TARGET_MODELS:
                if model_cfg["name"] not in all_results[category][transform.name]:
                    jobs.append((category, transform, model_cfg))

    if not jobs:
        print("All experiments already complete!")
        return

    print(f"Total jobs to run: {len(jobs)}")
    print(f"Concurrent API workers: {MAX_WORKERS}")

    for i, (category, transform, model_cfg) in enumerate(jobs):
        model_name = model_cfg["name"]
        print(f"\n[{i+1}/{len(jobs)}] {model_name} on {transform.name}")

        # Prepare transformed data
        t_dir = TRANSFORMS_DIR / transform.name
        if not (t_dir / "metadata.json").exists():
            print(f"  Applying transform: {transform.name}")
            apply_transform_to_dataset(SYNTHETIC_DIR, t_dir, transform)
        samples = load_benchmark(t_dir)

        # Run concurrent evaluation
        mc_results = run_concurrent_eval(model_cfg, samples, t_dir)
        mc_scores = score_mc_results(mc_results)

        all_results[category][transform.name][model_name] = {
            "overall_accuracy": mc_scores["overall_accuracy"],
            "per_property": mc_scores["per_property"],
        }

        print(f"  Overall: {mc_scores['overall_accuracy']:.3f}")
        for prop, acc in mc_scores["per_property"].items():
            print(f"    {prop}: {acc:.3f}")

        # Save incrementally
        save_results(all_results)
        print(f"  Saved.")

    print(f"\nAll done! Results at {RESULTS_PATH}")


if __name__ == "__main__":
    main()
