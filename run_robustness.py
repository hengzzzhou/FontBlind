"""Run robustness evaluation: mild and severe levels for 4 transform categories."""
import json
from pathlib import Path
from tqdm import tqdm
from fontbench.config import MODELS, SYNTHETIC_DIR, RESULTS_DIR
from fontbench.evaluator import VLMEvaluator
from fontbench.scoring import score_mc_results
from fontbench.transforms import (
    GaussianNoise, GaussianBlur, JPEGCompression, Rotation,
    apply_transform_to_dataset,
)

MODEL_NAMES = ["Qwen3-VL-8B", "Gemini-3-Flash", "GPT-5"]

# Mild and severe only (matching update_paper_tables.py format)
TRANSFORMS = {
    "gaussian_noise": [GaussianNoise(sigma=10), GaussianNoise(sigma=50)],
    "gaussian_blur": [GaussianBlur(radius=1), GaussianBlur(radius=4)],
    "jpeg_compression": [JPEGCompression(quality=75), JPEGCompression(quality=10)],
    "rotation": [Rotation(angle=5), Rotation(angle=45)],
}


def load_benchmark(data_dir):
    with open(data_dir / "metadata.json") as f:
        return json.load(f)


def run_mc_evaluation(evaluator, samples, data_dir):
    results = []
    for sample in tqdm(samples, desc=f"MC eval ({evaluator.model_name})"):
        image_path = str(data_dir / sample["image_path"])
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


def main():
    data_dir = SYNTHETIC_DIR
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "transform_results.json"

    # Load existing results if present
    if results_file.exists():
        with open(results_file) as f:
            transform_results = json.load(f)
    else:
        transform_results = {}

    transforms_dir = data_dir.parent / "transformed"
    transforms_dir.mkdir(parents=True, exist_ok=True)

    models_to_eval = [m for m in MODELS if m["name"] in MODEL_NAMES]

    for category, transforms in TRANSFORMS.items():
        if category not in transform_results:
            transform_results[category] = {}

        for transform in transforms:
            print(f"\n--- Applying transform: {transform.name} ---")
            t_dir = transforms_dir / transform.name
            apply_transform_to_dataset(data_dir, t_dir, transform)
            samples = load_benchmark(t_dir)

            if transform.name not in transform_results[category]:
                transform_results[category][transform.name] = {}

            for model_cfg in models_to_eval:
                if model_cfg["name"] in transform_results[category][transform.name]:
                    print(f"  Skipping {model_cfg['name']} on {transform.name} (already done)")
                    continue

                print(f"  Evaluating {model_cfg['name']} on {transform.name}")
                evaluator = VLMEvaluator(
                    model_id=model_cfg["id"], model_name=model_cfg["name"]
                )
                mc_results = run_mc_evaluation(evaluator, samples, t_dir)
                mc_scores = score_mc_results(mc_results)
                transform_results[category][transform.name][model_cfg["name"]] = {
                    "overall_accuracy": mc_scores["overall_accuracy"],
                    "per_property": mc_scores["per_property"],
                }
                print(f"    MC accuracy: {mc_scores['overall_accuracy']:.3f}")

                # Incremental save
                with open(results_file, "w") as f:
                    json.dump(transform_results, f, indent=2, ensure_ascii=False)
                print(f"    Saved to {results_file}")

    print("\nRobustness evaluation complete!")
    # Print summary
    for category, transforms in TRANSFORMS.items():
        print(f"\n{category}:")
        for transform in transforms:
            print(f"  {transform.name}:")
            for name in MODEL_NAMES:
                acc = transform_results.get(category, {}).get(transform.name, {}).get(name, {}).get("overall_accuracy", 0)
                print(f"    {name}: {acc*100:.1f}%")


if __name__ == "__main__":
    main()
