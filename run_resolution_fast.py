"""Run resolution ablation efficiently: skip 1.0x (use main eval), process fast levels first."""
import json
from pathlib import Path
from tqdm import tqdm
from fontbench.config import MODELS, SYNTHETIC_DIR, RESULTS_DIR
from fontbench.evaluator import VLMEvaluator
from fontbench.scoring import score_mc_results
from fontbench.transforms import Resize, apply_transform_to_dataset

MODEL_NAMES = ["Qwen3-VL-8B", "Gemini-3-Flash", "GPT-5"]

# Process smallest (fastest) first, skip 1.0x (use main eval results), 2.0x last (slowest)
RESOLUTION_SCALES = [0.25, 0.5, 2.0]


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

    if "resolution" not in transform_results:
        transform_results["resolution"] = {}

    # Add 1.0x from main eval results (no API calls needed)
    resize_1x_key = "resize_1.0x"
    if resize_1x_key not in transform_results["resolution"]:
        transform_results["resolution"][resize_1x_key] = {}
    main_accuracy = {"Qwen3-VL-8B": 0.509, "Gemini-3-Flash": 0.667, "GPT-5": 0.535}
    main_per_property = {
        "Qwen3-VL-8B": {"font_family": 0.360, "font_size": 0.392, "font_style": 0.288, "font_color": 0.996},
        "Gemini-3-Flash": {"font_family": 0.808, "font_size": 0.524, "font_style": 0.336, "font_color": 1.0},
        "GPT-5": {"font_family": 0.332, "font_size": 0.540, "font_style": 0.272, "font_color": 0.996},
    }
    for name in MODEL_NAMES:
        transform_results["resolution"][resize_1x_key][name] = {
            "overall_accuracy": main_accuracy[name],
            "per_property": main_per_property[name],
        }

    transforms_dir = data_dir.parent / "transformed"
    transforms_dir.mkdir(parents=True, exist_ok=True)

    models_to_eval = [m for m in MODELS if m["name"] in MODEL_NAMES]

    for scale in RESOLUTION_SCALES:
        transform = Resize(scale=scale)
        print(f"\n--- Applying transform: {transform.name} ---")
        t_dir = transforms_dir / transform.name
        apply_transform_to_dataset(data_dir, t_dir, transform)
        samples = load_benchmark(t_dir)

        if transform.name not in transform_results["resolution"]:
            transform_results["resolution"][transform.name] = {}

        for model_cfg in models_to_eval:
            if model_cfg["name"] in transform_results["resolution"][transform.name]:
                print(f"  Skipping {model_cfg['name']} on {transform.name} (already done)")
                continue

            print(f"  Evaluating {model_cfg['name']} on {transform.name}")
            evaluator = VLMEvaluator(model_id=model_cfg["id"], model_name=model_cfg["name"])
            mc_results = run_mc_evaluation(evaluator, samples, t_dir)
            mc_scores = score_mc_results(mc_results)
            transform_results["resolution"][transform.name][model_cfg["name"]] = {
                "overall_accuracy": mc_scores["overall_accuracy"],
                "per_property": mc_scores["per_property"],
            }
            print(f"    MC accuracy: {mc_scores['overall_accuracy']:.3f}")

            # Incremental save
            with open(results_file, "w") as f:
                json.dump(transform_results, f, indent=2, ensure_ascii=False)
            print(f"    Saved to {results_file}")

    print("\nResolution ablation complete!")
    # Print summary table
    print(f"\n{'Model':<20} {'0.25x':>8} {'0.5x':>8} {'1.0x':>8} {'2.0x':>8}")
    print("-" * 52)
    for name in MODEL_NAMES:
        vals = []
        for key in ["resize_0.25x", "resize_0.5x", "resize_1.0x", "resize_2.0x"]:
            acc = transform_results["resolution"].get(key, {}).get(name, {}).get("overall_accuracy", 0)
            vals.append(f"{acc*100:.1f}%")
        print(f"{name:<20} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8}")


if __name__ == "__main__":
    main()
