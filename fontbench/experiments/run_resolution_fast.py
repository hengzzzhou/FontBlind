"""Run resolution ablation efficiently: skip 1.0x (use main eval), process fast levels first."""
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fontbench.config import MODELS, SYNTHETIC_DIR, RESULTS_DIR
from fontbench.evaluator import VLMEvaluator
from fontbench.scoring import score_mc_results
from fontbench.transforms import Resize, apply_transform_to_dataset

DEFAULT_MODEL_NAMES = ["Qwen3-VL-8B", "Gemini-3-Flash", "GPT-5.2"]

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to evaluate. Defaults to the paper subset.",
    )
    return parser.parse_args()


def load_main_eval_scores(model_name):
    results_path = RESULTS_DIR / f"{model_name.replace('/', '_')}.json"
    if not results_path.exists():
        return None

    with open(results_path) as results_file:
        data = json.load(results_file)

    scores = data["mc"]["scores"]
    return {
        "overall_accuracy": scores["overall_accuracy"],
        "per_property": scores["per_property"],
    }


def main():
    args = parse_args()
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

    model_names = args.models or DEFAULT_MODEL_NAMES
    models_to_eval = [model_cfg for model_cfg in MODELS if model_cfg["name"] in model_names]
    unknown_models = [name for name in model_names if name not in {model_cfg["name"] for model_cfg in MODELS}]
    if unknown_models:
        raise SystemExit(f"Unknown model names: {', '.join(unknown_models)}")

    resize_1x_key = "resize_1.0x"
    if resize_1x_key not in transform_results["resolution"]:
        transform_results["resolution"][resize_1x_key] = {}
    resolution_cache_updated = False
    for name in model_names:
        if name in transform_results["resolution"][resize_1x_key]:
            continue
        main_scores = load_main_eval_scores(name)
        if main_scores is None:
            print(f"  Warning: no base evaluation found for {name}; skipping cached 1.0x entry")
            continue
        transform_results["resolution"][resize_1x_key][name] = main_scores
        resolution_cache_updated = True

    if resolution_cache_updated:
        with open(results_file, "w") as f:
            json.dump(transform_results, f, indent=2, ensure_ascii=False)

    transforms_dir = data_dir.parent / "transformed"
    transforms_dir.mkdir(parents=True, exist_ok=True)

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
    for name in model_names:
        vals = []
        for key in ["resize_0.25x", "resize_0.5x", "resize_1.0x", "resize_2.0x"]:
            acc = transform_results["resolution"].get(key, {}).get(name, {}).get("overall_accuracy", 0)
            vals.append(f"{acc*100:.1f}%")
        print(f"{name:<20} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8}")


if __name__ == "__main__":
    main()
