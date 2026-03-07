"""Run resolution evaluation for a single model at a single scale."""
import sys
import json
import fcntl
from pathlib import Path
from tqdm import tqdm
from fontbench.config import MODELS, SYNTHETIC_DIR, RESULTS_DIR
from fontbench.evaluator import VLMEvaluator
from fontbench.scoring import score_mc_results
from fontbench.transforms import Resize, apply_transform_to_dataset


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


def atomic_save(results_file, category, transform_name, model_name, data):
    """Thread-safe save using file locking."""
    with open(results_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        transform_results = json.load(f)
        if category not in transform_results:
            transform_results[category] = {}
        if transform_name not in transform_results[category]:
            transform_results[category][transform_name] = {}
        transform_results[category][transform_name][model_name] = data
        f.seek(0)
        f.truncate()
        json.dump(transform_results, f, indent=2, ensure_ascii=False)
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model_name> <scale>")
        print(f"  e.g.: {sys.argv[0]} GPT-5 0.5")
        sys.exit(1)

    model_name = sys.argv[1]
    scale = float(sys.argv[2])

    data_dir = SYNTHETIC_DIR
    results_file = RESULTS_DIR / "transform_results.json"

    # Check if already done
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)
        transform_name = f"resize_{scale}x"
        if "resolution" in existing and transform_name in existing["resolution"]:
            if model_name in existing["resolution"][transform_name]:
                acc = existing["resolution"][transform_name][model_name]["overall_accuracy"]
                print(f"Already done: {model_name} @ {scale}x = {acc*100:.1f}%")
                return

    # Find model config
    model_cfg = next((m for m in MODELS if m["name"] == model_name), None)
    if not model_cfg:
        print(f"Model {model_name} not found in config")
        sys.exit(1)

    # Prepare transformed images
    transform = Resize(scale=scale)
    transforms_dir = data_dir.parent / "transformed"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    t_dir = transforms_dir / transform.name

    print(f"=== {model_name} @ {transform.name} ===")
    apply_transform_to_dataset(data_dir, t_dir, transform)
    samples = load_benchmark(t_dir)

    # Run evaluation
    evaluator = VLMEvaluator(model_id=model_cfg["id"], model_name=model_cfg["name"])
    mc_results = run_mc_evaluation(evaluator, samples, t_dir)
    mc_scores = score_mc_results(mc_results)

    print(f"  MC accuracy: {mc_scores['overall_accuracy']:.3f}")
    print(f"  Per-property: {mc_scores['per_property']}")

    # Atomic save
    atomic_save(results_file, "resolution", transform.name, model_name, {
        "overall_accuracy": mc_scores["overall_accuracy"],
        "per_property": mc_scores["per_property"],
    })
    print(f"  Saved to {results_file}")


if __name__ == "__main__":
    main()
