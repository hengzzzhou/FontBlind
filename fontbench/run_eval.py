"""Run full FontBench evaluation across all models."""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from fontbench.config import MODELS, SYNTHETIC_DIR, RESULTS_DIR
from fontbench.evaluator import VLMEvaluator
from fontbench.scoring import score_mc_results, score_open_ended_results


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


def run_open_ended_evaluation(evaluator, samples, data_dir):
    results = []
    for sample in tqdm(samples, desc=f"Open-ended eval ({evaluator.model_name})"):
        image_path = str(data_dir / sample["image_path"])
        q = sample["open_ended_question"]
        result = evaluator.evaluate_open_ended(image_path, q["question"])
        results.append({
            "sample_id": sample["id"],
            "response": result["response"],
            "ground_truth": q["ground_truth"],
            "difficulty": sample["metadata"]["difficulty"],
            "script": sample["metadata"]["script"],
            "source": sample["source"],
        })
    return results


def run_transformed_evaluation(models_to_eval, data_dir, output_dir, transform_categories):
    """Run evaluation on transformed versions of the dataset.

    Args:
        models_to_eval: List of model config dicts.
        data_dir: Path to the original benchmark dataset.
        output_dir: Path to save results.
        transform_categories: List of transform category names (e.g., ["gaussian_noise", "resolution"]).
    """
    from fontbench.transforms import ALL_TRANSFORMS, apply_transform_to_dataset

    transform_results = {}
    transforms_dir = data_dir.parent / "transformed"
    transforms_dir.mkdir(parents=True, exist_ok=True)

    for category in transform_categories:
        if category not in ALL_TRANSFORMS:
            print(f"Warning: unknown transform category '{category}', skipping")
            continue

        transform_results[category] = {}

        for transform in ALL_TRANSFORMS[category]:
            print(f"\n--- Applying transform: {transform.name} ---")
            t_dir = transforms_dir / transform.name
            apply_transform_to_dataset(data_dir, t_dir, transform)
            samples = load_benchmark(t_dir)

            transform_results[category][transform.name] = {}

            for model_cfg in models_to_eval:
                print(f"  Evaluating {model_cfg['name']} on {transform.name}")
                evaluator = VLMEvaluator(model_id=model_cfg["id"], model_name=model_cfg["name"])
                mc_results = run_mc_evaluation(evaluator, samples, t_dir)
                mc_scores = score_mc_results(mc_results)
                transform_results[category][transform.name][model_cfg["name"]] = {
                    "overall_accuracy": mc_scores["overall_accuracy"],
                    "per_property": mc_scores["per_property"],
                }
                print(f"    MC accuracy: {mc_scores['overall_accuracy']:.3f}")

    # Save transform results
    with open(output_dir / "transform_results.json", "w") as f:
        json.dump(transform_results, f, indent=2, ensure_ascii=False)
    print(f"\nTransform results saved to {output_dir / 'transform_results.json'}")

    return transform_results


def run_cv_baseline_evaluation(samples, data_dir, output_dir):
    """Run traditional CV baseline evaluation."""
    from fontbench.cv_baseline import CVBaselineEvaluator
    from fontbench.config import CV_BASELINE_MODEL

    print(f"\n--- Evaluating {CV_BASELINE_MODEL['name']} (CV Baseline) ---")
    cv_eval = CVBaselineEvaluator()

    results = []
    for sample in tqdm(samples, desc=f"MC eval ({CV_BASELINE_MODEL['name']})"):
        image_path = str(data_dir / sample["image_path"])
        for q in sample["mc_questions"]:
            result = cv_eval.evaluate_mc(image_path, q["question"], q["options"], metadata=sample["metadata"])
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

    mc_scores = score_mc_results(results)
    model_results = {
        "model": CV_BASELINE_MODEL,
        "mc": {"results": results, "scores": mc_scores},
    }

    print(f"  MC accuracy: {mc_scores['overall_accuracy']:.3f}")
    for prop, acc in mc_scores["per_property"].items():
        print(f"    {prop}: {acc:.3f}")

    model_file = output_dir / f"{CV_BASELINE_MODEL['name']}.json"
    with open(model_file, "w") as f:
        json.dump(model_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {model_file}")

    return CV_BASELINE_MODEL["name"], model_results


def main():
    parser = argparse.ArgumentParser(description="Run FontBench evaluation")
    parser.add_argument("--data-dir", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models", nargs="*", help="Model names to evaluate (default: all)")
    parser.add_argument("--task", choices=["mc", "open_ended", "both"], default="both")
    parser.add_argument("--transforms", nargs="*",
                        help="Transform categories to evaluate (e.g., gaussian_noise resolution)")
    parser.add_argument("--include-cv-baseline", action="store_true",
                        help="Include traditional CV baseline in evaluation")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_benchmark(args.data_dir)
    print(f"Loaded {len(samples)} samples")

    models_to_eval = MODELS
    if args.models:
        models_to_eval = [m for m in MODELS if m["name"] in args.models]

    all_results = {}

    # Run CV baseline if requested
    if args.include_cv_baseline:
        cv_name, cv_results = run_cv_baseline_evaluation(samples, args.data_dir, args.output_dir)
        all_results[cv_name] = cv_results

    # Run VLM evaluations
    for model_cfg in models_to_eval:
        print(f"\n--- Evaluating {model_cfg['name']} ---")
        evaluator = VLMEvaluator(model_id=model_cfg["id"], model_name=model_cfg["name"])

        model_results = {"model": model_cfg}

        if args.task in ("mc", "both"):
            mc_results = run_mc_evaluation(evaluator, samples, args.data_dir)
            mc_scores = score_mc_results(mc_results)
            model_results["mc"] = {"results": mc_results, "scores": mc_scores}
            print(f"  MC accuracy: {mc_scores['overall_accuracy']:.3f}")
            for prop, acc in mc_scores["per_property"].items():
                print(f"    {prop}: {acc:.3f}")

        if args.task in ("open_ended", "both"):
            oe_results = run_open_ended_evaluation(evaluator, samples, args.data_dir)
            model_results["open_ended"] = {"results": oe_results}
            print(f"  Open-ended: {len(oe_results)} responses collected")

        all_results[model_cfg["name"]] = model_results

        # Save per-model results
        model_file = args.output_dir / f"{model_cfg['name'].replace('/', '_')}.json"
        with open(model_file, "w") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {model_file}")

    # Save summary
    summary = {}
    for name, r in all_results.items():
        model_info = r.get("model", {})
        s = {"model": model_info.get("name", name)}
        if "mc" in r:
            s["mc_accuracy"] = r["mc"]["scores"]["overall_accuracy"]
            s["mc_per_property"] = r["mc"]["scores"]["per_property"]
            s["mc_by_difficulty"] = r["mc"]["scores"]["by_difficulty"]
            s["mc_by_script"] = r["mc"]["scores"]["by_script"]
        summary[name] = s

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {args.output_dir / 'summary.json'}")

    # Run transform evaluations if requested
    if args.transforms:
        run_transformed_evaluation(models_to_eval, args.data_dir, args.output_dir, args.transforms)


if __name__ == "__main__":
    main()
