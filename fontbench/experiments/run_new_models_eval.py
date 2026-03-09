"""Run FontBench MC evaluation for new models that don't have results yet."""
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fontbench.config import MODELS, SYNTHETIC_DIR, RESULTS_DIR
from fontbench.evaluator import VLMEvaluator
from fontbench.scoring import score_mc_results
from fontbench.run_eval import load_benchmark, run_mc_evaluation


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing = set()
    for results_file in RESULTS_DIR.glob("*.json"):
        if results_file.stem not in ("summary", "frb_results", "transform_results", "leaderboard"):
            existing.add(results_file.stem)

    print(f"Existing results: {sorted(existing)}")

    models_to_run = []
    for model_cfg in MODELS:
        name = model_cfg["name"]
        safe_name = name.replace("/", "_")
        if safe_name not in existing and name not in existing:
            models_to_run.append(model_cfg)

    if len(sys.argv) > 1:
        target = sys.argv[1]
        models_to_run = [model_cfg for model_cfg in models_to_run if model_cfg["name"] == target]

    if not models_to_run:
        print("All models already evaluated!")
        sys.exit(0)

    print(f"\nModels to evaluate ({len(models_to_run)}):")
    for model_cfg in models_to_run:
        print(f"  {model_cfg['name']} ({model_cfg['id']})")

    samples = load_benchmark(SYNTHETIC_DIR)
    print(f"\nLoaded {len(samples)} samples")

    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as summary_file:
            summary = json.load(summary_file)
    else:
        summary = {}

    for model_cfg in models_to_run:
        name = model_cfg["name"]
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name}")
        print(f"{'=' * 60}")

        evaluator = VLMEvaluator(model_id=model_cfg["id"], model_name=name)
        mc_results = run_mc_evaluation(evaluator, samples, SYNTHETIC_DIR)
        mc_scores = score_mc_results(mc_results)

        model_results = {
            "model": model_cfg,
            "mc": {"results": mc_results, "scores": mc_scores},
        }

        model_file = RESULTS_DIR / f"{name.replace('/', '_')}.json"
        with open(model_file, "w") as model_output:
            json.dump(model_results, model_output, indent=2, ensure_ascii=False)
        print(f"  Saved to {model_file}")

        print(f"  MC accuracy: {mc_scores['overall_accuracy']:.3f}")
        for prop, acc in mc_scores["per_property"].items():
            print(f"    {prop}: {acc:.3f}")

        summary[name] = {
            "model": name,
            "mc_accuracy": mc_scores["overall_accuracy"],
            "mc_per_property": mc_scores["per_property"],
            "mc_by_difficulty": mc_scores["by_difficulty"],
            "mc_by_script": mc_scores["by_script"],
        }

        with open(summary_path, "w") as summary_output:
            json.dump(summary, summary_output, indent=2, ensure_ascii=False)
        print("  Summary updated")

    print("\n\nAll evaluations complete!")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
