"""Run FontBench MC evaluation for new models that don't have results yet."""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fontbench.config import MODELS, SYNTHETIC_DIR, RESULTS_DIR
from fontbench.evaluator import VLMEvaluator
from fontbench.scoring import score_mc_results
from fontbench.run_eval import load_benchmark, run_mc_evaluation

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Determine which models still need evaluation
existing = set()
for f in RESULTS_DIR.glob("*.json"):
    if f.stem not in ("summary", "frb_results", "transform_results", "leaderboard"):
        existing.add(f.stem)

print(f"Existing results: {sorted(existing)}")

# Find models that need evaluation
models_to_run = []
for m in MODELS:
    name = m["name"]
    safe_name = name.replace("/", "_")
    if safe_name not in existing and name not in existing:
        models_to_run.append(m)

# Filter to specific model if passed as argument
if len(sys.argv) > 1:
    target = sys.argv[1]
    models_to_run = [m for m in models_to_run if m["name"] == target]

if not models_to_run:
    print("All models already evaluated!")
    sys.exit(0)

print(f"\nModels to evaluate ({len(models_to_run)}):")
for m in models_to_run:
    print(f"  {m['name']} ({m['id']})")

# Load benchmark
samples = load_benchmark(SYNTHETIC_DIR)
print(f"\nLoaded {len(samples)} samples")

# Run evaluations
summary_path = RESULTS_DIR / "summary.json"
summary = json.load(open(summary_path)) if summary_path.exists() else {}

for model_cfg in models_to_run:
    name = model_cfg["name"]
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    evaluator = VLMEvaluator(model_id=model_cfg["id"], model_name=name)
    mc_results = run_mc_evaluation(evaluator, samples, SYNTHETIC_DIR)
    mc_scores = score_mc_results(mc_results)

    model_results = {
        "model": model_cfg,
        "mc": {"results": mc_results, "scores": mc_scores},
    }

    # Save per-model results
    model_file = RESULTS_DIR / f"{name.replace('/', '_')}.json"
    with open(model_file, "w") as f:
        json.dump(model_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {model_file}")

    # Print scores
    print(f"  MC accuracy: {mc_scores['overall_accuracy']:.3f}")
    for prop, acc in mc_scores["per_property"].items():
        print(f"    {prop}: {acc:.3f}")

    # Update summary
    summary[name] = {
        "model": name,
        "mc_accuracy": mc_scores["overall_accuracy"],
        "mc_per_property": mc_scores["per_property"],
        "mc_by_difficulty": mc_scores["by_difficulty"],
        "mc_by_script": mc_scores["by_script"],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary updated")

print("\n\nAll evaluations complete!")
print(f"Summary: {summary_path}")
