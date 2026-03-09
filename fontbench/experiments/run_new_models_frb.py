"""Run FRB evaluation for new models that don't have FRB results yet."""
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fontbench.config import MODELS, RESULTS_DIR
from fontbench.frb_eval import FRB_DATA_DIR, run_frb_evaluation


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frb_path = RESULTS_DIR / "frb_results.json"
    existing_frb = {}
    if frb_path.exists():
        with open(frb_path) as frb_file:
            existing_frb = json.load(frb_file)

    print(f"Existing FRB results: {sorted(existing_frb.keys())}")

    models_to_run = [model_cfg for model_cfg in MODELS if model_cfg["name"] not in existing_frb]

    if len(sys.argv) > 1:
        target = sys.argv[1]
        models_to_run = [model_cfg for model_cfg in models_to_run if model_cfg["name"] == target]

    if not models_to_run:
        print("All models already have FRB results!")
        sys.exit(0)

    print(f"\nModels to evaluate ({len(models_to_run)}):")
    for model_cfg in models_to_run:
        print(f"  {model_cfg['name']} ({model_cfg['id']})")

    new_results = run_frb_evaluation(models_to_eval=models_to_run)

    existing_frb.update(new_results)
    with open(frb_path, "w") as frb_output:
        json.dump(existing_frb, frb_output, indent=2, ensure_ascii=False)

    print(f"\nMerged FRB results saved to {frb_path}")
    print(f"Total models with FRB results: {len(existing_frb)}")


if __name__ == "__main__":
    main()
