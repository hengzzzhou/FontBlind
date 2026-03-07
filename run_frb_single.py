"""Run FRB evaluation for a single model with atomic save."""
import sys
import json
import fcntl
from pathlib import Path
from fontbench.config import MODELS, RESULTS_DIR
from fontbench.frb_eval import FRB_DATA_DIR, FRB_FONTS, run_frb_evaluation


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    frb_results_path = RESULTS_DIR / "frb_results.json"

    # Check if already done
    if frb_results_path.exists():
        with open(frb_results_path) as f:
            existing = json.load(f)
        if model_name in existing:
            acc = existing[model_name]["scores"]["overall_accuracy"]
            print(f"Already done: {model_name} FRB = {acc*100:.1f}%")
            return

    # Find model config
    model_cfg = next((m for m in MODELS if m["name"] == model_name), None)
    if not model_cfg:
        print(f"Model {model_name} not found in config")
        sys.exit(1)

    # Load FRB samples
    with open(FRB_DATA_DIR / "metadata.json") as f:
        samples = json.load(f)

    print(f"=== FRB eval: {model_name} ({len(samples)} samples) ===")
    new_results = run_frb_evaluation(samples, [model_cfg])

    # Atomic save with file locking
    with open(frb_results_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        existing = json.load(f)
        existing.update(new_results)
        f.seek(0)
        f.truncate()
        json.dump(existing, f, indent=2, ensure_ascii=False)
        fcntl.flock(f, fcntl.LOCK_UN)

    scores = new_results[model_name]["scores"]
    print(f"  Overall: {scores['overall_accuracy']:.3f}")
    print(f"  Easy:    {scores['easy_accuracy']:.3f}")
    print(f"  Hard:    {scores['hard_accuracy']:.3f}")
    print(f"  Saved to {frb_results_path}")


if __name__ == "__main__":
    main()
