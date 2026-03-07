"""Run FRB evaluation for new models that don't have FRB results yet."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fontbench.config import MODELS, RESULTS_DIR
from fontbench.frb_eval import FRB_DATA_DIR, run_frb_evaluation

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load existing FRB results
frb_path = RESULTS_DIR / "frb_results.json"
existing_frb = {}
if frb_path.exists():
    with open(frb_path) as f:
        existing_frb = json.load(f)

print(f"Existing FRB results: {sorted(existing_frb.keys())}")

# Find models needing FRB evaluation
models_to_run = [m for m in MODELS if m["name"] not in existing_frb]

# Filter to specific model if passed as argument
if len(sys.argv) > 1:
    target = sys.argv[1]
    models_to_run = [m for m in models_to_run if m["name"] == target]

if not models_to_run:
    print("All models already have FRB results!")
    sys.exit(0)

print(f"\nModels to evaluate ({len(models_to_run)}):")
for m in models_to_run:
    print(f"  {m['name']} ({m['id']})")

# Run FRB evaluation
new_results = run_frb_evaluation(models_to_eval=models_to_run)

# Merge with existing
existing_frb.update(new_results)
with open(frb_path, "w") as f:
    json.dump(existing_frb, f, indent=2, ensure_ascii=False)

print(f"\nMerged FRB results saved to {frb_path}")
print(f"Total models with FRB results: {len(existing_frb)}")
