"""Evaluate fine-tuned models on FontBench (open-ended) and FRB.

Fine-tuned models are served via vLLM on the remote server.
This script connects to them and runs evaluation.

Usage:
    # Evaluate on FontBench (open-ended format, matching paper Table 5):
    python -m fontbench.finetuning.eval_finetuned \
        --api-base http://101.126.156.90:8001/v1 \
        --model-id font-lora \
        --model-name "Qwen2.5-VL-7B+LoRA" \
        --task fontbench

    # Evaluate on FRB:
    python -m fontbench.finetuning.eval_finetuned \
        --api-base http://101.126.156.90:8001/v1 \
        --model-id font-lora \
        --model-name "Qwen2.5-VL-7B+LoRA" \
        --task frb

    # Evaluate on both:
    python -m fontbench.finetuning.eval_finetuned \
        --api-base http://101.126.156.90:8001/v1 \
        --model-id font-lora \
        --model-name "Qwen2.5-VL-7B+LoRA" \
        --task both
"""
import argparse
import json
import base64
import time
from pathlib import Path
from tqdm import tqdm
import openai

from fontbench.config import SYNTHETIC_DIR, RESULTS_DIR


def _encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def eval_fontbench_open_ended(client, model_id, model_name, data_dir, output_dir):
    """Evaluate on FontBench using open-ended question format.

    This matches the fine-tuning evaluation format in the paper (Table 5),
    where models answer with free-text descriptions of font properties.
    """
    with open(data_dir / "metadata.json") as f:
        samples = json.load(f)

    question = (
        "Describe all font properties of the text in this image, "
        "including font family, size, style, and color."
    )

    results = []
    for sample in tqdm(samples, desc=f"FontBench eval ({model_name})"):
        image_path = str(data_dir / sample["image_path"])
        b64_image = _encode_image(image_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                {"type": "text", "text": question},
            ],
        }]

        response_text = ""
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_id, messages=messages,
                    max_tokens=256, temperature=0.0,
                )
                response_text = resp.choices[0].message.content
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    response_text = f"ERROR: {e}"

        # Extract properties from response
        gt = sample["metadata"]
        extracted = _extract_properties(response_text)

        results.append({
            "sample_id": sample["id"],
            "response": response_text,
            "ground_truth": {
                "font_family": gt["font_family"],
                "font_size": gt["font_size_bucket"],
                "font_style": gt["font_style"],
                "font_color": gt["font_color"],
            },
            "extracted": extracted,
        })

    # Score
    from fontbench.scoring import score_open_ended_results
    scores = score_open_ended_results(results)

    output = {
        "model_name": model_name,
        "task": "fontbench_open_ended",
        "scores": scores,
        "results": results,
    }

    output_file = output_dir / f"finetuned_{model_name.replace('/', '_').replace('+', '_')}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{model_name} FontBench open-ended results:")
    print(f"  Overall: {scores['overall_score']:.3f}")
    for prop, score in scores["per_property_f1"].items():
        print(f"  {prop}: {score:.3f}")
    print(f"  Saved to {output_file}")

    return scores


def _extract_properties(response):
    """Extract font properties from free-text response."""
    response_lower = response.lower()
    extracted = {}

    # Font family: look for "font family: X" or "family: X"
    for line in response.split("\n"):
        ll = line.lower().strip("- ")
        if "family" in ll and ":" in ll:
            extracted["font_family"] = ll.split(":", 1)[1].strip()
        elif "size" in ll and ":" in ll:
            extracted["font_size"] = ll.split(":", 1)[1].strip()
        elif "style" in ll and ":" in ll:
            extracted["font_style"] = ll.split(":", 1)[1].strip()
        elif "color" in ll and ":" in ll:
            extracted["font_color"] = ll.split(":", 1)[1].strip()

    return extracted


def eval_frb(client, model_id, model_name, output_dir):
    """Evaluate on FRB cross-benchmark."""
    from fontbench.frb_eval import FRB_DATA_DIR, FRB_FONTS, _evaluate_mc_n_way

    meta_path = FRB_DATA_DIR / "metadata.json"
    if not meta_path.exists():
        print("FRB dataset not found. Generate it first with: python -m fontbench.frb_eval --generate-only")
        return None

    with open(meta_path) as f:
        samples = json.load(f)

    results = []
    for sample in tqdm(samples, desc=f"FRB eval ({model_name})"):
        image_path = str(FRB_DATA_DIR / sample["image_path"])
        question = "What font is used to render the text in this image?"
        options = sample["options"]

        result = _evaluate_mc_n_way(client, model_id, image_path, question, options)
        results.append({
            "sample_id": sample["id"],
            "font_name": sample["font_name"],
            "difficulty": sample["difficulty"],
            "parsed_answer": result["parsed_answer"],
            "response": result["response"],
            "correct": result["parsed_answer"] == sample["font_name"],
        })

    # Compute scores
    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    easy_results = [r for r in results if r["difficulty"] == "easy"]
    hard_results = [r for r in results if r["difficulty"] == "hard"]

    scores = {
        "overall_accuracy": total_correct / max(total, 1),
        "easy_accuracy": sum(1 for r in easy_results if r["correct"]) / max(len(easy_results), 1),
        "hard_accuracy": sum(1 for r in hard_results if r["correct"]) / max(len(hard_results), 1),
        "total_samples": total,
    }

    # Save results
    frb_results_path = RESULTS_DIR / "frb_results.json"
    existing = {}
    if frb_results_path.exists():
        with open(frb_results_path) as f:
            existing = json.load(f)

    existing[model_name] = {
        "model": {"id": model_id, "name": model_name, "type": "finetuned"},
        "scores": scores,
        "results": results,
    }

    with open(frb_results_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\n{model_name} FRB results:")
    print(f"  Overall: {scores['overall_accuracy']:.3f}")
    print(f"  Easy:    {scores['easy_accuracy']:.3f}")
    print(f"  Hard:    {scores['hard_accuracy']:.3f}")
    print(f"  Merged into {frb_results_path}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument("--api-base", required=True, help="vLLM API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key (default: EMPTY for vLLM)")
    parser.add_argument("--model-id", default="font-lora", help="Model ID on the server")
    parser.add_argument("--model-name", required=True, help="Display name for results")
    parser.add_argument("--task", choices=["fontbench", "frb", "both"], default="both")
    args = parser.parse_args()

    client = openai.OpenAI(base_url=args.api_base, api_key=args.api_key)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.task in ("fontbench", "both"):
        eval_fontbench_open_ended(client, args.model_id, args.model_name, SYNTHETIC_DIR, RESULTS_DIR)

    if args.task in ("frb", "both"):
        eval_frb(client, args.model_id, args.model_name, RESULTS_DIR)
