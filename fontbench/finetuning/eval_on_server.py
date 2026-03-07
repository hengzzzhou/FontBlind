"""Self-contained evaluation of fine-tuned LoRA models on the GPU server.

Runs FontBench MC (4-way) and FRB (15-way) evaluations using transformers + peft
directly, without requiring vLLM.

Usage:
    python eval_on_server.py \
        --model-key qwen2.5-vl-7b \
        --base-model /fs-computility-new/Uma4agi/shared/models/Qwen2.5-VL-7B-Instruct \
        --lora-path /root/fontvlm/checkpoints/qwen2.5-vl-7b_lora \
        --data-dir /root/fontvlm/eval_data \
        --output-dir /root/fontvlm/eval_results
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


# Model configurations
MODEL_CONFIGS = {
    "qwen2.5-vl-7b": {
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "display_name": "Qwen2.5-VL-7B+LoRA",
    },
    "qwen3-vl-8b": {
        "model_class": "Qwen3VLForConditionalGeneration",
        "display_name": "Qwen3-VL-8B+LoRA",
    },
    "qwen2.5-vl-32b": {
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "display_name": "Qwen2.5-VL-32B+LoRA",
    },
}


def load_model(model_key, base_model_path, lora_path, use_4bit=False):
    """Load base model + LoRA adapter."""
    from transformers import AutoProcessor
    from peft import PeftModel

    config = MODEL_CONFIGS[model_key]
    print(f"Loading model: {model_key}")
    print(f"  Base: {base_model_path}")
    print(f"  LoRA: {lora_path}")
    print(f"  4-bit: {use_4bit}")

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # Import the right model class
    if config["model_class"] == "Qwen2_5_VLForConditionalGeneration":
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "auto"

    model = ModelClass.from_pretrained(base_model_path, **load_kwargs)
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    print(f"  Model loaded on {model.device}")
    return model, processor


def run_inference(model, processor, image_path, prompt, model_key):
    """Run single inference with the model."""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # For Qwen3-VL, disable thinking mode for faster/cleaner MC answers
    if "qwen3" in model_key:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(model.device)

    gen_kwargs = {"max_new_tokens": 64, "do_sample": False}
    if "qwen3" in model_key:
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None

    output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    response = processor.decode(generated, skip_special_tokens=True)
    return response.strip()


def parse_mc_answer(response, options, num_options=4):
    """Parse MC answer letter from response."""
    valid_letters = [chr(65 + i) for i in range(num_options)]  # A, B, C, D, ...

    # For Qwen3 thinking mode, extract from </think> block
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    for char in response.strip().upper():
        if char in valid_letters:
            idx = ord(char) - 65
            if idx < len(options):
                return options[idx]
            break
    return None


def eval_fontbench(model, processor, model_key, data_dir, output_dir):
    """Evaluate on FontBench MC (4-way)."""
    meta_path = data_dir / "synthetic" / "metadata.json"
    with open(meta_path) as f:
        samples = json.load(f)

    results = []
    for sample in tqdm(samples, desc="FontBench MC"):
        image_path = str(data_dir / "synthetic" / sample["image_path"])

        for q in sample["mc_questions"]:
            options = q["options"]
            options_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
            prompt = f"{q['question']}\n\n{options_str}\n\nAnswer with only the letter (A, B, C, or D)."

            try:
                response = run_inference(model, processor, image_path, prompt, model_key)
            except Exception as e:
                response = f"ERROR: {e}"

            parsed = parse_mc_answer(response, options, num_options=4)

            results.append({
                "sample_id": sample["id"],
                "property": q["property"],
                "answer": q["answer"],
                "parsed_answer": parsed,
                "response": response,
                "difficulty": sample["metadata"].get("difficulty", ""),
                "script": sample["metadata"].get("script", ""),
                "source": "synthetic",
            })

    # Score
    scores = score_mc_results(results)

    display_name = MODEL_CONFIGS[model_key]["display_name"]
    output = {
        "model": {"id": model_key, "name": display_name, "type": "finetuned"},
        "mc": {"results": results, "scores": scores},
    }

    output_file = output_dir / f"{display_name.replace('+', '_').replace('/', '_')}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFontBench MC Results ({display_name}):")
    print(f"  Overall: {scores['overall_accuracy']*100:.1f}%")
    for prop, acc in scores.get("per_property", {}).items():
        print(f"  {prop}: {acc*100:.1f}%")
    print(f"  Saved to {output_file}")

    return scores


def eval_frb(model, processor, model_key, data_dir, output_dir):
    """Evaluate on FRB (15-way MC)."""
    meta_path = data_dir / "frb" / "metadata.json"
    with open(meta_path) as f:
        samples = json.load(f)

    results = []
    for sample in tqdm(samples, desc="FRB"):
        image_path = str(data_dir / "frb" / sample["image_path"])
        options = sample["options"]

        options_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
        prompt = f"What font is used to render the text in this image?\n\n{options_str}\n\nAnswer with only the letter."

        try:
            response = run_inference(model, processor, image_path, prompt, model_key)
        except Exception as e:
            response = f"ERROR: {e}"

        parsed = parse_mc_answer(response, options, num_options=15)

        results.append({
            "sample_id": sample["id"],
            "font_name": sample["font_name"],
            "difficulty": sample["difficulty"],
            "parsed_answer": parsed,
            "response": response,
            "correct": parsed == sample["font_name"],
        })

    # Score
    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    easy = [r for r in results if r["difficulty"] == "easy"]
    hard = [r for r in results if r["difficulty"] == "hard"]

    scores = {
        "overall_accuracy": total_correct / max(total, 1),
        "easy_accuracy": sum(1 for r in easy if r["correct"]) / max(len(easy), 1),
        "hard_accuracy": sum(1 for r in hard if r["correct"]) / max(len(hard), 1),
        "total_samples": total,
    }

    # Save
    display_name = MODEL_CONFIGS[model_key]["display_name"]
    frb_file = output_dir / "frb_finetuned_results.json"
    existing = {}
    if frb_file.exists():
        with open(frb_file) as f:
            existing = json.load(f)

    existing[display_name] = {
        "model": {"id": model_key, "name": display_name, "type": "finetuned"},
        "scores": scores,
        "results": results,
    }

    with open(frb_file, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\nFRB Results ({display_name}):")
    print(f"  Overall: {scores['overall_accuracy']*100:.1f}%")
    print(f"  Easy:    {scores['easy_accuracy']*100:.1f}%")
    print(f"  Hard:    {scores['hard_accuracy']*100:.1f}%")
    print(f"  Saved to {frb_file}")

    return scores


def score_mc_results(results):
    """Score MC results."""
    from collections import defaultdict

    correct = sum(1 for r in results if r["parsed_answer"] == r["answer"])
    total = len(results)

    per_property = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        prop = r["property"]
        per_property[prop]["total"] += 1
        if r["parsed_answer"] == r["answer"]:
            per_property[prop]["correct"] += 1

    per_property_acc = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for k, v in per_property.items()
    }

    by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r.get("difficulty"):
            d = r["difficulty"]
            by_difficulty[d]["total"] += 1
            if r["parsed_answer"] == r["answer"]:
                by_difficulty[d]["correct"] += 1

    by_difficulty_acc = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for k, v in by_difficulty.items()
    }

    return {
        "overall_accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "per_property": dict(per_property_acc),
        "by_difficulty": dict(by_difficulty_acc),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--base-model", required=True, help="Path to base model")
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--data-dir", default="/root/fontvlm/eval_data")
    parser.add_argument("--output-dir", default="/root/fontvlm/eval_results")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--task", choices=["fontbench", "frb", "both"], default="both")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model_key, args.base_model, args.lora_path, args.use_4bit)

    if args.task in ("fontbench", "both"):
        eval_fontbench(model, processor, args.model_key, data_dir, output_dir)

    if args.task in ("frb", "both"):
        eval_frb(model, processor, args.model_key, data_dir, output_dir)

    print("\nAll evaluations complete!")
