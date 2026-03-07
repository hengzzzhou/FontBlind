"""Extract attention heatmaps from Qwen2.5-VL-7B on hard-difficulty FontBench samples.

Two phases:
  Phase 1 — Evaluate all hard-difficulty samples (85 images × 4 questions = 340 inferences)
  Phase 2 — Extract attention maps for 6 selected failure cases (2 per property)

Usage (on GPU server):
    # Phase 1: evaluate hard samples
    python extract_attention.py \
        --model-path /path/to/Qwen2.5-VL-7B-Instruct \
        --data-dir /root/fontvlm/eval_data \
        --output-dir ./attention_output \
        --phase eval

    # Phase 2: extract attention (after reviewing phase 1 results)
    python extract_attention.py \
        --model-path /path/to/Qwen2.5-VL-7B-Instruct \
        --data-dir /root/fontvlm/eval_data \
        --output-dir ./attention_output \
        --phase extract

    # Both phases in one run (auto-selects failures)
    python extract_attention.py \
        --model-path /path/to/Qwen2.5-VL-7B-Instruct \
        --data-dir /root/fontvlm/eval_data \
        --output-dir ./attention_output \
        --phase both
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_model(model_path):
    """Load Qwen2.5-VL-7B with eager attention for weight extraction."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")
    return model, processor


def build_prompt(question, options):
    """Build MC prompt from question and options."""
    options_str = "\n".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options))
    return f"{question}\n\n{options_str}\n\nAnswer with only the letter (A, B, C, or D)."


def parse_mc_answer(response, options):
    """Parse MC answer letter from response."""
    valid_letters = [chr(65 + i) for i in range(len(options))]
    for char in response.strip().upper():
        if char in valid_letters:
            idx = ord(char) - 65
            if idx < len(options):
                return options[idx]
            break
    return None


def run_inference(model, processor, image_path, prompt):
    """Run single inference, return response text."""
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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(
        **inputs, max_new_tokens=64, do_sample=False
    )
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Phase 1: Evaluate hard samples
# ---------------------------------------------------------------------------
def phase_eval(model, processor, data_dir, output_dir):
    """Evaluate all hard-difficulty samples and save results."""
    meta_path = data_dir / "synthetic" / "metadata.json"
    with open(meta_path) as f:
        all_samples = json.load(f)

    hard_samples = [s for s in all_samples if s["metadata"].get("difficulty") == "hard"]
    print(f"Found {len(hard_samples)} hard-difficulty samples")

    results = []
    for sample in tqdm(hard_samples, desc="Evaluating hard samples"):
        image_path = str(data_dir / "synthetic" / sample["image_path"])
        for q in sample["mc_questions"]:
            prompt = build_prompt(q["question"], q["options"])
            try:
                response = run_inference(model, processor, image_path, prompt)
            except Exception as e:
                response = f"ERROR: {e}"

            parsed = parse_mc_answer(response, q["options"])
            results.append({
                "sample_id": sample["id"],
                "property": q["property"],
                "answer": q["answer"],
                "parsed_answer": parsed,
                "correct": parsed == q["answer"],
                "response": response,
                "script": sample["metadata"].get("script", ""),
                "font_family": sample["metadata"]["font_family"],
                "font_size": sample["metadata"]["font_size"],
                "font_style": sample["metadata"]["font_style"],
                "image_path": sample["image_path"],
            })

    # Save results
    results_path = output_dir / "attention_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(f"\nPhase 1 complete: {correct}/{total} correct ({correct/total*100:.1f}%)")

    by_prop = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_prop[r["property"]]["total"] += 1
        if r["correct"]:
            by_prop[r["property"]]["correct"] += 1
    for prop, counts in sorted(by_prop.items()):
        acc = counts["correct"] / counts["total"] * 100
        failures = counts["total"] - counts["correct"]
        print(f"  {prop}: {acc:.1f}% ({failures} failures)")

    print(f"  Saved to {results_path}")
    return results


# ---------------------------------------------------------------------------
# Image selection: pick 2 diverse failures per property
# ---------------------------------------------------------------------------
def select_failures(results, target_properties=("font_family", "font_size", "font_style")):
    """Select 6 failure cases: 2 per property, maximizing diversity."""
    selected = []
    used_sample_ids = set()

    for prop in target_properties:
        # Get all failures for this property
        failures = [r for r in results if r["property"] == prop and not r["correct"]]
        if len(failures) < 2:
            print(f"WARNING: Only {len(failures)} failures for {prop}")
            selected.extend(failures)
            for f in failures:
                used_sample_ids.add(f["sample_id"])
            continue

        # Sort by script diversity: prefer picking from different scripts
        by_script = defaultdict(list)
        for f in failures:
            by_script[f["script"]].append(f)

        picked = []
        scripts = list(by_script.keys())

        # First pick: prefer a sample not yet used, from any script
        for script in scripts:
            candidates = [f for f in by_script[script] if f["sample_id"] not in used_sample_ids]
            if candidates:
                picked.append(candidates[0])
                used_sample_ids.add(candidates[0]["sample_id"])
                break
        if not picked:
            picked.append(failures[0])
            used_sample_ids.add(failures[0]["sample_id"])

        # Second pick: prefer different script + different sample
        for script in scripts:
            if script == picked[0]["script"] and len(scripts) > 1:
                continue
            candidates = [f for f in by_script[script] if f["sample_id"] not in used_sample_ids]
            if candidates:
                picked.append(candidates[0])
                used_sample_ids.add(candidates[0]["sample_id"])
                break
        if len(picked) < 2:
            # Fall back: any unused sample
            for f in failures:
                if f["sample_id"] not in used_sample_ids:
                    picked.append(f)
                    used_sample_ids.add(f["sample_id"])
                    break
        if len(picked) < 2:
            # Last resort: allow duplicate sample
            picked.append(failures[1] if failures[1]["sample_id"] != picked[0]["sample_id"] else failures[0])

        selected.extend(picked[:2])

    return selected


# ---------------------------------------------------------------------------
# Phase 2: Extract attention maps
# ---------------------------------------------------------------------------
def extract_attention_for_sample(model, processor, image_path, question, options):
    """Extract GradCAM-style attribution from the ViT visual encoder.

    Hooks into the last block of the visual encoder to capture activations
    and gradients. The gradient of the predicted answer token's logit w.r.t.
    the ViT's last-block output gives a spatially-resolved importance map
    at the ViT's native patch resolution (before 2×2 merging), which is
    4× finer than the language-model image token grid.
    """
    torch.backends.cudnn.enabled = False

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt(question, options)},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(model.device)

    input_ids = inputs["input_ids"][0]
    seq_len = len(input_ids)

    # Get spatial grid info before forward pass
    image_grid_thw = inputs.get("image_grid_thw", None)
    if image_grid_thw is not None:
        t, grid_h, grid_w = image_grid_thw[0].tolist()
    else:
        raise ValueError("image_grid_thw not found in inputs")

    print(f"    seq_len: {seq_len}, ViT grid: ({grid_h}, {grid_w})")

    # --- Hook into ViT last block ---
    vit = model.model.visual
    last_block = vit.blocks[-1]
    captured = {}

    def fwd_hook(module, input, output):
        # output: [num_patches, hidden_dim]
        captured["activation"] = output
        output.retain_grad()

    handle = last_block.register_forward_hook(fwd_hook)

    # Forward pass WITH gradients enabled
    outputs = model(**inputs)

    # Get logits for the last token (the one about to generate the answer)
    logits = outputs.logits[0, -1, :]  # [vocab_size]
    pred_token_id = logits.argmax()

    # Backward pass: gradient of predicted token's logit w.r.t. ViT activation
    logits[pred_token_id].backward(retain_graph=False)

    handle.remove()

    activation = captured["activation"].detach()  # [num_patches, hidden_dim]
    grad = captured["activation"].grad.detach()    # [num_patches, hidden_dim]

    # GradCAM: weight each channel by its gradient, then sum
    # weights: [hidden_dim] — global importance of each channel
    weights = grad.mean(dim=0)  # average gradient over spatial positions
    # cam: [num_patches]
    cam = (activation * weights).sum(dim=-1)
    cam = torch.relu(cam)  # only positive contributions

    cam_np = cam.float().cpu().numpy()
    num_patches = grid_h * grid_w

    print(f"    ViT patches: {len(cam_np)}, expected: {num_patches}")

    # Handle potential temporal dimension (t frames × grid_h × grid_w)
    if len(cam_np) >= num_patches:
        cam_np = cam_np[:num_patches]  # take first frame if multiple
    else:
        cam_np = np.pad(cam_np, (0, num_patches - len(cam_np)))

    cam_map = cam_np.reshape(int(grid_h), int(grid_w))

    # Clear gradients to free memory
    model.zero_grad()

    return cam_map, image.size  # (width, height)


def phase_extract(model, processor, data_dir, output_dir, selections=None):
    """Extract attention maps for selected failure cases."""
    # Load evaluation results if selections not provided
    if selections is None:
        results_path = output_dir / "attention_eval_results.json"
        if not results_path.exists():
            raise FileNotFoundError(
                f"Run phase 'eval' first. Missing: {results_path}"
            )
        with open(results_path) as f:
            results = json.load(f)
        selections = select_failures(results)

    print(f"\nPhase 2: Extracting attention for {len(selections)} samples")
    for i, sel in enumerate(selections):
        print(f"  [{i+1}] {sel['sample_id']} | {sel['property']} | "
              f"true={sel['answer']} pred={sel['parsed_answer']} | {sel['script']}")

    # Load metadata to get full question info
    meta_path = data_dir / "synthetic" / "metadata.json"
    with open(meta_path) as f:
        all_samples = json.load(f)
    sample_lookup = {s["id"]: s for s in all_samples}

    attention_dir = output_dir / "attention_maps"
    attention_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, sel in enumerate(selections):
        sample_id = sel["sample_id"]
        prop = sel["property"]
        sample = sample_lookup[sample_id]
        image_path = str(data_dir / "synthetic" / sample["image_path"])

        # Find the matching question
        question_data = None
        for q in sample["mc_questions"]:
            if q["property"] == prop:
                question_data = q
                break

        if question_data is None:
            print(f"  WARNING: No question for {prop} in {sample_id}, skipping")
            continue

        print(f"\n  Extracting [{i+1}/{len(selections)}]: {sample_id} / {prop}")
        try:
            attn_map, img_size = extract_attention_for_sample(
                model, processor, image_path,
                question_data["question"], question_data["options"]
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        # Save attention map
        npz_path = attention_dir / f"{sample_id}_{prop}.npz"
        np.savez_compressed(
            npz_path,
            attention_map=attn_map,
            image_width=img_size[0],
            image_height=img_size[1],
            sample_id=sample_id,
            property=prop,
            true_answer=sel["answer"],
            pred_answer=sel["parsed_answer"],
            script=sel["script"],
            font_family=sample["metadata"]["font_family"],
            font_size=sample["metadata"]["font_size"],
            font_style=sample["metadata"]["font_style"],
            image_path=sample["image_path"],
        )
        print(f"    Saved: {npz_path} (shape={attn_map.shape})")

        manifest.append({
            "npz_file": str(npz_path.name),
            "sample_id": sample_id,
            "property": prop,
            "true_answer": str(sel["answer"]),
            "pred_answer": str(sel["parsed_answer"]),
            "script": sel["script"],
            "image_path": sample["image_path"],
            "attn_shape": list(attn_map.shape),
        })

    # Save manifest
    manifest_path = attention_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract attention heatmaps from Qwen2.5-VL-7B")
    parser.add_argument("--model-path", required=True, help="Path to Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data-dir", default="/root/fontvlm/eval_data", help="Path to eval data")
    parser.add_argument("--output-dir", default="./attention_output", help="Output directory")
    parser.add_argument("--phase", choices=["eval", "extract", "both"], default="both",
                        help="Which phase to run")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model_path)

    if args.phase in ("eval", "both"):
        results = phase_eval(model, processor, data_dir, output_dir)

    if args.phase in ("extract", "both"):
        selections = None
        if args.phase == "both":
            selections = select_failures(results)
        phase_extract(model, processor, data_dir, output_dir, selections)

    print("\nDone!")
