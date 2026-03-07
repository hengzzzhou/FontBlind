"""FRB (Font Recognition Benchmark) cross-benchmark evaluation.

Replicates the FRB benchmark format (Li et al., 2025) for cross-benchmark
comparison. Generates images in 15 common Latin fonts and evaluates models
on font family identification via MCQ.

Two difficulty modes:
- easy: render normal English sentences
- hard: render the font name itself as text (stroop effect)
"""
import json
import base64
import time
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import openai
from fontbench.config import MODELS, RESULTS_DIR, API_BASE_URL, API_KEY
from fontbench.evaluator import VLMEvaluator, _encode_image
from fontbench.fonts import _find_font_path

# 15 common Latin fonts matching FRB benchmark
FRB_FONTS = [
    ("Arial", "Arial.ttf"),
    ("Times New Roman", "Times New Roman.ttf"),
    ("Helvetica", "Helvetica.ttc"),
    ("Georgia", "Georgia.ttf"),
    ("Courier New", "Courier New.ttf"),
    ("Verdana", "Verdana.ttf"),
    ("Palatino", "Palatino.ttc"),
    ("Trebuchet MS", "Trebuchet MS.ttf"),
    ("Futura", "Futura.ttc"),
    ("Baskerville", "Baskerville.ttc"),
    ("Didot", "Didot.ttc"),
    ("Gill Sans", "GillSans.ttc"),
    ("Optima", "Optima.ttc"),
    ("American Typewriter", "AmericanTypewriter.ttc"),
    ("Cochin", "Cochin.ttc"),
]

# 10 English sentences for the easy condition
EASY_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "The five boxing wizards jump quickly.",
    "Sphinx of black quartz, judge my vow.",
    "Two driven jocks help fax my big quiz.",
    "The jay, pig, fox, zebra and my wolves quack!",
    "Crazy Frederick bought many very exquisite opal jewels.",
    "We promptly judged antique ivory buckles for the next prize.",
    "A mad boxer shot a quick, gloved jab to the jaw of his dizzy opponent.",
]

FRB_DATA_DIR = Path(__file__).parent / "data" / "frb"
FRB_IMAGES_DIR = FRB_DATA_DIR / "images"


def _render_frb_image(text, font_name, font_path, font_size=36):
    """Render text in the given font on a white background."""
    try:
        font_obj = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font_obj = ImageFont.load_default()

    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = 40
    img_w = text_w + padding * 2
    img_h = text_h + padding * 2
    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = (img_w - text_w) // 2
    y = (img_h - text_h) // 2
    draw.text((x, y), text, font=font_obj, fill=(0, 0, 0))
    return img


def generate_frb_dataset():
    """Generate FRB-style images locally."""
    FRB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    samples = []
    font_names = [name for name, _ in FRB_FONTS]

    for font_name, font_file in FRB_FONTS:
        font_path = _find_font_path(font_file)
        if font_path is None:
            print(f"Warning: font {font_name} ({font_file}) not found, skipping")
            continue

        # Easy condition: 10 sentences
        for i, sentence in enumerate(EASY_SENTENCES):
            img_name = f"frb_easy_{font_name.replace(' ', '_')}_{i}.png"
            img = _render_frb_image(sentence, font_name, font_path)
            img.save(FRB_IMAGES_DIR / img_name)

            # Generate options: correct + 14 distractors (all 15 fonts)
            options = list(font_names)
            random.shuffle(options)

            samples.append({
                "id": f"frb_easy_{font_name}_{i}",
                "image_path": f"images/{img_name}",
                "font_name": font_name,
                "difficulty": "easy",
                "text": sentence,
                "options": options,
            })

        # Hard condition: font name as text (stroop effect)
        for i, other_font_name in enumerate(font_names):
            img_name = f"frb_hard_{font_name.replace(' ', '_')}_{i}.png"
            # Render another font's NAME using this font (stroop effect)
            img = _render_frb_image(other_font_name, font_name, font_path)
            img.save(FRB_IMAGES_DIR / img_name)

            options = list(font_names)
            random.shuffle(options)

            samples.append({
                "id": f"frb_hard_{font_name}_{i}",
                "image_path": f"images/{img_name}",
                "font_name": font_name,
                "difficulty": "hard",
                "text": other_font_name,
                "options": options,
            })

    # Save metadata
    with open(FRB_DATA_DIR / "metadata.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(samples)} FRB samples ({FRB_IMAGES_DIR})")
    return samples


def _evaluate_mc_n_way(client, model_id, image_path, question, options, max_retries=3):
    """Evaluate an N-way MCQ (supports more than 4 options)."""
    # Build letter labels: A, B, C, ..., O (up to 15)
    labels = [chr(65 + i) for i in range(len(options))]
    valid_labels = set(labels)
    label_str = ", ".join(labels)

    options_str = "\n".join(f"{lbl}) {opt}" for lbl, opt in zip(labels, options))
    prompt = f"{question}\n\n{options_str}\n\nAnswer with only the letter ({label_str})."

    b64_image = _encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    response = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
            )
            response = resp.choices[0].message.content
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                response = f"ERROR: {e}"

    # Parse answer letter (support A-O)
    parsed = None
    for char in response.strip().upper():
        if char in valid_labels:
            idx = ord(char) - 65
            if idx < len(options):
                parsed = options[idx]
            break

    return {"response": response, "parsed_answer": parsed, "options": options}


def run_frb_evaluation(samples=None, models_to_eval=None, api_base=None, api_key=None):
    """Run FRB evaluation on all models."""
    if samples is None:
        with open(FRB_DATA_DIR / "metadata.json") as f:
            samples = json.load(f)

    if models_to_eval is None:
        models_to_eval = MODELS

    _api_base = api_base or API_BASE_URL
    _api_key = api_key or API_KEY

    font_names = [name for name, _ in FRB_FONTS]
    all_results = {}

    for model_cfg in models_to_eval:
        print(f"\n--- FRB eval: {model_cfg['name']} ---")
        client = openai.OpenAI(base_url=_api_base, api_key=_api_key)

        results = []
        for sample in tqdm(samples, desc=f"FRB eval ({model_cfg['name']})"):
            image_path = str(FRB_DATA_DIR / sample["image_path"])
            question = "What font is used to render the text in this image?"
            options = sample["options"]

            result = _evaluate_mc_n_way(client, model_cfg["id"], image_path, question, options)
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

        easy_acc = sum(1 for r in easy_results if r["correct"]) / max(len(easy_results), 1)
        hard_acc = sum(1 for r in hard_results if r["correct"]) / max(len(hard_results), 1)

        scores = {
            "overall_accuracy": total_correct / max(total, 1),
            "easy_accuracy": easy_acc,
            "hard_accuracy": hard_acc,
            "total_samples": total,
        }

        all_results[model_cfg["name"]] = {
            "model": model_cfg,
            "scores": scores,
            "results": results,
        }

        print(f"  Overall: {scores['overall_accuracy']:.3f}")
        print(f"  Easy:    {scores['easy_accuracy']:.3f}")
        print(f"  Hard:    {scores['hard_accuracy']:.3f}")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FRB Cross-Benchmark Evaluation")
    parser.add_argument("--models", nargs="*", help="Model names to evaluate (default: all)")
    parser.add_argument("--generate-only", action="store_true", help="Only generate dataset, skip eval")
    parser.add_argument("--api-base", type=str, default=None, help="Custom API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="Custom API key")
    args = parser.parse_args()

    print("=== FRB Cross-Benchmark Evaluation ===\n")

    # Step 1: Generate FRB dataset (if not already generated)
    if not (FRB_DATA_DIR / "metadata.json").exists():
        print("Step 1: Generating FRB-style images...")
        samples = generate_frb_dataset()
    else:
        print("Step 1: Loading existing FRB dataset...")
        with open(FRB_DATA_DIR / "metadata.json") as f:
            samples = json.load(f)
        print(f"  Loaded {len(samples)} samples")

    if args.generate_only:
        return

    # Step 2: Run evaluation
    models_to_eval = MODELS
    if args.models:
        models_to_eval = [m for m in MODELS if m["name"] in args.models]

    # Load existing results to append to
    frb_results_path = RESULTS_DIR / "frb_results.json"
    existing_results = {}
    if frb_results_path.exists():
        with open(frb_results_path) as f:
            existing_results = json.load(f)

    # Skip models already evaluated
    remaining = [m for m in models_to_eval if m["name"] not in existing_results]
    if not remaining:
        print("All requested models already evaluated.")
        return

    print(f"\nStep 2: Running FRB evaluation on {len(remaining)} models...")
    new_results = run_frb_evaluation(samples, remaining, api_base=args.api_base, api_key=args.api_key)

    # Merge with existing results
    existing_results.update(new_results)
    with open(frb_results_path, "w") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)
    print(f"\nMerged results saved to {frb_results_path}")


if __name__ == "__main__":
    main()
