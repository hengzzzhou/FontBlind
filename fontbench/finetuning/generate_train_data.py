"""Generate training data for fine-tuning VLMs on font recognition."""
import json
import random
from pathlib import Path
from tqdm import tqdm
from fontbench.generator import SyntheticGenerator
from fontbench.fonts import FontRegistry
from fontbench.config import FONT_SIZES, FONT_COLORS, FONT_STYLES
from fontbench.build_benchmark import TEXTS_BY_SUB_SCRIPT

BACKGROUNDS = ["white", "colored", "gradient", "textured"]


def generate_training_data(
    output_dir,
    num_samples=3000,
    seed=123,
):
    """Generate training QA pairs for fine-tuning.

    Each sample: image + conversation (question + answer about font properties).
    Output format: JSONL compatible with common VLM fine-tuning frameworks.
    """
    random.seed(seed)
    gen = SyntheticGenerator()
    registry = FontRegistry()
    all_fonts = registry.all_fonts()

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    conversations = []

    for i in tqdm(range(num_samples), desc="Generating training data"):
        font = random.choice(all_fonts)
        style = random.choice(font.styles)

        text = random.choice(TEXTS_BY_SUB_SCRIPT[font.sub_script])

        font_size = random.choice(FONT_SIZES)
        font_color = random.choice(list(FONT_COLORS.keys()))
        background = random.choice(BACKGROUNDS)

        result = gen.generate_one(
            text=text,
            font_name=font.name,
            font_size=font_size,
            font_color=font_color,
            font_style=style,
            background=background,
            difficulty="easy",  # training data uses simple setup
        )

        image_filename = f"train_{i:05d}.png"
        result["image"].save(images_dir / image_filename)

        meta = result["metadata"]

        # Create QA conversation
        answer = (
            f"The text in this image has the following font properties:\n"
            f"- Font family: {meta['font_family']}\n"
            f"- Font size: {meta['font_size_bucket']}\n"
            f"- Font style: {meta['font_style']}\n"
            f"- Font color: {meta['font_color']}"
        )

        conversation = {
            "id": f"train_{i:05d}",
            "image": f"images/{image_filename}",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescribe all font properties of the text in this image, including font family, size, style, and color.",
                },
                {
                    "from": "gpt",
                    "value": answer,
                },
            ],
        }
        conversations.append(conversation)

    # Save as JSONL
    output_file = output_dir / "train.jsonl"
    with open(output_file, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Generated {len(conversations)} training samples")
    print(f"Images: {images_dir}")
    print(f"Conversations: {output_file}")
    return conversations


if __name__ == "__main__":
    output = Path("fontbench/finetuning/train_data")
    generate_training_data(output, num_samples=3000)
