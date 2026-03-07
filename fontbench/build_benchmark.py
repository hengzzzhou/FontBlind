"""Build the FontBench benchmark dataset."""
import json
import random
from pathlib import Path
from tqdm import tqdm
from fontbench.generator import SyntheticGenerator
from fontbench.questions import generate_mc_questions, generate_open_ended_question
from fontbench.fonts import FontRegistry
from fontbench.config import FONT_SIZES, FONT_COLORS, FONT_STYLES, DIFFICULTIES

# Sample texts organized by sub_script
TEXTS_BY_SUB_SCRIPT = {
    "latin": [
        "The quick brown fox jumps over the lazy dog",
        "Hello, World!",
        "Typography is the art of arranging type",
        "Design is not just what it looks like",
        "Fonts matter more than you think",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz 0123456789",
        "The five boxing wizards jump quickly",
    ],
    "chinese": [
        "你好世界",
        "字体识别是一项重要的任务",
        "视觉语言模型的能力评估",
        "人工智能改变未来",
        "深度学习与计算机视觉",
        "自然语言处理技术",
    ],
    "arabic": [
        "مرحبا بالعالم",
        "الخط العربي فن جميل",
        "التعلم العميق والذكاء الاصطناعي",
    ],
    "devanagari": [
        "नमस्ते दुनिया",
        "हिंदी एक सुंदर भाषा है",
        "कृत्रिम बुद्धिमत्ता का भविष्य",
    ],
}

# Backward-compatible aliases for generate_train_data.py
LATIN_TEXTS = TEXTS_BY_SUB_SCRIPT["latin"]
CJK_TEXTS = TEXTS_BY_SUB_SCRIPT["chinese"]
OTHER_TEXTS = TEXTS_BY_SUB_SCRIPT["arabic"] + TEXTS_BY_SUB_SCRIPT["devanagari"]

BACKGROUNDS_BY_DIFFICULTY = {
    "easy": ["white"],
    "medium": ["colored", "gradient"],
    "hard": ["textured", "gradient"],
}


def build_synthetic_subset(
    output_dir,
    num_samples=250,
    seed=42,
):
    random.seed(seed)
    gen = SyntheticGenerator()
    registry = FontRegistry()

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    samples = []
    all_fonts = registry.all_fonts()

    for i in tqdm(range(num_samples), desc="Generating synthetic samples"):
        font = random.choice(all_fonts)
        style = random.choice(font.styles)

        text = random.choice(TEXTS_BY_SUB_SCRIPT[font.sub_script])

        font_size = random.choice(FONT_SIZES)
        font_color = random.choice(list(FONT_COLORS.keys()))
        difficulty = random.choice(DIFFICULTIES)
        background = random.choice(BACKGROUNDS_BY_DIFFICULTY[difficulty])

        result = gen.generate_one(
            text=text,
            font_name=font.name,
            font_size=font_size,
            font_color=font_color,
            font_style=style,
            background=background,
            difficulty=difficulty,
        )

        image_filename = f"synth_{i:04d}.png"
        result["image"].save(images_dir / image_filename)

        mc_questions = generate_mc_questions(result["metadata"])
        open_ended = generate_open_ended_question(result["metadata"])

        sample = {
            "id": f"synth_{i:04d}",
            "image_path": f"images/{image_filename}",
            "metadata": result["metadata"],
            "mc_questions": mc_questions,
            "open_ended_question": open_ended,
            "source": "synthetic",
        }
        samples.append(sample)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    return samples


if __name__ == "__main__":
    from fontbench.config import SYNTHETIC_DIR
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_synthetic_subset(SYNTHETIC_DIR, num_samples=250)
    print(f"Generated {len(samples)} synthetic samples")
