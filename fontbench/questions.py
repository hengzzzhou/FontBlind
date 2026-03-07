"""Question generator for FontBench MC and open-ended tasks."""
import random
from fontbench.fonts import FontRegistry

_registry = FontRegistry()

# Distractor pools by sub_script
_FONTS_BY_SUB_SCRIPT = {}
for _f in _registry.all_fonts():
    _FONTS_BY_SUB_SCRIPT.setdefault(_f.sub_script, []).append(_f.name)

# Distractor pools by script
_FONTS_BY_SCRIPT = {}
for _f in _registry.all_fonts():
    _FONTS_BY_SCRIPT.setdefault(_f.script, []).append(_f.name)

_ALL_FONTS = [f.name for f in _registry.all_fonts()]

_SIZE_BUCKETS = ["small", "medium", "large", "xlarge"]
_COLORS = ["black", "red", "blue", "green", "gray", "orange", "purple", "brown"]
_STYLES = ["regular", "bold", "italic", "bold_italic"]


def _pick_distractors(correct, pool, n=3):
    candidates = [x for x in pool if x != correct]
    return random.sample(candidates, min(n, len(candidates)))


def generate_mc_questions(metadata):
    questions = []

    # Font family — use sub_script-aware distractor pool
    sub_script = metadata.get("sub_script", "latin")
    pool = _FONTS_BY_SUB_SCRIPT.get(sub_script, [])
    if len(pool) < 4:
        # Fallback to same script
        script = metadata.get("script", "latin")
        pool = _FONTS_BY_SCRIPT.get(script, [])
    if len(pool) < 4:
        # Fallback to all fonts
        pool = _ALL_FONTS
    distractors = _pick_distractors(metadata["font_family"], pool, 3)
    options = [metadata["font_family"]] + distractors
    random.shuffle(options)
    questions.append({
        "property": "font_family",
        "question": "What is the font family of the text in this image?",
        "options": options,
        "answer": metadata["font_family"],
    })

    # Font size bucket
    distractors = _pick_distractors(metadata["font_size_bucket"], _SIZE_BUCKETS, 3)
    options = [metadata["font_size_bucket"]] + distractors
    random.shuffle(options)
    questions.append({
        "property": "font_size",
        "question": "What is the approximate size of the text in this image?",
        "options": options,
        "answer": metadata["font_size_bucket"],
    })

    # Font style
    distractors = _pick_distractors(metadata["font_style"], _STYLES, 3)
    options = [metadata["font_style"]] + distractors
    random.shuffle(options)
    questions.append({
        "property": "font_style",
        "question": "What is the style of the text in this image?",
        "options": options,
        "answer": metadata["font_style"],
    })

    # Font color
    distractors = _pick_distractors(metadata["font_color"], _COLORS, 3)
    options = [metadata["font_color"]] + distractors
    random.shuffle(options)
    questions.append({
        "property": "font_color",
        "question": "What is the color of the text in this image?",
        "options": options,
        "answer": metadata["font_color"],
    })

    return questions


def generate_open_ended_question(metadata):
    return {
        "question": "Describe all font properties of the text in this image, including font family, size, style (regular/bold/italic), and color.",
        "ground_truth": {
            "font_family": metadata["font_family"],
            "font_size": metadata.get("font_size_bucket", "unknown"),
            "font_style": metadata["font_style"],
            "font_color": metadata["font_color"],
        },
    }
