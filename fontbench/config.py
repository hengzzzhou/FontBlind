"""Central configuration for FontBench."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
REALWORLD_DIR = DATA_DIR / "realworld"
RESULTS_DIR = PROJECT_ROOT / "results"

API_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/")
API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODELS = [
    # --- Open-Source: Qwen2.5-VL series ---
    {"id": "Pro/Qwen/Qwen2.5-VL-7B-Instruct", "name": "Qwen2.5-VL-7B", "type": "open-source", "size": "7B"},
    {"id": "Qwen/Qwen2.5-VL-32B-Instruct", "name": "Qwen2.5-VL-32B", "type": "open-source", "size": "32B"},
    {"id": "Qwen/Qwen2.5-VL-72B-Instruct", "name": "Qwen2.5-VL-72B", "type": "open-source", "size": "72B"},
    # --- Open-Source: Qwen3-VL series ---
    {"id": "qwen3-vl-8b-instruct", "name": "Qwen3-VL-8B", "type": "open-source", "size": "8B"},
    {"id": "qwen3-vl-30b-a3b-instruct", "name": "Qwen3-VL-30B-A3B", "type": "open-source", "size": "30B-A3B"},
    {"id": "Qwen/Qwen3-VL-32B-Instruct", "name": "Qwen3-VL-32B", "type": "open-source", "size": "32B"},
    {"id": "qwen-vl-max", "name": "Qwen3-Max", "type": "open-source", "size": "large"},
    # --- Open-Source: Other ---
    {"id": "mistralai/pixtral-12b", "name": "Pixtral-12B", "type": "open-source", "size": "12B"},
    # --- Commercial APIs ---
    {"id": "gpt-5.2", "name": "GPT-5.2", "type": "commercial", "size": "-"},
    {"id": "gemini-3-flash-preview", "name": "Gemini-3-Flash", "type": "commercial", "size": "-"},
    {"id": "gemini-3-pro-preview", "name": "Gemini-3-Pro", "type": "commercial", "size": "-"},
    {"id": "claude-sonnet-4-6", "name": "Claude-Sonnet-4.6", "type": "commercial", "size": "-"},
    {"id": "doubao-seed-1-6-250615", "name": "Doubao-Seed-1.6", "type": "commercial", "size": "-"},
    {"id": "glm-4.5v", "name": "GLM-4.5V", "type": "commercial", "size": "-"},
    {"id": "glm-4.6v", "name": "GLM-4.6V", "type": "commercial", "size": "-"},
]

# Font properties
FONT_SIZES = [12, 16, 20, 24, 32, 40, 48, 64]
SIZE_BUCKETS = {"small": (12, 20), "medium": (20, 36), "large": (36, 52), "xlarge": (52, 72)}
FONT_COLORS = {
    "black": (0, 0, 0),
    "red": (220, 50, 50),
    "blue": (50, 50, 220),
    "green": (50, 150, 50),
    "gray": (128, 128, 128),
    "orange": (230, 130, 30),
    "purple": (150, 50, 180),
    "brown": (140, 90, 40),
}
FONT_STYLES = ["regular", "bold", "italic", "bold_italic"]
DIFFICULTIES = ["easy", "medium", "hard"]

CV_BASELINE_MODEL = {"id": "cv-baseline", "name": "CV-Baseline", "type": "traditional", "size": "-"}
