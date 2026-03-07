"""Traditional CV baseline for FontBench evaluation."""
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontbench.config import FONT_COLORS, SIZE_BUCKETS
from fontbench.fonts import FontRegistry


# Pre-compute color name -> RGB mapping for nearest-neighbor lookup
_COLOR_LIST = list(FONT_COLORS.items())  # [(name, (r, g, b)), ...]

# Template matching parameters
_REFERENCE_SIZE = 48  # Rendering size for reference templates
_MATCH_HEIGHT = 64    # Normalized height for binary comparison


def _binarize_text(img):
    """Convert image to binary text mask via background subtraction."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
    bg_val = np.mean(corners)
    if bg_val > 128:
        binary = gray < (bg_val - 40)
    else:
        binary = gray > (bg_val + 40)
    return binary.astype(np.uint8)


def _crop_to_bbox(binary):
    """Crop binary image to text bounding box."""
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return binary[rmin:rmax + 1, cmin:cmax + 1]


def _resize_to_height(binary, target_height):
    """Resize binary image to target height, preserving aspect ratio."""
    h, w = binary.shape
    if h == 0 or w == 0:
        return None
    target_width = max(1, int(w * target_height / h))
    img = Image.fromarray((binary * 255).astype(np.uint8), mode="L")
    resized = img.resize((target_width, target_height), Image.NEAREST)
    return (np.array(resized) > 128).astype(np.uint8)


def _normalized_cross_correlation(bin1, bin2):
    """Compute NCC between two binary images, padding to equal width."""
    h1, w1 = bin1.shape
    h2, w2 = bin2.shape
    if h1 != h2:
        return 0.0
    max_w = max(w1, w2)
    if w1 < max_w:
        bin1 = np.pad(bin1, ((0, 0), (0, max_w - w1)))
    if w2 < max_w:
        bin2 = np.pad(bin2, ((0, 0), (0, max_w - w2)))
    b1 = bin1.astype(np.float32)
    b2 = bin2.astype(np.float32)
    b1 -= b1.mean()
    b2 -= b2.mean()
    denom = np.sqrt(np.sum(b1 ** 2) * np.sum(b2 ** 2))
    if denom < 1e-8:
        return 0.0
    return float(np.sum(b1 * b2) / denom)


def _render_text_image(text, font_path, font_size):
    """Render text in given font as black on white."""
    try:
        font_obj = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        return None
    dummy = Image.new("RGB", (1, 1), (255, 255, 255))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font_obj)
    w = bbox[2] - bbox[0] + 20
    h = bbox[3] - bbox[1] + 20
    if w <= 0 or h <= 0:
        return None
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).text((10 - bbox[0], 10 - bbox[1]), text, font=font_obj, fill=(0, 0, 0))
    return img


def _detect_text_color(img):
    """Detect dominant text color via background subtraction and nearest-neighbor matching."""
    arr = np.array(img, dtype=np.float32)
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    bg = np.mean(corners, axis=0)

    diff = np.sqrt(np.sum((arr - bg) ** 2, axis=2))
    text_mask = diff > 50.0

    if not np.any(text_mask):
        return "black"

    text_pixels = arr[text_mask]
    mean_color = np.mean(text_pixels, axis=0)

    best_name = "black"
    best_dist = float("inf")
    for name, rgb in _COLOR_LIST:
        dist = np.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(mean_color, rgb)))
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name


def _detect_font_size(img):
    """Estimate font size bucket from text bounding box height."""
    arr = np.array(img, dtype=np.float32)
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    bg = np.mean(corners, axis=0)

    diff = np.sqrt(np.sum((arr - bg) ** 2, axis=2))
    text_mask = diff > 50.0

    if not np.any(text_mask):
        return "medium"

    rows = np.any(text_mask, axis=1)
    row_indices = np.where(rows)[0]
    text_height = row_indices[-1] - row_indices[0] + 1

    for bucket, (lo, hi) in SIZE_BUCKETS.items():
        estimated_size = text_height / 1.2
        if lo <= estimated_size < hi:
            return bucket

    return "xlarge"


def _detect_font_style(img):
    """Estimate font style using edge density and horizontal skew analysis."""
    arr = np.array(img.convert("L"), dtype=np.float32)

    grad_x = np.abs(np.diff(arr, axis=1))
    grad_y = np.abs(np.diff(arr, axis=0))

    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    bg_val = np.mean(corners)
    text_mask = np.abs(arr - bg_val) > 30

    if not np.any(text_mask):
        return "regular"

    text_area = np.sum(text_mask)
    edge_pixels_x = np.sum(grad_x > 30)
    edge_pixels_y = np.sum(grad_y > 30)
    edge_density = (edge_pixels_x + edge_pixels_y) / max(text_area, 1)

    is_bold = edge_density > 0.8

    rows = np.any(text_mask, axis=1)
    row_indices = np.where(rows)[0]
    if len(row_indices) > 10:
        top_rows = row_indices[:len(row_indices) // 4]
        bot_rows = row_indices[-len(row_indices) // 4:]

        top_cols = [np.mean(np.where(text_mask[r])[0]) for r in top_rows if np.any(text_mask[r])]
        bot_cols = [np.mean(np.where(text_mask[r])[0]) for r in bot_rows if np.any(text_mask[r])]

        if top_cols and bot_cols:
            skew = abs(np.mean(top_cols) - np.mean(bot_cols))
            is_italic = skew > 5.0
        else:
            is_italic = False
    else:
        is_italic = False

    if is_bold and is_italic:
        return "bold_italic"
    elif is_bold:
        return "bold"
    elif is_italic:
        return "italic"
    return "regular"


class CVBaselineEvaluator:
    """Traditional CV baseline using template matching and image analysis."""

    def __init__(self):
        self.registry = FontRegistry()

    def _detect_font_family(self, img, options, text=None):
        """Detect font family via template matching against rendered references.

        Renders the same text in each candidate font, binarizes both input and
        references, normalizes to a common height, and selects the candidate
        with the highest normalized cross-correlation.
        """
        input_binary = _binarize_text(img)
        input_cropped = _crop_to_bbox(input_binary)
        if input_cropped is None:
            return random.choice(options)

        input_resized = _resize_to_height(input_cropped, _MATCH_HEIGHT)
        if input_resized is None:
            return random.choice(options)

        ref_text = text if text else "ABCDEFGHabcdefgh"

        best_score = -1.0
        best_font = random.choice(options)

        for font_name in options:
            font = self.registry.get_font(font_name)
            if font is None:
                continue
            ref_img = _render_text_image(ref_text, font.path, _REFERENCE_SIZE)
            if ref_img is None:
                continue
            ref_binary = _binarize_text(ref_img)
            ref_cropped = _crop_to_bbox(ref_binary)
            if ref_cropped is None:
                continue
            ref_resized = _resize_to_height(ref_cropped, _MATCH_HEIGHT)
            if ref_resized is None:
                continue
            score = _normalized_cross_correlation(input_resized, ref_resized)
            if score > best_score:
                best_score = score
                best_font = font_name

        return best_font

    def evaluate_mc(self, image_path, question, options, metadata=None):
        """Evaluate a multiple-choice question using traditional CV methods.

        Args:
            image_path: Path to the image file.
            question: The question text.
            options: List of answer options.
            metadata: Optional sample metadata dict containing 'text' for template matching.

        Returns:
            dict with "parsed_answer" and "response" keys.
        """
        img = Image.open(image_path).convert("RGB")

        q_lower = question.lower()
        if "font family" in q_lower or "typeface" in q_lower:
            text = metadata.get("text") if metadata else None
            answer = self._detect_font_family(img, options, text)
            response = f"CV baseline: template matching -> {answer}"
        elif "size" in q_lower:
            detected = _detect_font_size(img)
            answer = detected if detected in options else random.choice(options)
            response = f"CV baseline: size detection -> {detected}"
        elif "style" in q_lower:
            detected = _detect_font_style(img)
            answer = detected if detected in options else random.choice(options)
            response = f"CV baseline: style detection -> {detected}"
        elif "color" in q_lower:
            detected = _detect_text_color(img)
            answer = detected if detected in options else random.choice(options)
            response = f"CV baseline: color detection -> {detected}"
        else:
            answer = random.choice(options)
            response = f"CV baseline: unknown question type -> {answer}"

        return {
            "parsed_answer": answer,
            "response": response,
        }
