"""Synthetic image generator for FontBench."""
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from fontbench.fonts import FontRegistry
from fontbench.config import FONT_COLORS, SIZE_BUCKETS


def _size_to_bucket(size):
    for bucket, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= size < hi:
            return bucket
    return "xlarge"


def _make_background(width, height, bg_type):
    if bg_type == "white":
        return Image.new("RGB", (width, height), (255, 255, 255))
    elif bg_type == "colored":
        color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        return Image.new("RGB", (width, height), color)
    elif bg_type == "gradient":
        img = Image.new("RGB", (width, height))
        c1 = [random.randint(180, 255) for _ in range(3)]
        c2 = [random.randint(180, 255) for _ in range(3)]
        for y in range(height):
            r = int(c1[0] + (c2[0] - c1[0]) * y / height)
            g = int(c1[1] + (c2[1] - c1[1]) * y / height)
            b = int(c1[2] + (c2[2] - c1[2]) * y / height)
            for x in range(width):
                img.putpixel((x, y), (r, g, b))
        return img
    elif bg_type == "textured":
        img = Image.new("RGB", (width, height), (240, 240, 240))
        # Add noise
        for _ in range(width * height // 10):
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            gray = random.randint(200, 255)
            img.putpixel((x, y), (gray, gray, gray))
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        return img
    else:
        return Image.new("RGB", (width, height), (255, 255, 255))


class SyntheticGenerator:
    def __init__(self):
        self.registry = FontRegistry()

    def _load_font(self, font_name, size, style):
        font = self.registry.get_font(font_name)
        if font is None:
            raise ValueError(f"Font not found: {font_name}")
        # For .ttc files, try index 0; for style variants we use the base file
        try:
            return ImageFont.truetype(str(font.path), size)
        except Exception:
            return ImageFont.load_default()

    def generate_one(
        self,
        text,
        font_name,
        font_size,
        font_color,
        font_style,
        background,
        difficulty,
    ):
        font_obj = self._load_font(font_name, font_size, font_style)
        font_entry = self.registry.get_font(font_name)
        color_rgb = FONT_COLORS.get(font_color, (0, 0, 0))

        # Measure text to size the image
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font_obj)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        padding = max(40, font_size)
        img_w = text_w + padding * 2
        img_h = text_h + padding * 2

        img = _make_background(img_w, img_h, background)
        draw = ImageDraw.Draw(img)
        x = (img_w - text_w) // 2
        y = (img_h - text_h) // 2
        draw.text((x, y), text, font=font_obj, fill=color_rgb)

        return {
            "image": img,
            "metadata": {
                "font_family": font_name,
                "font_size": font_size,
                "font_size_bucket": _size_to_bucket(font_size),
                "font_color": font_color,
                "font_style": font_style,
                "script": font_entry.script if font_entry else "unknown",
                "sub_script": font_entry.sub_script if font_entry else "unknown",
                "difficulty": difficulty,
                "text": text,
                "background": background,
            },
        }

    def generate_multi(
        self,
        text_specs,
        background,
        difficulty,
    ):
        # Render each text region, then compose onto one image
        regions = []
        max_w = 0
        total_h = 40  # top padding

        rendered = []
        for spec in text_specs:
            font_obj = self._load_font(spec["font_name"], spec["font_size"], spec["font_style"])
            font_entry = self.registry.get_font(spec["font_name"])
            color_rgb = FONT_COLORS.get(spec["font_color"], (0, 0, 0))

            dummy_img = Image.new("RGB", (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_img)
            bbox = dummy_draw.textbbox((0, 0), spec["text"], font=font_obj)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            rendered.append({
                "font_obj": font_obj,
                "color_rgb": color_rgb,
                "text": spec["text"],
                "text_w": text_w,
                "text_h": text_h,
                "spec": spec,
                "font_entry": font_entry,
            })
            max_w = max(max_w, text_w)
            total_h += text_h + 20  # spacing between regions

        padding = 40
        img_w = max_w + padding * 2
        img_h = total_h + padding

        img = _make_background(img_w, img_h, background)
        draw = ImageDraw.Draw(img)

        y_cursor = padding
        for r in rendered:
            x = padding
            draw.text((x, y_cursor), r["text"], font=r["font_obj"], fill=r["color_rgb"])
            regions.append({
                "bbox": [x, y_cursor, x + r["text_w"], y_cursor + r["text_h"]],
                "metadata": {
                    "font_family": r["spec"]["font_name"],
                    "font_size": r["spec"]["font_size"],
                    "font_size_bucket": _size_to_bucket(r["spec"]["font_size"]),
                    "font_color": r["spec"]["font_color"],
                    "font_style": r["spec"]["font_style"],
                    "script": r["font_entry"].script if r["font_entry"] else "unknown",
                    "sub_script": r["font_entry"].sub_script if r["font_entry"] else "unknown",
                    "text": r["text"],
                },
            })
            y_cursor += r["text_h"] + 20

        return {
            "image": img,
            "regions": regions,
            "difficulty": difficulty,
            "background": background,
        }
