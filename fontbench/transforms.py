"""Image transforms for robustness and resolution ablation experiments."""
import json
import shutil
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np


class GaussianNoise:
    """Add Gaussian noise to an image."""

    def __init__(self, sigma):
        self.sigma = sigma
        self.name = f"gaussian_noise_sigma{sigma}"

    def __call__(self, img):
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, self.sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class GaussianBlur:
    """Apply Gaussian blur to an image."""

    def __init__(self, radius):
        self.radius = radius
        self.name = f"gaussian_blur_r{radius}"

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class JPEGCompression:
    """Simulate JPEG compression artifacts."""

    def __init__(self, quality):
        self.quality = quality
        self.name = f"jpeg_q{quality}"

    def __call__(self, img):
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class Rotation:
    """Rotate image by a given angle."""

    def __init__(self, angle):
        self.angle = angle
        self.name = f"rotation_{angle}deg"

    def __call__(self, img):
        return img.rotate(self.angle, expand=True, fillcolor=(255, 255, 255))


class Resize:
    """Resize image by a scale factor (relative to original)."""

    def __init__(self, scale):
        self.scale = scale
        self.name = f"resize_{scale}x"

    def __call__(self, img):
        new_w = max(1, int(img.width * self.scale))
        new_h = max(1, int(img.height * self.scale))
        return img.resize((new_w, new_h), Image.LANCZOS)


# Predefined transform sets
ROBUSTNESS_TRANSFORMS = {
    "gaussian_noise": [GaussianNoise(sigma=s) for s in [10, 25, 50]],
    "gaussian_blur": [GaussianBlur(radius=r) for r in [1, 2, 4]],
    "jpeg_compression": [JPEGCompression(quality=q) for q in [75, 50, 25, 10]],
    "rotation": [Rotation(angle=a) for a in [5, 15, 30, 45]],
}

RESOLUTION_TRANSFORMS = {
    "resolution": [Resize(scale=s) for s in [2.0, 1.0, 0.5, 0.25]],
}

ALL_TRANSFORMS = {**ROBUSTNESS_TRANSFORMS, **RESOLUTION_TRANSFORMS}


def apply_transform_to_dataset(data_dir, output_dir, transform):
    """Apply a transform to all images in a dataset, copying metadata.

    Args:
        data_dir: Path to original dataset (with images/ and metadata.json).
        output_dir: Path to write transformed dataset.
        transform: A callable transform with a .name attribute.

    Returns:
        Path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_images = output_dir / "images"
    out_images.mkdir(exist_ok=True)

    # Copy metadata unchanged
    shutil.copy2(data_dir / "metadata.json", output_dir / "metadata.json")

    # Transform each image
    src_images = data_dir / "images"
    for img_path in sorted(src_images.glob("*.png")):
        img = Image.open(img_path).convert("RGB")
        transformed = transform(img)
        transformed.save(out_images / img_path.name)

    return output_dir
