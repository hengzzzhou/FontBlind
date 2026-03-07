# tests/test_transforms.py
"""Tests for image transforms."""
from PIL import Image
from fontbench.transforms import (
    GaussianNoise,
    GaussianBlur,
    JPEGCompression,
    Rotation,
    Resize,
    ROBUSTNESS_TRANSFORMS,
    RESOLUTION_TRANSFORMS,
    ALL_TRANSFORMS,
)


def _make_test_image(width=200, height=100):
    """Create a simple test image."""
    return Image.new("RGB", (width, height), (128, 64, 32))


def test_gaussian_noise_produces_valid_image():
    img = _make_test_image()
    transform = GaussianNoise(sigma=25)
    result = transform(img)
    assert isinstance(result, Image.Image)
    assert result.size == img.size
    assert result.mode == "RGB"


def test_gaussian_blur_produces_valid_image():
    img = _make_test_image()
    transform = GaussianBlur(radius=2)
    result = transform(img)
    assert isinstance(result, Image.Image)
    assert result.size == img.size


def test_jpeg_compression_produces_valid_image():
    img = _make_test_image()
    transform = JPEGCompression(quality=50)
    result = transform(img)
    assert isinstance(result, Image.Image)
    assert result.size == img.size
    assert result.mode == "RGB"


def test_rotation_produces_valid_image():
    img = _make_test_image()
    transform = Rotation(angle=15)
    result = transform(img)
    assert isinstance(result, Image.Image)
    # Rotation with expand=True may change dimensions
    assert result.width >= img.width or result.height >= img.height


def test_resize_produces_valid_image():
    img = _make_test_image(200, 100)
    transform = Resize(scale=0.5)
    result = transform(img)
    assert isinstance(result, Image.Image)
    assert result.size == (100, 50)


def test_resize_upscale():
    img = _make_test_image(200, 100)
    transform = Resize(scale=2.0)
    result = transform(img)
    assert result.size == (400, 200)


def test_transform_name_attribute():
    assert GaussianNoise(10).name == "gaussian_noise_sigma10"
    assert GaussianBlur(2).name == "gaussian_blur_r2"
    assert JPEGCompression(75).name == "jpeg_q75"
    assert Rotation(45).name == "rotation_45deg"
    assert Resize(0.5).name == "resize_0.5x"


def test_predefined_transform_dicts():
    assert "gaussian_noise" in ROBUSTNESS_TRANSFORMS
    assert "gaussian_blur" in ROBUSTNESS_TRANSFORMS
    assert "jpeg_compression" in ROBUSTNESS_TRANSFORMS
    assert "rotation" in ROBUSTNESS_TRANSFORMS
    assert "resolution" in RESOLUTION_TRANSFORMS
    assert len(ALL_TRANSFORMS) == len(ROBUSTNESS_TRANSFORMS) + len(RESOLUTION_TRANSFORMS)


def test_all_predefined_transforms_callable():
    img = _make_test_image()
    for category, transforms in ALL_TRANSFORMS.items():
        for t in transforms:
            result = t(img)
            assert isinstance(result, Image.Image), f"{t.name} did not return an Image"
