"""Comprehensive test suite for NST dataset and preprocessing pipeline.
Tests:
- Device selection and hardware acceleration detection
- Proportional resizing without enlarging smaller images
- RGB image loading and validation
- Tensor shape (1x3xHxW) and value range [0, 1]
- ImageNet normalization and prevention of double normalization
- Denormalization back to [0, 1] and numerical reconstruction accuracy
- Tensor-to-image conversion and file saving
- Corrupted, missing, and invalid file error handling
- Independent sizing for content and style
"""

import os
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch

from src.utils import (
    get_device,
    calculate_proportional_size,
    load_image,
    image_to_tensor,
    normalize_imagenet,
    denormalize_imagenet,
    is_normalized,
    tensor_to_image,
    save_image,
    CorruptedImageError,
    DoubleNormalizationError,
    InvalidImageError,
)


class TestNSTPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        print(f"\n[Test Setup] Running tests on compute device: {cls.device}")

        # Dataset paths
        cls.content_mountain = "data/content/landscape_mountain.jpg"
        cls.content_lake = "data/content/landscape_lake.jpg"
        cls.content_forest = "data/content/landscape_forest.jpg"

        cls.style_van_gogh = "data/style/style_van_gogh.jpg"
        cls.style_hokusai = "data/style/style_hokusai.jpg"
        cls.style_kandinsky = "data/style/style_kandinsky.jpg"

    def test_01_device_acceleration(self):
        """Test device detection matches available hardware in current environment."""
        dev = get_device()
        self.assertIsInstance(dev, torch.device)
        # Verify valid compute device string
        self.assertIn(dev.type, ["cpu", "cuda", "mps"])
        print(f"Verified active device: {dev.type}")

    def test_02_proportional_resizing_rules(self):
        """Test proportional resize logic: max 256px, no enlargement of small images."""
        # 1. Smaller image: MUST NOT enlarge
        small_size = (120, 80)
        res_small = calculate_proportional_size(small_size, max_size=256)
        self.assertEqual(res_small, (120, 80), "Small image must not be enlarged!")

        # 2. Large landscape image (e.g., 3520 x 1980)
        large_w = (3520, 1980)
        new_w, new_h = calculate_proportional_size(large_w, max_size=256)
        self.assertEqual(new_w, 256)
        self.assertEqual(new_h, round(1980 * (256 / 3520)))
        self.assertTrue(new_h <= 256)

        # 3. Large portrait image (e.g., 1000 x 2000)
        large_h = (1000, 2000)
        new_w, new_h = calculate_proportional_size(large_h, max_size=256)
        self.assertEqual(new_h, 256)
        self.assertEqual(new_w, round(1000 * (256 / 2000)))
        self.assertTrue(new_w <= 256)

        # 4. Exact 256 boundary
        exact = (256, 128)
        self.assertEqual(calculate_proportional_size(exact, max_size=256), (256, 128))

        # 5. Invalid sizes should raise error
        with self.assertRaises(InvalidImageError):
            calculate_proportional_size((0, 100))
        with self.assertRaises(InvalidImageError):
            calculate_proportional_size((-50, 100))

    def test_03_load_image_rgb_and_dimensions(self):
        """Test loading actual images, ensuring RGB mode and max 256 proportional bound."""
        all_images = [
            self.content_mountain,
            self.content_lake,
            self.content_forest,
            self.style_van_gogh,
            self.style_hokusai,
            self.style_kandinsky,
        ]

        for img_path in all_images:
            self.assertTrue(os.path.exists(img_path), f"File should exist: {img_path}")
            img = load_image(img_path, max_size=256)
            self.assertEqual(img.mode, "RGB", f"Image {img_path} must be RGB")
            w, h = img.size
            self.assertTrue(max(w, h) == 256, f"Max dimension of {img_path} should be 256, got {w}x{h}")
            self.assertTrue(min(w, h) > 0)

    def test_04_tensor_conversion_and_range(self):
        """Test tensor dimensions (1x3xHxW), float32 dtype, and [0, 1] range."""
        for path in [self.content_mountain, self.style_van_gogh]:
            tensor = image_to_tensor(path, max_size=256, device=self.device)

            # Dimensions check: 1 x 3 x H x W
            self.assertEqual(tensor.ndim, 4)
            self.assertEqual(tensor.shape[0], 1, "Batch dimension must be 1")
            self.assertEqual(tensor.shape[1], 3, "Channel dimension must be 3 (RGB)")
            self.assertTrue(tensor.shape[2] <= 256)
            self.assertTrue(tensor.shape[3] <= 256)
            self.assertEqual(max(tensor.shape[2], tensor.shape[3]), 256)

            # Dtype check
            self.assertEqual(tensor.dtype, torch.float32)

            # Range check: strictly in [0.0, 1.0]
            t_min = tensor.min().item()
            t_max = tensor.max().item()
            self.assertGreaterEqual(t_min, 0.0, f"Min tensor value {t_min} < 0.0")
            self.assertLessEqual(t_max, 1.0, f"Max tensor value {t_max} > 1.0")

            # Check raw tensor is marked as NOT normalized
            self.assertFalse(is_normalized(tensor))

    def test_05_different_sizes_content_and_style(self):
        """Verify content and style images can have different dimensions."""
        t_content = image_to_tensor(self.content_mountain, max_size=256)
        t_style = image_to_tensor(self.style_van_gogh, max_size=256)

        # Mountain: 3520x1980 -> 256x144
        # Van Gogh: 4000x3184 -> 256x204
        c_shape = tuple(t_content.shape)
        s_shape = tuple(t_style.shape)

        self.assertNotEqual(c_shape, s_shape, "Content and style shapes can differ based on aspect ratio")
        self.assertEqual(c_shape[0], 1)
        self.assertEqual(c_shape[1], 3)
        self.assertEqual(s_shape[0], 1)
        self.assertEqual(s_shape[1], 3)
        print(f"Content tensor shape: {c_shape} | Style tensor shape: {s_shape}")

    def test_06_imagenet_normalization_and_prevention(self):
        """Test ImageNet normalization and verify double normalization is prevented."""
        raw = image_to_tensor(self.content_lake, max_size=256, device=self.device)
        self.assertFalse(is_normalized(raw))

        # Perform first normalization
        norm = normalize_imagenet(raw)
        self.assertTrue(is_normalized(norm))

        # Normalized values should have negative minimum (due to mean subtraction)
        self.assertLess(norm.min().item(), 0.0)

        # Attempting second normalization MUST raise DoubleNormalizationError
        with self.assertRaises(DoubleNormalizationError):
            normalize_imagenet(norm)

    def test_07_denormalization_and_reconstruction(self):
        """Test denormalization reconstructs original raw tensor within numerical precision."""
        raw = image_to_tensor(self.content_forest, max_size=256, device=self.device)
        norm = normalize_imagenet(raw)
        reconstructed = denormalize_imagenet(norm)

        # Reconstructed must have values in [0, 1]
        self.assertGreaterEqual(reconstructed.min().item(), 0.0)
        self.assertLessEqual(reconstructed.max().item(), 1.0)
        self.assertFalse(is_normalized(reconstructed))

        # Check accuracy of reconstruction
        diff = torch.abs(raw - reconstructed).max().item()
        self.assertLess(diff, 1e-5, f"Reconstruction difference {diff} exceeds tolerance 1e-5")

    def test_08_tensor_to_image_and_save(self):
        """Test conversion of tensor (raw & normalized) to PIL image and saving to disk."""
        raw = image_to_tensor(self.style_hokusai, max_size=256)
        norm = normalize_imagenet(raw)

        # Convert raw to image
        img_from_raw = tensor_to_image(raw)
        self.assertIsInstance(img_from_raw, Image.Image)
        self.assertEqual(img_from_raw.mode, "RGB")
        self.assertEqual(img_from_raw.size, (raw.shape[3], raw.shape[2]))

        # Convert normalized to image (auto-denormalized)
        img_from_norm = tensor_to_image(norm)
        self.assertEqual(img_from_norm.size, (norm.shape[3], norm.shape[2]))

        # Save to disk
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_saved.jpg"
            saved_str = save_image(norm, out_path)
            self.assertTrue(os.path.isfile(saved_str))
            self.assertGreater(os.path.getsize(saved_str), 1000)

            # Reload and check
            with Image.open(saved_str) as reloaded:
                self.assertEqual(reloaded.mode, "RGB")
                self.assertEqual(reloaded.size, (norm.shape[3], norm.shape[2]))

    def test_09_corrupted_and_missing_file_handling(self):
        """Test graceful handling of missing, empty, and corrupted files."""
        # 1. Non-existent file
        with self.assertRaises(FileNotFoundError):
            load_image("data/content/does_not_exist.jpg")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 2. Empty file (0 bytes)
            empty_path = Path(tmpdir) / "empty.jpg"
            empty_path.touch()
            with self.assertRaises(CorruptedImageError):
                load_image(empty_path)

            # 3. Corrupted / garbage content
            corrupted_path = Path(tmpdir) / "corrupted.jpg"
            with open(corrupted_path, "wb") as f:
                f.write(b"NOT_A_JPEG_FILE_JUST_RANDOM_GARBAGE_BYTES_1234567890")
            with self.assertRaises(CorruptedImageError):
                load_image(corrupted_path)

            # 4. Truncated JPEG header
            truncated_path = Path(tmpdir) / "truncated.jpg"
            with open(truncated_path, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
            with self.assertRaises(CorruptedImageError):
                load_image(truncated_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)

