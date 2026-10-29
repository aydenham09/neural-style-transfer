"""Automated test suite for Streamlit Demo workflow and components.

Verifies:
1. All 8 landscape presets and 3 style presets exist, are readable, and non-corrupted.
2. Cached VGG-19 feature extractor initialization.
3. Complete end-to-end optimization workflow matching app.py logic with UI progress callback.
4. Download artifact preparation: byte stream extraction for PNG and JSON download buttons.
5. Robust error handling for missing files and corrupted images.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from app import LANDSCAPE_PRESETS, STYLE_PRESETS, load_vgg_extractor_cached, validate_image_file
from src.transfer import StyleTransferOptimizer
from src.utils import get_device


class TestDemoWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        print(f"\n[Demo Test Setup] Running on compute device: {cls.device}")

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="demo_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_all_presets_exist_and_valid(self):
        """Verify all 8 landscape presets and 3 style presets exist and are readable."""
        self.assertGreaterEqual(len(LANDSCAPE_PRESETS), 8)
        self.assertGreaterEqual(len(STYLE_PRESETS), 3)

        for key, info in LANDSCAPE_PRESETS.items():
            path = Path(info["path"])
            valid, msg = validate_image_file(path)
            self.assertTrue(valid, f"Landscape preset '{key}' failed validation: {msg}")
            with Image.open(path) as img:
                self.assertGreater(img.width, 0)
                self.assertGreater(img.height, 0)

        for key, info in STYLE_PRESETS.items():
            path = Path(info["path"])
            valid, msg = validate_image_file(path)
            self.assertTrue(valid, f"Style preset '{key}' failed validation: {msg}")
            with Image.open(path) as img:
                self.assertGreater(img.width, 0)
                self.assertGreater(img.height, 0)

    def test_02_cached_vgg_extractor_initialization(self):
        """Verify cached VGG-19 extractor loads correctly and is frozen."""
        extractor = load_vgg_extractor_cached()
        self.assertIsNotNone(extractor)
        self.assertTrue(all(not p.requires_grad for p in extractor.parameters()))

    def test_03_end_to_end_demo_workflow_and_downloads(self):
        """Test full demo workflow up to downloading PNG and JSON bytes."""
        extractor = load_vgg_extractor_cached()

        content_path = Path("data/content/landscape_mountain.jpg")
        style_path = Path("data/style/style_van_gogh.jpg")
        out_dir = Path(self.temp_dir)

        progress_calls = []

        def dummy_progress(step, total, metrics):
            progress_calls.append((step, total, metrics.get("total_loss")))

        # Run short 5-step optimization to verify demo pipeline logic
        optimizer = StyleTransferOptimizer(
            content_path=content_path,
            style_path=style_path,
            output_dir=out_dir,
            max_size=128,
            num_steps=5,
            lr=0.02,
            content_weight=1.0,
            style_weight=1e5,
            tv_weight=0.0,
            save_progress_every=2,
            device=self.device,
            extractor=extractor,
            progress_callback=dummy_progress,
        )

        summary = optimizer.run()

        # 1. Verify progress callback was triggered
        self.assertGreater(len(progress_calls), 0)
        self.assertEqual(progress_calls[-1][0], 5)

        # 2. Verify output files
        final_png = out_dir / "final_stylized.png"
        metrics_json = out_dir / "run_config_and_metrics.json"

        self.assertTrue(final_png.is_file())
        self.assertTrue(metrics_json.is_file())

        # 3. Simulate Streamlit PNG download button
        with open(final_png, "rb") as f:
            png_bytes = f.read()
        self.assertGreater(len(png_bytes), 1000)
        # Check that PNG header is valid
        self.assertEqual(png_bytes[:8], b"\x89PNG\r\n\x1a\n")

        # 4. Simulate Streamlit JSON download button
        json_str = json.dumps(summary, indent=2)
        parsed = json.loads(json_str)
        self.assertIn("metrics_summary", parsed)
        self.assertIn("quantization_delta_total_loss", parsed["metrics_summary"])

    def test_04_error_handling_missing_and_corrupt_files(self):
        """Verify proper exception handling on nonexistent and corrupt images."""
        # Nonexistent file
        bogus_path = Path(self.temp_dir) / "nonexistent.jpg"
        valid, msg = validate_image_file(bogus_path)
        self.assertFalse(valid)
        self.assertIn("tidak ditemukan", msg.lower())

        # Corrupt file
        corrupt_path = Path(self.temp_dir) / "corrupt.jpg"
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE_BYTES")
        valid, msg = validate_image_file(corrupt_path)
        self.assertFalse(valid)
        self.assertIn("rusak", msg.lower())


if __name__ == "__main__":
    unittest.main()

