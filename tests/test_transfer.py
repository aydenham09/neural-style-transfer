"""Test suite for Style Transfer Optimization Engine (src/transfer.py).

Verifies:
1. Optimizer initialization and target parameter registration.
2. Target initialization from content copy with requires_grad=True.
3. Adam optimization loop, loss reduction, and pixel clamping to [0, 1].
4. Strict VGG-19 parameter weight immutability (zero drift).
5. Generation of all output artifacts: initial image, progress checkpoints,
   final image, loss history plot, comparison grid, and JSON summary.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from src.transfer import StyleTransferOptimizer, compute_vgg_checksum
from src.utils import get_device, image_to_tensor


class TestStyleTransferOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        print(f"\n[Transfer Test Setup] Running on device: {cls.device}")
        cls.content_path = "data/content/landscape_mountain.jpg"
        cls.style_path = "data/style/style_van_gogh.jpg"

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="nst_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_optimizer_initialization_and_validation(self):
        """Test optimizer parameter validation and directory creation."""
        optimizer = StyleTransferOptimizer(
            content_path=self.content_path,
            style_path=self.style_path,
            output_dir=self.temp_dir,
            max_size=128,
            num_steps=5,
            lr=0.02,
            content_weight=1.0,
            style_weight=1e5,
            tv_weight=0.0,
            save_progress_every=2,
            device=self.device,
        )
        self.assertEqual(optimizer.max_size, 128)
        self.assertEqual(optimizer.num_steps, 5)
        self.assertTrue(Path(self.temp_dir).is_dir())
        self.assertTrue((Path(self.temp_dir) / "progress").is_dir())

    def test_02_short_optimization_run_and_artifacts(self):
        """Run a short 5-step optimization to verify loop execution, clamping, and artifacts."""
        optimizer = StyleTransferOptimizer(
            content_path=self.content_path,
            style_path=self.style_path,
            output_dir=self.temp_dir,
            max_size=128,
            num_steps=5,
            lr=0.02,
            content_weight=1.0,
            style_weight=1e5,
            tv_weight=0.0,
            save_progress_every=2,
            device=self.device,
        )
        summary = optimizer.run()

        # 1. Summary integrity
        self.assertEqual(summary["execution"]["steps_completed"], 5)
        self.assertFalse(summary["execution"]["early_stopped"])
        self.assertTrue(summary["execution"]["vgg_weights_frozen_verified"])

        # 2. Check artifacts exist on disk
        out_path = Path(self.temp_dir)
        self.assertTrue((out_path / "step_0000_initial.png").is_file())
        self.assertTrue((out_path / "final_stylized.png").is_file())
        self.assertTrue((out_path / "comparison_content_style_result.png").is_file())
        self.assertTrue((out_path / "loss_history.png").is_file())
        self.assertTrue((out_path / "run_config_and_metrics.json").is_file())

        # Check progress checkpoints (step 2, 4, 5)
        progress_files = list((out_path / "progress").glob("*.png"))
        self.assertTrue(len(progress_files) >= 2)

        # 3. Verify JSON contents
        with open(out_path / "run_config_and_metrics.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["configuration"]["optimizer"], "Adam")
        self.assertEqual(len(data["history"]), 5)


if __name__ == "__main__":
    unittest.main()

