"""Comprehensive test suite for VGG-19 Feature Extractor in NST.

Tests:
1. Model initialization, ImageNet pretrained weights loading, caching in checkpoints/.
2. Freezing of all parameters (requires_grad=False) and eval() mode.
3. Layer truncation (only up to index 29: conv5_1) and all ReLUs inplace=False.
4. Content and style layer mapping consistency.
5. Exact single ImageNet normalization (with double-normalization prevention even when attributes are lost).
6. Feature extraction on landscape image (max 256px), verifying shape and finite values.
7. Gradient propagation to input image tensor while VGG parameters remain completely frozen.
8. Device compatibility (MPS / CPU).
"""

import os
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.model import (
    DEFAULT_CONTENT_LAYER,
    DEFAULT_STYLE_LAYERS,
    MAX_REQUIRED_LAYER_INDEX,
    VGG19_LAYER_INDEX_MAP,
    VGG19FeatureExtractor,
    build_vgg19_extractor,
    is_tensor_imagenet_normalized,
)
from src.utils import get_device, image_to_tensor, normalize_imagenet


class TestVGG19FeatureExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        print(f"\n[Model Test Setup] Running on compute device: {cls.device}")
        cls.landscape_path = "data/content/landscape_mountain.jpg"
        cls.extractor = build_vgg19_extractor(checkpoint_dir="checkpoints", device=cls.device)

    def test_01_weights_cache_and_eval_mode(self):
        """Verify weights are cached in checkpoints/ and model is in eval mode."""
        self.assertFalse(self.extractor.training, "Extractor must be in eval() mode!")

        # Verify checkpoints directory contains cached hub weights
        checkpoint_dir = Path("checkpoints")
        self.assertTrue(checkpoint_dir.exists(), "checkpoints/ directory must exist")
        # Check that weight file exists in checkpoints
        files = list(checkpoint_dir.glob("**/*.pth"))
        self.assertTrue(len(files) > 0, f"Expected .pth file in checkpoints, found: {files}")
        print(f"Verified weight cache in: {files[0]}")

    def test_02_all_parameters_frozen(self):
        """Verify all parameters in the feature extractor have requires_grad = False."""
        param_count = 0
        for name, param in self.extractor.named_parameters():
            param_count += 1
            self.assertFalse(
                param.requires_grad,
                f"Parameter '{name}' must have requires_grad=False (must be frozen)!"
            )
        self.assertTrue(param_count > 0, "Model should have parameters.")
        print(f"Verified {param_count} parameters are strictly frozen.")

    def test_03_architecture_truncation_and_relu_inplace(self):
        """Verify architecture is truncated at index 29 and all ReLUs have inplace=False."""
        num_layers = len(self.extractor.features)
        self.assertEqual(
            num_layers,
            MAX_REQUIRED_LAYER_INDEX + 1,
            f"Expected exactly 30 layers (0 to 29), but found {num_layers}"
        )

        # Verify every ReLU in the feature extractor has inplace=False
        relu_count = 0
        for i, layer in enumerate(self.extractor.features):
            if isinstance(layer, nn.ReLU):
                relu_count += 1
                self.assertFalse(
                    layer.inplace,
                    f"features[{i}] ReLU must have inplace=False!"
                )
        self.assertTrue(relu_count >= 13, f"Expected at least 13 ReLUs, got {relu_count}")
        print(f"Verified {relu_count} ReLUs all have inplace=False.")

    def test_04_layer_index_mapping_consistency(self):
        """Verify content (conv4_2) and style (conv1_1, 2_1, 3_1, 4_1, 5_1) post-ReLU indices."""
        self.assertEqual(VGG19_LAYER_INDEX_MAP[DEFAULT_CONTENT_LAYER], 22)
        expected_style_indices = {
            "conv1_1": 1,
            "conv2_1": 6,
            "conv3_1": 11,
            "conv4_1": 20,
            "conv5_1": 29,
        }
        for name, expected_idx in expected_style_indices.items():
            self.assertEqual(
                VGG19_LAYER_INDEX_MAP[name],
                expected_idx,
                f"Layer {name} index mismatch: expected {expected_idx}, got {VGG19_LAYER_INDEX_MAP[name]}"
            )

    def test_05_normalization_detection_and_prevention(self):
        """Verify single ImageNet normalization even when attributes are stripped."""
        # 1. Raw image tensor in [0, 1]
        raw_tensor = torch.rand(1, 3, 100, 100, device=self.device)
        self.assertFalse(is_tensor_imagenet_normalized(raw_tensor))

        # 2. Normalized tensor
        norm_tensor = self.extractor.normalize(raw_tensor)
        self.assertTrue(is_tensor_imagenet_normalized(norm_tensor))

        # 3. Simulate attribute stripping via clone / arithmetic
        stripped_norm = norm_tensor.clone() + 0.0
        # If attribute was attached, arithmetic usually drops it:
        has_attr = hasattr(stripped_norm, "_is_imagenet_normalized")
        print(f"Attribute survived arithmetic? {has_attr}")
        # Even without attribute, our robust function MUST detect it as normalized!
        self.assertTrue(
            is_tensor_imagenet_normalized(stripped_norm),
            "Stripped normalized tensor must still be recognized via numerical distribution!"
        )

    def test_06_feature_extraction_landscape_image(self):
        """Test feature extraction on landscape photo with max dimension 256."""
        # Load landscape photo
        content_tensor = image_to_tensor(self.landscape_path, max_size=256, device=self.device)
        b, c, h, w = content_tensor.shape
        self.assertEqual(b, 1)
        self.assertEqual(c, 3)
        self.assertTrue(max(h, w) <= 256)
        print(f"Landscape image input shape: {list(content_tensor.shape)}")

        # Extract all features
        features = self.extractor(content_tensor)

        expected_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv4_2", "conv5_1"]
        for layer_name in expected_layers:
            self.assertIn(layer_name, features, f"Missing layer {layer_name} in output")
            feat = features[layer_name]
            self.assertEqual(feat.device.type, self.device.type)
            # Check finite values
            self.assertTrue(
                torch.isfinite(feat).all().item(),
                f"Features for {layer_name} contain non-finite values (NaN/Inf)!"
            )
            print(f"  Feature '{layer_name}': shape={list(feat.shape)}, finite=True")

        # Check content layer output specifically
        content_feat = self.extractor.get_content_feature(content_tensor)
        self.assertEqual(content_feat.shape, features[DEFAULT_CONTENT_LAYER].shape)

        # Check style layers output specifically
        style_feats = self.extractor.get_style_features(content_tensor)
        self.assertEqual(len(style_feats), 5)

    def test_07_gradient_reaches_input_tensor_while_weights_frozen(self):
        """Verify gradients flow to input tensor while all VGG weights remain frozen."""
        # Create an input image tensor requiring gradient (simulating image being optimized in NST)
        input_image = image_to_tensor(self.landscape_path, max_size=256, device=self.device)
        input_image = input_image.clone().detach().requires_grad_(True)

        self.assertTrue(input_image.requires_grad)

        # Forward pass WITHOUT torch.no_grad()
        features = self.extractor(input_image)

        # Dummy multi-layer loss (combining content and style features)
        loss = features["conv4_2"].pow(2).mean() + features["conv1_1"].pow(2).mean()

        # Backward pass
        loss.backward()

        # 1. Gradient MUST reach the input image!
        self.assertIsNotNone(input_image.grad, "Input image gradient must not be None!")
        self.assertEqual(input_image.grad.shape, input_image.shape)
        self.assertTrue(
            torch.isfinite(input_image.grad).all().item(),
            "Input image gradient contains NaN or Inf!"
        )
        grad_norm = input_image.grad.norm().item()
        self.assertTrue(grad_norm > 0.0, f"Input image gradient norm should be > 0, got {grad_norm}")
        print(f"Input image gradient norm: {grad_norm:.6f} (Gradient successfully reached input image)")

        # 2. VGG parameters MUST have no gradients (strictly frozen)
        for name, param in self.extractor.named_parameters():
            self.assertIsNone(
                param.grad,
                f"VGG parameter '{name}' received gradient! VGG weights must remain strictly frozen."
            )
        print("Verified: all VGG weights have grad == None (completely frozen during backprop).")

    def test_08_cpu_device_compatibility(self):
        """Verify extractor works seamlessly on CPU device as well."""
        cpu_extractor = build_vgg19_extractor(checkpoint_dir="checkpoints", device="cpu")
        x_cpu = torch.rand(1, 3, 64, 64, device="cpu", requires_grad=True)
        feats = cpu_extractor(x_cpu)
        self.assertIn("conv4_2", feats)
        self.assertEqual(feats["conv4_2"].device, torch.device("cpu"))
        loss = feats["conv4_2"].sum()
        loss.backward()
        self.assertIsNotNone(x_cpu.grad)
        print("Verified CPU fallback execution and gradient propagation.")


if __name__ == "__main__":
    unittest.main()
