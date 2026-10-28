"""Comprehensive test suite for NST loss functions and Gram Matrix.

Tests:
1. Gram Matrix computation, shape [B, C, C], symmetry, scale invariance, and gradients.
2. Content Loss computation, exact 0.5 * MSE, zero-loss on identical features, target detachment.
3. Single-layer and Multi-layer Style Loss, layer weighting, and cross-resolution compatibility.
4. Total Variation (TV) loss on flat vs noisy images and gradient propagation.
5. Composite NSTLoss manager: end-to-end integration with VGG19FeatureExtractor on actual images.
"""

import unittest

import torch
import torch.nn.functional as F

from src.loss import (
    ContentLoss,
    NSTLoss,
    SingleLayerStyleLoss,
    StyleLoss,
    TotalVariationLoss,
    gram_matrix,
)
from src.model import build_vgg19_extractor
from src.utils import get_device, image_to_tensor


class TestNSTLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        print(f"\n[Loss Test Setup] Running on compute device: {cls.device}")
        cls.landscape_path = "data/content/landscape_mountain.jpg"
        cls.style_path = "data/style/style_van_gogh.jpg"

    def test_01_gram_matrix_shape_and_symmetry(self):
        """Test Gram matrix output shape [B, C, C] and matrix symmetry (G == G^T)."""
        b, c, h, w = 1, 64, 30, 40
        feat = torch.randn(b, c, h, w, device=self.device, requires_grad=True)

        g = gram_matrix(feat, normalize=True)

        # Check shape
        self.assertEqual(list(g.shape), [b, c, c])

        # Check symmetry: G should equal G.transpose(1, 2)
        diff = (g - g.transpose(1, 2)).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5, msg="Gram matrix must be symmetric!")

        # Check backpropagation
        loss = g.sum()
        loss.backward()
        self.assertIsNotNone(feat.grad)
        self.assertTrue(torch.isfinite(feat.grad).all().item())
        print(f"Gram matrix shape: {list(g.shape)} | Symmetry error: {diff:.8f}")

    def test_02_gram_matrix_scale_invariance(self):
        """Test normalized Gram matrix remains numerically stable regardless of resolution."""
        # Same pattern repeated over different resolutions (e.g. 20x20 vs 80x80)
        c = 32
        feat_small = torch.ones(1, c, 20, 20, device=self.device)
        feat_large = torch.ones(1, c, 80, 80, device=self.device)

        g_small = gram_matrix(feat_small, normalize=True)
        g_large = gram_matrix(feat_large, normalize=True)

        # Because all values are 1.0 and normalized by (C * H * W), values should be identical
        diff = (g_small - g_large).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)
        print(f"Gram normalization scale invariance verified (max diff: {diff:.6e})")

    def test_03_content_loss_zero_and_mse(self):
        """Test ContentLoss yields 0 on identical inputs and matches 0.5 * MSE."""
        target = torch.randn(1, 128, 36, 64, device=self.device)
        content_loss_fn = ContentLoss(target_feature=target)

        # 1. Zero loss when input is identical to target
        loss_zero = content_loss_fn(target)
        self.assertAlmostEqual(loss_zero.item(), 0.0, places=6)

        # 2. Known offset
        offset = 2.0
        input_feat = (target + offset).clone().requires_grad_(True)
        loss_val = content_loss_fn(input_feat)
        # MSE is offset^2 = 4.0; ContentLoss is 0.5 * MSE = 2.0
        expected = 0.5 * (offset ** 2)
        self.assertAlmostEqual(loss_val.item(), expected, places=5)

        # 3. Gradient flows back to input
        loss_val.backward()
        self.assertIsNotNone(input_feat.grad)
        self.assertTrue(torch.isfinite(input_feat.grad).all().item())
        print(f"Content loss exact value check: {loss_val.item():.4f} (expected: {expected:.4f})")

    def test_04_style_loss_different_resolutions(self):
        """Test StyleLoss allows style and content features to have different HxW."""
        # Style feature has H=50, W=70
        style_feat = torch.randn(1, 64, 50, 70, device=self.device)
        # Generated image feature has H=36, W=64
        gen_feat = torch.randn(1, 64, 36, 64, device=self.device, requires_grad=True)

        single_style = SingleLayerStyleLoss(target_feature=style_feat, normalize_gram=True)
        loss = single_style(gen_feat)

        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(gen_feat.grad)
        self.assertEqual(gen_feat.grad.shape, gen_feat.shape)
        print(f"Cross-resolution style loss computed successfully: {loss.item():.6f}")

    def test_05_multi_layer_style_loss_aggregation(self):
        """Test multi-layer style loss aggregates layers according to layer_weights."""
        layer_names = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        channels = [64, 128, 256, 512, 512]

        style_targets = {
            name: torch.randn(1, ch, 32, 32, device=self.device)
            for name, ch in zip(layer_names, channels)
        }
        gen_features = {
            name: torch.randn(1, ch, 32, 32, device=self.device, requires_grad=True)
            for name, ch in zip(layer_names, channels)
        }

        weights = {"conv1_1": 0.1, "conv2_1": 0.2, "conv3_1": 0.3, "conv4_1": 0.2, "conv5_1": 0.2}
        style_loss_module = StyleLoss(target_features=style_targets, layer_weights=weights)

        total_style, breakdown = style_loss_module(gen_features)

        # Verify breakdown has all 5 layers
        self.assertEqual(len(breakdown), 5)

        # Verify manual weighted sum matches total
        manual_sum = sum(weights[k] * breakdown[k] for k in layer_names)
        diff = (total_style - manual_sum).abs().item()
        self.assertAlmostEqual(diff, 0.0, places=5)
        print(f"Multi-layer style loss matches weighted sum: {total_style.item():.6f}")

    def test_06_total_variation_loss_behavior(self):
        """Test TV loss is zero for uniform images and high for noise."""
        tv_fn = TotalVariationLoss(reduction="mean")

        # Flat uniform image (e.g. all 0.5)
        flat_img = torch.full((1, 3, 64, 64), 0.5, device=self.device, requires_grad=True)
        loss_flat = tv_fn(flat_img)
        self.assertEqual(loss_flat.item(), 0.0)

        # High-frequency noise image
        noise_img = torch.rand(1, 3, 64, 64, device=self.device, requires_grad=True)
        loss_noise = tv_fn(noise_img)
        self.assertTrue(loss_noise.item() > 0.1)

        # Test gradient propagation
        loss_noise.backward()
        self.assertIsNotNone(noise_img.grad)
        self.assertTrue(torch.isfinite(noise_img.grad).all().item())
        print(f"TV loss on flat: {loss_flat.item():.4f} | TV loss on noise: {loss_noise.item():.4f}")

    def test_07_end_to_end_nst_loss_with_vgg19(self):
        """End-to-end integration test: VGG19 extractor + NSTLoss on real images."""
        extractor = build_vgg19_extractor(checkpoint_dir="checkpoints", device=self.device)

        # Load landscape content image and Van Gogh style image
        content_img = image_to_tensor(self.landscape_path, max_size=256, device=self.device)
        style_img = image_to_tensor(self.style_path, max_size=256, device=self.device)

        print(f"Content tensor shape: {list(content_img.shape)}")
        print(f"Style tensor shape: {list(style_img.shape)}")

        # Extract target features (under torch.no_grad since targets are constants)
        with torch.no_grad():
            content_target_feat = extractor.get_content_feature(content_img)
            style_target_feats = extractor.get_style_features(style_img)

        # Initialize composite NSTLoss
        nst_loss = NSTLoss(
            content_weight=1.0,
            style_weight=1e6,
            tv_weight=1e-3,
        ).to(self.device)

        nst_loss.set_content_target(content_target_feat)
        nst_loss.set_style_targets(style_target_feats)

        # Initialize image to be optimized (e.g. clone of content with requires_grad=True)
        opt_image = content_img.clone().detach().requires_grad_(True)

        # Forward pass (NO torch.no_grad())
        opt_features = extractor(opt_image)
        total_loss, metrics = nst_loss(opt_features, opt_image)

        # Assert loss is valid and finite
        self.assertTrue(torch.isfinite(total_loss).item())
        self.assertTrue(metrics["total_loss"] > 0)
        self.assertTrue(metrics["content_loss"] >= 0)
        self.assertTrue(metrics["style_loss"] > 0)
        self.assertTrue(metrics["tv_loss"] > 0)

        # Backward pass
        total_loss.backward()

        # Check gradient reached the input image tensor
        self.assertIsNotNone(opt_image.grad)
        self.assertEqual(opt_image.grad.shape, opt_image.shape)
        self.assertTrue(torch.isfinite(opt_image.grad).all().item())
        grad_norm = opt_image.grad.norm().item()
        self.assertTrue(grad_norm > 0.0)

        # Check that all VGG parameters remain completely frozen
        vgg_frozen = all(p.grad is None for p in extractor.parameters())
        self.assertTrue(vgg_frozen)

        print(f"End-to-end NSTLoss Metrics:")
        print(f"  Total Loss:       {metrics['total_loss']:.4f}")
        print(f"  Content Loss:     {metrics['content_loss']:.6f} (Weighted: {metrics['weighted_content']:.6f})")
        print(f"  Style Loss:       {metrics['style_loss']:.6e} (Weighted: {metrics['weighted_style']:.4f})")
        print(f"  TV Loss:          {metrics['tv_loss']:.6f} (Weighted: {metrics['weighted_tv']:.6f})")
        print(f"  Input Grad Norm:  {grad_norm:.6f}")
        print(f"  VGG Frozen:       {vgg_frozen}")


if __name__ == "__main__":
    unittest.main()

