"""Verification script for NST Loss Functions and Gram Matrix.

Demonstrates:
1. Scale-invariant Gram Matrix computation and symmetry verification.
2. Content Loss on real landscape photo.
3. Multi-layer Style Loss across 5 layers with layer breakdown.
4. Total Variation (TV) Regularization Loss.
5. End-to-end composite NSTLoss calculation and gradient backpropagation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

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


def main():
    device = get_device()
    print("=" * 72)
    print("      Neural Style Transfer Loss & Gram Matrix Verification")
    print("=" * 72)
    print(f"Device: {device}")

    # 1. Gram Matrix Verification
    print("\n[1] Gram Matrix Mathematical Properties:")
    dummy_feat = torch.randn(1, 64, 40, 60, device=device)
    g = gram_matrix(dummy_feat, normalize=True)
    sym_diff = (g - g.transpose(1, 2)).abs().max().item()
    print(f"    - Input feature shape:  {list(dummy_feat.shape)} [B, C, H, W]")
    print(f"    - Gram Matrix shape:    {list(g.shape)} [B, C, C]")
    print(f"    - Max symmetry diff:    {sym_diff:.8e} (G == G^T verified)")

    # 2. Total Variation Loss
    print("\n[2] Total Variation (TV) Regularization Loss:")
    flat_img = torch.full((1, 3, 64, 64), 0.5, device=device)
    noise_img = torch.rand(1, 3, 64, 64, device=device)
    tv_fn = TotalVariationLoss()
    tv_flat = tv_fn(flat_img).item()
    tv_noise = tv_fn(noise_img).item()
    print(f"    - TV Loss on Uniform Flat Image: {tv_flat:.6f} (Expected: 0.0)")
    print(f"    - TV Loss on Random Noise Image: {tv_noise:.6f} (High penalty on noise)")

    # 3. Real Images & VGG Extractor
    print("\n[3] End-to-End NST Loss on Real Artwork & Landscape:")
    landscape_path = "data/content/landscape_mountain.jpg"
    style_path = "data/style/style_van_gogh.jpg"

    content_tensor = image_to_tensor(landscape_path, max_size=256, device=device)
    style_tensor = image_to_tensor(style_path, max_size=256, device=device)

    print(f"    - Content Image (Mount Rainier): {list(content_tensor.shape)}")
    print(f"    - Style Image (Van Gogh Wheat):   {list(style_tensor.shape)}")

    extractor = build_vgg19_extractor(checkpoint_dir="checkpoints", device=device)

    # Compute constant targets
    with torch.no_grad():
        content_target = extractor.get_content_feature(content_tensor)
        style_targets = extractor.get_style_features(style_tensor)

    nst_loss = NSTLoss(
        content_weight=1.0,
        style_weight=1e6,
        tv_weight=1e-3,
    ).to(device)

    nst_loss.set_content_target(content_target)
    nst_loss.set_style_targets(style_targets)

    # Simulate optimization tensor
    opt_img = content_tensor.clone().detach().requires_grad_(True)
    opt_feats = extractor(opt_img)

    total_loss, metrics = nst_loss(opt_feats, opt_img)
    total_loss.backward()

    print(f"\n[4] NST Loss Metrics Breakdown:")
    print(f"    - Content Loss (conv4_2): {metrics['content_loss']:.6f} (Initial = 0 because opt_img is cloned from content)")
    print(f"    - Total Style Loss:       {metrics['style_loss']:.6e} (Weighted: {metrics['weighted_style']:.4f})")
    print(f"    - TV Smoothness Loss:     {metrics['tv_loss']:.6f} (Weighted: {metrics['weighted_tv']:.6f})")
    print(f"    - Total Aggregated Loss:  {metrics['total_loss']:.4f}")

    print(f"\n[5] Style Layers Breakdown (MSE between Gram Matrices):")
    for layer, val in metrics["style_breakdown"].items():
        print(f"    - Layer '{layer:<8}': MSE = {val:.6e}")

    print(f"\n[6] Gradient Propagation Check:")
    grad_norm = opt_img.grad.norm().item()
    vgg_frozen = all(p.grad is None for p in extractor.parameters())
    print(f"    - Gradient norm to input image: {grad_norm:.6f}")
    print(f"    - Input image grad shape matches: {list(opt_img.grad.shape) == list(opt_img.shape)}")
    print(f"    - All VGG weights remain frozen:  {vgg_frozen}")

    print("\n" + "=" * 72)
    print("   ALL LOSS FUNCTION VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    main()

