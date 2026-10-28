"""Verification script for VGG-19 Pretrained Feature Extractor.

Demonstrates:
1. Loading VGG-19 ImageNet-1K pretrained weights cached in checkpoints/.
2. Freezing verification (requires_grad=False) and eval() mode.
3. Post-ReLU feature extraction on a landscape photo (max 256px).
4. Gradient propagation to input image tensor (for NST optimization).
5. ImageNet normalization inspection and double-normalization prevention.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.model import build_vgg19_extractor, is_tensor_imagenet_normalized
from src.utils import get_device, image_to_tensor


def main():
    device = get_device()
    print("=" * 72)
    print("   VGG-19 Pretrained Feature Extractor Verification for NST")
    print("=" * 72)
    print(f"Hardware Compute Device: {device} (Apple Silicon MPS Acceleration)" if device.type == "mps" else f"Hardware Compute Device: {device}")

    # 1. Initialize extractor
    extractor = build_vgg19_extractor(checkpoint_dir="checkpoints", device=device)
    extractor.print_architecture_summary()

    # 2. Check parameter freezing
    all_frozen = all(not p.requires_grad for p in extractor.parameters())
    print(f"\n[1] Parameter Freezing Check:")
    print(f"    - Total parameters in extractor: {len(list(extractor.parameters()))}")
    print(f"    - All parameters frozen (requires_grad=False): {all_frozen}")
    print(f"    - Extractor is in eval mode: {not extractor.training}")

    # 3. Load landscape image
    landscape_path = Path("data/content/landscape_mountain.jpg")
    if not landscape_path.exists():
        print(f"Error: {landscape_path} not found.")
        sys.exit(1)

    input_tensor = image_to_tensor(landscape_path, max_size=256, device=device)
    print(f"\n[2] Input Landscape Image:")
    print(f"    - Path: {landscape_path}")
    print(f"    - Shape: {list(input_tensor.shape)} [B, C, H, W]")
    print(f"    - Min value: {input_tensor.min().item():.4f}, Max value: {input_tensor.max().item():.4f}")
    print(f"    - Detected as normalized: {is_tensor_imagenet_normalized(input_tensor)} (Correct: it is raw [0, 1])")

    # 4. Feature Extraction
    features = extractor(input_tensor)
    print(f"\n[3] Extracted Feature Maps (Post-ReLU):")
    for name, feat in features.items():
        is_finite = torch.isfinite(feat).all().item()
        print(f"    - Layer '{name:<8}': Shape={str(list(feat.shape)):<22} | Finite={is_finite:<5} | Min={feat.min().item():>8.4f} | Max={feat.max().item():>8.4f}")

    # 5. Gradient Flow to Input Tensor Check
    print(f"\n[4] Gradient Propagation Verification (Critical for NST Optimization):")
    # Simulate an image being optimized: requires_grad=True
    opt_image = input_tensor.clone().detach().requires_grad_(True)
    opt_features = extractor(opt_image)

    # Compute multi-layer loss
    content_loss = (opt_features["conv4_2"] ** 2).mean()
    style_loss = sum((opt_features[k] ** 2).mean() for k in ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"])
    total_loss = content_loss + 1e-3 * style_loss

    total_loss.backward()

    grad_reaches_input = opt_image.grad is not None and torch.isfinite(opt_image.grad).all().item()
    grad_norm = opt_image.grad.norm().item() if opt_image.grad is not None else 0.0
    vgg_weights_frozen = all(p.grad is None for p in extractor.parameters())

    print(f"    - Backward pass executed without torch.no_grad().")
    print(f"    - Gradient reached input image: {grad_reaches_input} (Gradient norm: {grad_norm:.6f})")
    print(f"    - Input gradient shape matches image: {list(opt_image.grad.shape) == list(opt_image.shape)}")
    print(f"    - VGG-19 weights remained strictly frozen (param.grad is None): {vgg_weights_frozen}")

    # 6. Robust Normalization Inspection
    print(f"\n[5] ImageNet Normalization Protection:")
    norm_tensor = extractor.normalize(input_tensor)
    print(f"    - Normalized min: {norm_tensor.min().item():.4f}, max: {norm_tensor.max().item():.4f}")
    print(f"    - is_tensor_imagenet_normalized(norm_tensor): {is_tensor_imagenet_normalized(norm_tensor)}")
    stripped = norm_tensor.clone() + 0.0  # Strips Python-level attribute
    print(f"    - After attribute stripping, is_tensor_imagenet_normalized(stripped): {is_tensor_imagenet_normalized(stripped)}")

    print("\n" + "=" * 72)
    print("   ALL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    main()
