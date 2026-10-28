"""Pretrained VGG-19 Feature Extractor for Neural Style Transfer.

This module implements a custom feature extractor based on VGG-19
with pretrained ImageNet weights (VGG19_Weights.IMAGENET1K_V1).

Architecture & Design Principles:
1. Pretrained Weights:
   - Weights originate from ImageNet-1K pretraining (torchvision VGG19).
   - ALL network parameters are strictly frozen (requires_grad = False).
   - The network runs exclusively in eval() mode.
   - The fully connected classification head (classifier) is discarded.
   - Layers beyond conv5_1 / relu5_1 (index 29) are truncated to save compute & memory.
2. Custom Pipeline:
   - All layer extraction logic, robust normalization inspection,
     gradient propagation to input tensors, and layer selection are
     implemented from scratch without relying on third-party NST wrappers.
3. Layer Mapping (Gatys et al., 2015/2016):
   - Content Representation: conv4_2 (post-ReLU: index 22)
   - Style Representation: conv1_1 (idx 1), conv2_1 (idx 6), conv3_1 (idx 11),
     conv4_1 (idx 20), conv5_1 (idx 29)
   - Consistent Post-ReLU features: We capture activations after the ReLU activation.
   - In-place ReLU disabled: inplace=False is enforced on all ReLUs to prevent
     gradient corruption during input-image optimization.
4. Robust ImageNet Normalization:
   - Exact single normalization guarantee: Automatically inspects tensor value
     distributions so normalization is never duplicated, even if custom tensor
     attributes are stripped by PyTorch operations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

from src.utils import IMAGENET_MEAN, IMAGENET_STD, get_device

# Default layer configuration matching Gatys et al.
DEFAULT_CONTENT_LAYER = "conv4_2"
DEFAULT_STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

# Comprehensive mapping of canonical layer names to indices in vgg19.features.
# NOTE: Each layer name corresponds to the post-ReLU activation for that convolution.
# For example, 'conv1_1' refers to the activation immediately following Conv2d(0) + ReLU(1).
VGG19_LAYER_INDEX_MAP: Dict[str, int] = {
    # Block 1
    "conv1_1": 1,   # features[0]: Conv2d(3, 64),   features[1]: ReLU(inplace=False) -> STYLE 1
    "relu1_1": 1,
    "conv1_2": 3,   # features[2]: Conv2d(64, 64),  features[3]: ReLU(inplace=False)
    "relu1_2": 3,
    "pool1": 4,     # features[4]: MaxPool2d
    # Block 2
    "conv2_1": 6,   # features[5]: Conv2d(64, 128), features[6]: ReLU(inplace=False) -> STYLE 2
    "relu2_1": 6,
    "conv2_2": 8,   # features[7]: Conv2d(128, 128),features[8]: ReLU(inplace=False)
    "relu2_2": 8,
    "pool2": 9,     # features[9]: MaxPool2d
    # Block 3
    "conv3_1": 11,  # features[10]: Conv2d(128, 256), features[11]: ReLU(inplace=False) -> STYLE 3
    "relu3_1": 11,
    "conv3_2": 13,  # features[12]: Conv2d(256, 256), features[13]: ReLU(inplace=False)
    "relu3_2": 13,
    "conv3_3": 15,  # features[14]: Conv2d(256, 256), features[15]: ReLU(inplace=False)
    "relu3_3": 15,
    "conv3_4": 17,  # features[16]: Conv2d(256, 256), features[17]: ReLU(inplace=False)
    "relu3_4": 17,
    "pool3": 18,    # features[18]: MaxPool2d
    # Block 4
    "conv4_1": 20,  # features[19]: Conv2d(256, 512), features[20]: ReLU(inplace=False) -> STYLE 4
    "relu4_1": 20,
    "conv4_2": 22,  # features[21]: Conv2d(512, 512), features[22]: ReLU(inplace=False) -> CONTENT
    "relu4_2": 22,
    "conv4_3": 24,  # features[23]: Conv2d(512, 512), features[24]: ReLU(inplace=False)
    "relu4_3": 24,
    "conv4_4": 26,  # features[25]: Conv2d(512, 512), features[26]: ReLU(inplace=False)
    "relu4_4": 26,
    "pool4": 27,    # features[27]: MaxPool2d
    # Block 5
    "conv5_1": 29,  # features[28]: Conv2d(512, 512), features[29]: ReLU(inplace=False) -> STYLE 5
    "relu5_1": 29,
}

# The maximum layer index needed for standard NST (conv5_1 / relu5_1 at index 29)
MAX_REQUIRED_LAYER_INDEX = 29


def is_tensor_imagenet_normalized(tensor: torch.Tensor) -> bool:
    """Robustly detect whether an image tensor has already undergone ImageNet normalization.

    Inspection Strategy:
    1. Check explicit metadata attribute `_is_imagenet_normalized` if present.
    2. Fallback to numerical value inspection:
       - Raw RGB image tensors strictly reside in [0.0, 1.0].
       - ImageNet normalization subtracts mean ([0.485, 0.456, 0.406]) and divides
         by std ([0.229, 0.224, 0.225]). As a result:
         - Values for dark regions become negative (min reaches ~ -2.11).
         - Values for bright regions exceed 1.0 (max reaches ~ +2.64).
       - If min < -0.05 or max > 1.5, the tensor is undeniably in normalized space.

    This ensures that even when PyTorch operations (cloning, slicing, arithmetic,
    requires_grad_) strip Python-level attributes, double normalization is prevented.

    Args:
        tensor: Image tensor of shape [B, 3, H, W].

    Returns:
        bool: True if tensor is already normalized, False if raw [0, 1].
    """
    if getattr(tensor, "_is_imagenet_normalized", None) is True:
        return True

    if getattr(tensor, "_is_imagenet_normalized", None) is False:
        # Explicitly marked as raw, verify numerical bounds
        if tensor.min().item() >= -0.05 and tensor.max().item() <= 1.5:
            return False

    # Attribute is missing or unverified — inspect numerical range
    t_min = tensor.min().item()
    t_max = tensor.max().item()

    if t_min < -0.05 or t_max > 1.5:
        return True

    return False


class VGG19FeatureExtractor(nn.Module):
    """Frozen VGG-19 Convolutional Feature Extractor for NST.

    Extracts intermediate post-ReLU feature maps for content and style loss.
    All weights are pretrained on ImageNet and completely frozen.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        device: Optional[Union[str, torch.device]] = None,
        content_layer: str = DEFAULT_CONTENT_LAYER,
        style_layers: Optional[Sequence[str]] = None,
    ):
        """Initialize the frozen VGG-19 feature extractor.

        Args:
            checkpoint_dir: Directory where pretrained weights are cached.
            device: Compute device (MPS, CUDA, or CPU). Defaults to get_device().
            content_layer: Layer name used for content representation (default: 'conv4_2').
            style_layers: Sequence of layer names for style representation
                          (default: ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']).
        """
        super().__init__()

        self.device = get_device(device)
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set PyTorch hub directory to cache downloads in checkpoints/
        torch.hub.set_dir(str(self.checkpoint_dir))

        # 1. Load pretrained VGG-19 from ImageNet-1K
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        # 2. Extract only convolutional features up to conv5_1 (index 29)
        # Slicing [:30] includes indices 0 through 29, pruning unused layers (30-36) and classifier.
        truncated_features = vgg.features[: MAX_REQUIRED_LAYER_INDEX + 1]

        # 3. Disable in-place ReLUs to preserve gradient computation for input images
        for module in truncated_features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        # 4. Freeze all parameters strictly (no gradient computation for network weights)
        for param in truncated_features.parameters():
            param.requires_grad = False

        self.features = truncated_features

        # 5. Register ImageNet mean and std buffers for differentiable normalization
        mean_tensor = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

        # Move full module to target device and set to eval mode
        self.to(self.device)
        self.eval()

        # Store layer configurations
        self.content_layer = content_layer
        self.style_layers = list(style_layers or DEFAULT_STYLE_LAYERS)

        # Precompute canonical layer to index mapping
        self._validate_layers([self.content_layer] + self.style_layers)

    def _validate_layers(self, layers: Iterable[str]) -> None:
        """Validate that all requested layers exist in the index map."""
        for layer_name in layers:
            if layer_name not in VGG19_LAYER_INDEX_MAP:
                raise ValueError(
                    f"Unknown layer '{layer_name}'. Available layers: {list(VGG19_LAYER_INDEX_MAP.keys())}"
                )
            if VGG19_LAYER_INDEX_MAP[layer_name] > MAX_REQUIRED_LAYER_INDEX:
                raise ValueError(
                    f"Layer '{layer_name}' (index {VGG19_LAYER_INDEX_MAP[layer_name]}) "
                    f"exceeds maximum retained layer index {MAX_REQUIRED_LAYER_INDEX}."
                )

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable ImageNet normalization using registered buffers.

        Args:
            tensor: Image tensor in [0, 1] range of shape [B, 3, H, W].

        Returns:
            Normalized tensor with mean subtracted and divided by std.
        """
        # Ensure buffers match input device and dtype
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)
        normalized = (tensor - mean) / std
        # Attach attribute as hint (may be lost downstream, handled by is_tensor_imagenet_normalized)
        normalized._is_imagenet_normalized = True  # type: ignore[attr-defined]
        return normalized

    def forward(
        self,
        x: torch.Tensor,
        layers: Optional[Sequence[str]] = None,
        normalize: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract intermediate post-ReLU features from specified VGG-19 layers.

        Args:
            x: Input image tensor of shape [B, 3, H, W]. May be raw [0, 1] or already normalized.
            layers: Specific layer names to extract. If None, extracts union of
                    content_layer and style_layers.
            normalize: Normalization control:
                       - None (default): Auto-detect whether x needs ImageNet normalization.
                         If already normalized, avoids double normalization.
                       - True: Force ImageNet normalization.
                       - False: Do not normalize (assume input is pre-normalized or raw intentionally).

        Returns:
            Dictionary mapping layer names to their corresponding post-ReLU activation tensors.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected 4D image tensor [B, 3, H, W], got shape {list(x.shape)}")

        # Ensure tensor is on the correct device
        if x.device != self.device:
            x = x.to(self.device)

        # 1. ImageNet Normalization Handling
        if normalize is None:
            # Auto-detect: only normalize if not already normalized
            if not is_tensor_imagenet_normalized(x):
                x = self.normalize(x)
        elif normalize is True:
            # Forced normalization
            x = self.normalize(x)
        # If normalize is False: pass x as-is

        # 2. Resolve requested layers and map to features indices
        target_layers = list(layers or ([self.content_layer] + self.style_layers))
        self._validate_layers(target_layers)

        # Map index -> list of layer names (to handle aliases like conv4_2 and relu4_2)
        index_to_names: Dict[int, List[str]] = {}
        for name in target_layers:
            idx = VGG19_LAYER_INDEX_MAP[name]
            index_to_names.setdefault(idx, []).append(name)

        max_idx_needed = max(index_to_names.keys())

        # 3. Sequential forward pass up to highest requested layer
        # NOTE: We do NOT use torch.no_grad() here because when extracting features
        # of the image being optimized, gradients MUST flow back to the input tensor!
        extracted_features: Dict[str, torch.Tensor] = {}
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            if i in index_to_names:
                for name in index_to_names[i]:
                    extracted_features[name] = out
            if i >= max_idx_needed:
                # Early stop: do not execute deeper layers if not needed
                break

        return extracted_features

    def get_content_feature(
        self,
        x: torch.Tensor,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        """Extract the content feature map (conv4_2 / relu4_2).

        Args:
            x: Input image tensor [B, 3, H, W].
            normalize: Normalization control (None=auto, True=force, False=skip).

        Returns:
            Content feature tensor of shape [B, 512, H/8, W/8].
        """
        feats = self.forward(x, layers=[self.content_layer], normalize=normalize)
        return feats[self.content_layer]

    def get_style_features(
        self,
        x: torch.Tensor,
        normalize: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract the style feature maps (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1).

        Args:
            x: Input image tensor [B, 3, H, W].
            normalize: Normalization control (None=auto, True=force, False=skip).

        Returns:
            Dictionary mapping each style layer name to its feature tensor.
        """
        return self.forward(x, layers=self.style_layers, normalize=normalize)

    def print_architecture_summary(self) -> None:
        """Print detailed layer indexing, names, and roles in the truncated feature extractor."""
        print("=" * 72)
        print("VGG-19 Pretrained Truncated Feature Extractor for NST")
        print("=" * 72)
        print(f"Device: {self.device}")
        print(f"Weights: ImageNet-1K (Frozen, requires_grad=False, eval mode)")
        print(f"Content Layer: {self.content_layer} (index {VGG19_LAYER_INDEX_MAP[self.content_layer]})")
        print(f"Style Layers: {', '.join(f'{l} (idx {VGG19_LAYER_INDEX_MAP[l]})' for l in self.style_layers)}")
        print("-" * 72)
        print(f"{'Idx':<4} | {'Layer Module':<35} | {'Role / Tag':<25}")
        print("-" * 72)

        role_tags = {
            1: "STYLE LAYER 1 (conv1_1)",
            6: "STYLE LAYER 2 (conv2_1)",
            11: "STYLE LAYER 3 (conv3_1)",
            20: "STYLE LAYER 4 (conv4_1)",
            22: "CONTENT LAYER (conv4_2)",
            29: "STYLE LAYER 5 (conv5_1)",
        }

        for i, layer in enumerate(self.features):
            layer_str = str(layer)
            if len(layer_str) > 35:
                layer_str = layer_str[:32] + "..."
            tag = role_tags.get(i, "")
            print(f"{i:<4} | {layer_str:<35} | {tag:<25}")
        print("=" * 72)


def build_vgg19_extractor(
    checkpoint_dir: Union[str, Path] = "checkpoints",
    device: Optional[Union[str, torch.device]] = None,
) -> VGG19FeatureExtractor:
    """Factory helper to build and return a ready-to-use VGG19FeatureExtractor.

    Args:
        checkpoint_dir: Path to directory for caching weights.
        device: Target compute device.

    Returns:
        Configured VGG19FeatureExtractor instance.
    """
    return VGG19FeatureExtractor(checkpoint_dir=checkpoint_dir, device=device)
