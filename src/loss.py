"""Loss Functions and Gram Matrix for Neural Style Transfer.

This module provides custom implementations of the core loss functions
formulated by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge in
"A Neural Algorithm of Artistic Style" (2015/2016):

1. Gram Matrix (G = F * F^T):
   Measures channel-wise feature correlations to capture artistic texture,
   brush strokes, and color motifs independently of spatial coordinates.
2. Content Loss:
   Mean Squared Error between feature activations of the content image and
   the generated image at layer conv4_2.
3. Multi-Layer Style Loss:
   Weighted sum of Mean Squared Errors between Gram matrices of the style
   image and the generated image across layers conv1_1, conv2_1, conv3_1,
   conv4_1, and conv5_1.
4. Total Variation (TV) Regularization Loss:
   Penalizes high-frequency noise and spatial dissimilarity between adjacent
   pixels, ensuring visual smoothness and stroke continuity.
5. Composite NSTLoss:
   Unified loss manager computing Total Loss = alpha * Content + beta * Style + gamma * TV.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import DEFAULT_CONTENT_LAYER, DEFAULT_STYLE_LAYERS


def gram_matrix(features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Compute the Gram matrix of a convolutional feature map.

    Mathematical Formulation:
        Let F in R^{C x (H * W)} be the vectorized feature representation at layer l,
        where C is the number of channels (filters) and M = H * W is spatial area.
        The Gram matrix G in R^{C x C} is:
            G_{i, j} = sum_{k=1}^{M} F_{i, k} * F_{j, k} = (F * F^T)_{i, j}

    Normalization:
        When normalize=True, G is divided by (C * H * W) (or H * W).
        This makes the Gram matrix scale-invariant with respect to image resolution,
        allowing content and style images to have completely different dimensions.

    Args:
        features: 4D tensor of shape [B, C, H, W] representing layer activations.
        normalize: Whether to normalize by spatial and channel dimensions (default: True).

    Returns:
        Gram matrix tensor of shape [B, C, C].
    """
    if features.ndim != 4:
        raise ValueError(f"Expected 4D feature tensor [B, C, H, W], got shape {list(features.shape)}")

    b, c, h, w = features.shape

    # Vectorize spatial dimensions: [B, C, H, W] -> [B, C, H * W]
    features_flat = features.view(b, c, h * w)

    # Batch matrix multiplication: [B, C, M] x [B, M, C] -> [B, C, C]
    gram = torch.bmm(features_flat, features_flat.transpose(1, 2))

    if normalize:
        # Normalization factor: total elements in the feature map (C * H * W)
        # Prevents high-dimensional layers (e.g. 512 channels) from dominating numerically
        norm_factor = float(c * h * w)
        gram = gram / norm_factor

    return gram


class ContentLoss(nn.Module):
    """Mean Squared Error Content Loss for NST.

    Measures semantic layout dissimilarity between generated image features
    and target content image features at a specific convolutional layer (conv4_2).

    Formula:
        L_{content} = 0.5 * MSE(F_{gen}, P_{content})
    """

    def __init__(self, target_feature: Optional[torch.Tensor] = None):
        """Initialize ContentLoss.

        Args:
            target_feature: Optional target feature tensor [B, C, H, W].
                            If provided, it is automatically detached from autograd.
        """
        super().__init__()
        self.target: Optional[torch.Tensor] = None
        if target_feature is not None:
            self.set_target(target_feature)

    def set_target(self, target_feature: torch.Tensor) -> None:
        """Set and freeze the target content feature map.

        Args:
            target_feature: Feature map tensor from content image.
        """
        # Detach target strictly so no gradients flow to target
        self.target = target_feature.detach().clone()

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """Compute content loss between input feature and stored target.

        Args:
            input_feature: Feature map tensor from the image being optimized.

        Returns:
            Scalar content loss tensor.
        """
        if self.target is None:
            raise RuntimeError("Content target has not been set. Call set_target() first.")

        if input_feature.shape != self.target.shape:
            raise ValueError(
                f"Feature shape mismatch: input {list(input_feature.shape)} "
                f"vs target {list(self.target.shape)}"
            )

        # 0.5 factor matches Gatys et al. analytical formulation
        return 0.5 * F.mse_loss(input_feature, self.target)


class SingleLayerStyleLoss(nn.Module):
    """Style Loss for a single convolutional layer using Gram Matrix.

    Measures texture/correlation dissimilarity:
        E_l = MSE(G_{gen}^l, A_{style}^l)
    """

    def __init__(
        self,
        target_feature: Optional[torch.Tensor] = None,
        normalize_gram: bool = True,
    ):
        """Initialize single-layer style loss.

        Args:
            target_feature: Feature map from style image [B, C, H_s, W_s].
            normalize_gram: Whether to normalize the Gram matrix by spatial size.
        """
        super().__init__()
        self.normalize_gram = normalize_gram
        self.target_gram: Optional[torch.Tensor] = None
        if target_feature is not None:
            self.set_target(target_feature)

    def set_target(self, target_feature: torch.Tensor) -> None:
        """Compute and store target Gram matrix from style features."""
        with torch.no_grad():
            self.target_gram = gram_matrix(target_feature, normalize=self.normalize_gram).detach()

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """Compute style loss for this layer.

        Args:
            input_feature: Feature map from image being optimized [B, C, H, W].

        Returns:
            Scalar style loss tensor.
        """
        if self.target_gram is None:
            raise RuntimeError("Style target Gram matrix has not been set. Call set_target() first.")

        input_gram = gram_matrix(input_feature, normalize=self.normalize_gram)

        if input_gram.shape != self.target_gram.shape:
            raise ValueError(
                f"Gram matrix shape mismatch: input {list(input_gram.shape)} "
                f"vs target {list(self.target_gram.shape)}"
            )

        return F.mse_loss(input_gram, self.target_gram)


class StyleLoss(nn.Module):
    """Multi-Layer Style Loss for NST.

    Aggregates style losses across multiple convolutional layers with configurable weights:
        L_{style} = sum_{l in Layers} w_l * E_l
    """

    DEFAULT_LAYER_WEIGHTS: Dict[str, float] = {
        "conv1_1": 0.2,
        "conv2_1": 0.2,
        "conv3_1": 0.2,
        "conv4_1": 0.2,
        "conv5_1": 0.2,
    }

    def __init__(
        self,
        target_features: Optional[Dict[str, torch.Tensor]] = None,
        layer_weights: Optional[Dict[str, float]] = None,
        normalize_gram: bool = True,
    ):
        """Initialize Multi-Layer Style Loss.

        Args:
            target_features: Optional dictionary mapping layer name to style feature tensor.
            layer_weights: Optional dictionary mapping layer name to weighting coefficient w_l.
                           Defaults to equal weighting (0.2 for each of 5 layers).
            normalize_gram: Whether Gram matrices should be normalized.
        """
        super().__init__()
        self.layer_weights = dict(layer_weights or self.DEFAULT_LAYER_WEIGHTS)
        self.normalize_gram = normalize_gram

        # Create submodules for each style layer
        self.layer_losses = nn.ModuleDict({
            layer_name: SingleLayerStyleLoss(normalize_gram=normalize_gram)
            for layer_name in self.layer_weights.keys()
        })

        if target_features is not None:
            self.set_targets(target_features)

    def set_targets(self, target_features: Dict[str, torch.Tensor]) -> None:
        """Set style target features for all monitored layers.

        Args:
            target_features: Dict mapping layer name to style feature tensor.
        """
        for layer_name, loss_module in self.layer_losses.items():
            if layer_name not in target_features:
                raise KeyError(
                    f"Style target features missing monitored layer '{layer_name}'. "
                    f"Available: {list(target_features.keys())}"
                )
            loss_module.set_target(target_features[layer_name])

    def forward(
        self,
        input_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute aggregated multi-layer style loss.

        Args:
            input_features: Dict mapping layer name to current generated feature tensor.

        Returns:
            Tuple of:
            - total_style_loss: Weighted scalar style loss tensor.
            - layer_breakdown: Dict mapping each layer name to its individual loss tensor.
        """
        total_loss = torch.tensor(0.0, device=next(iter(input_features.values())).device)
        layer_breakdown: Dict[str, torch.Tensor] = {}

        for layer_name, weight in self.layer_weights.items():
            if layer_name not in input_features:
                raise KeyError(
                    f"Monitored style layer '{layer_name}' not found in input features. "
                    f"Available: {list(input_features.keys())}"
                )
            layer_loss = self.layer_losses[layer_name](input_features[layer_name])
            layer_breakdown[layer_name] = layer_loss
            total_loss = total_loss + weight * layer_loss

        return total_loss, layer_breakdown


class TotalVariationLoss(nn.Module):
    """Total Variation (TV) Regularization Loss.

    Computes spatial gradient penalties between neighboring pixels to enforce
    smoothness and suppress high-frequency noise / checkerboard artifacts:
        L_{tv} = (1 / N) * sum ( (x_{i+1, j} - x_{i, j})^2 + (x_{i, j+1} - x_{i, j})^2 )
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize TV Loss.

        Args:
            reduction: 'mean' (normalized by dimensions) or 'sum'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute TV loss on an image tensor.

        Args:
            x: Image tensor [B, C, H, W] (raw or normalized).

        Returns:
            Scalar total variation loss tensor.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape {list(x.shape)}")

        b, c, h, w = x.shape
        if h <= 1 or w <= 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Difference between vertically adjacent pixels
        diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2)
        # Difference between horizontally adjacent pixels
        diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2)

        if self.reduction == "mean":
            loss_h = diff_h.sum() / (b * c * (h - 1) * w)
            loss_w = diff_w.sum() / (b * c * h * (w - 1))
            return loss_h + loss_w
        elif self.reduction == "sum":
            return diff_h.sum() + diff_w.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class NSTLoss(nn.Module):
    """Composite Neural Style Transfer Loss Manager.

    Combines Content Loss, Multi-Layer Style Loss, and Total Variation Loss:
        L_{total} = alpha * L_{content} + beta * L_{style} + gamma * L_{tv}
    """

    def __init__(
        self,
        content_weight: float = 1.0,
        style_weight: float = 1e6,
        tv_weight: float = 1e-4,
        content_layer: str = DEFAULT_CONTENT_LAYER,
        style_layer_weights: Optional[Dict[str, float]] = None,
        normalize_gram: bool = True,
    ):
        """Initialize Composite NST Loss.

        Args:
            content_weight: alpha weighting coefficient (default: 1.0).
            style_weight: beta weighting coefficient (default: 1.0e6).
            tv_weight: gamma weighting coefficient (default: 1.0e-4).
            content_layer: Name of content layer (default: 'conv4_2').
            style_layer_weights: Dict of style layer names and weights.
            normalize_gram: Whether to normalize Gram matrices.
        """
        super().__init__()
        self.content_weight = float(content_weight)
        self.style_weight = float(style_weight)
        self.tv_weight = float(tv_weight)
        self.content_layer = content_layer

        self.content_loss_fn = ContentLoss()
        self.style_loss_fn = StyleLoss(
            layer_weights=style_layer_weights,
            normalize_gram=normalize_gram,
        )
        self.tv_loss_fn = TotalVariationLoss(reduction="mean")

    def set_content_target(self, content_feature: torch.Tensor) -> None:
        """Set the target content feature map."""
        self.content_loss_fn.set_target(content_feature)

    def set_style_targets(self, style_features: Dict[str, torch.Tensor]) -> None:
        """Set the target style features for all monitored layers."""
        self.style_loss_fn.set_targets(style_features)

    def forward(
        self,
        generated_features: Dict[str, torch.Tensor],
        generated_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Union[float, torch.Tensor]]]:
        """Compute the weighted total NST loss.

        Args:
            generated_features: Dict of extracted feature maps for the generated image.
            generated_image: The generated image tensor [B, 3, H, W] for TV regularization.

        Returns:
            Tuple of:
            - total_loss: Weighted scalar total loss tensor for backpropagation.
            - metrics: Dict containing unweighted and weighted loss components for tracking.
        """
        # 1. Content Loss
        content_feat = generated_features[self.content_layer]
        raw_content_loss = self.content_loss_fn(content_feat)
        weighted_content_loss = self.content_weight * raw_content_loss

        # 2. Style Loss
        raw_style_loss, style_breakdown = self.style_loss_fn(generated_features)
        weighted_style_loss = self.style_weight * raw_style_loss

        # 3. Total Variation Loss (optional)
        if generated_image is not None and self.tv_weight > 0.0:
            raw_tv_loss = self.tv_loss_fn(generated_image)
            weighted_tv_loss = self.tv_weight * raw_tv_loss
        else:
            raw_tv_loss = torch.tensor(0.0, device=content_feat.device)
            weighted_tv_loss = torch.tensor(0.0, device=content_feat.device)

        total_loss = weighted_content_loss + weighted_style_loss + weighted_tv_loss

        metrics: Dict[str, Union[float, torch.Tensor]] = {
            "total_loss": total_loss.item(),
            "content_loss": raw_content_loss.item(),
            "style_loss": raw_style_loss.item(),
            "tv_loss": raw_tv_loss.item(),
            "weighted_content": weighted_content_loss.item(),
            "weighted_style": weighted_style_loss.item(),
            "weighted_tv": weighted_tv_loss.item(),
            "style_breakdown": {k: v.item() for k, v in style_breakdown.items()},
        }

        return total_loss, metrics

