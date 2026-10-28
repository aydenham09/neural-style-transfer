"""Utility functions for Neural Style Transfer.
Handles device detection, RGB image reading, proportional resizing,
tensor conversions, ImageNet normalization/denormalization with double-normalization
prevention, preview generation, and saving.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, UnidentifiedImageError

# Standard ImageNet statistics (RGB channels)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CorruptedImageError(Exception):
    """Raised when an image file exists but is corrupted or cannot be decoded."""
    pass


class InvalidImageError(Exception):
    """Raised when an image has invalid dimensions or zero channels."""
    pass


class DoubleNormalizationError(ValueError):
    """Raised when attempting to normalize a tensor that is already normalized."""
    pass


def get_device(requested_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Select the best available compute device for this environment.

    Priority:
    1. Explicit requested device (if provided and valid)
    2. CUDA (NVIDIA GPU acceleration)
    3. MPS (Apple Silicon GPU acceleration)
    4. CPU fallback

    Args:
        requested_device: Optional explicit device string or torch.device.

    Returns:
        torch.device instance representing active compute device.
    """
    if requested_device is not None:
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def calculate_proportional_size(
    original_size: Tuple[int, int],
    max_size: int = 256
) -> Tuple[int, int]:
    """Calculate proportional dimensions maintaining aspect ratio with max dimension limit.

    Rule:
    - If max(w, h) <= max_size: Do NOT upscale (keep original dimensions).
    - If max(w, h) > max_size: Scale down so max(w, h) == max_size proportionally.

    Args:
        original_size: Tuple (width, height)
        max_size: Maximum allowed dimension in pixels (default: 256).

    Returns:
        Tuple (new_width, new_height)
    """
    orig_w, orig_h = original_size
    if orig_w <= 0 or orig_h <= 0:
        raise InvalidImageError(f"Invalid original image dimensions: {orig_w}x{orig_h}")

    max_dim = max(orig_w, orig_h)
    if max_dim <= max_size:
        return orig_w, orig_h

    scale = max_size / float(max_dim)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    return new_w, new_h


def load_image(
    image_path: Union[str, Path],
    max_size: int = 256
) -> Image.Image:
    """Read an RGB image from disk, validate it, and resize proportionally up to max_size.

    Args:
        image_path: Path to the image file.
        max_size: Maximum dimension for proportional resizing (default: 256).
                  Images smaller than max_size are not enlarged.

    Returns:
        PIL.Image.Image in RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        CorruptedImageError: If the file is broken or cannot be identified.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path.resolve()}")

    try:
        with Image.open(path) as img:
            # Force load to catch truncated/corrupted files early
            img.load()
            orig_size = img.size
            img_rgb = img.convert("RGB")
    except UnidentifiedImageError as e:
        raise CorruptedImageError(f"Cannot identify or decode image file '{path}': {e}") from e
    except OSError as e:
        raise CorruptedImageError(f"Corrupted or unreadable image file '{path}': {e}") from e
    except Exception as e:
        raise CorruptedImageError(f"Unexpected error while opening image '{path}': {e}") from e

    target_size = calculate_proportional_size(orig_size, max_size=max_size)
    if target_size != orig_size:
        img_rgb = img_rgb.resize(target_size, Image.Resampling.LANCZOS)

    return img_rgb


def image_to_tensor(
    image_or_path: Union[Image.Image, str, Path],
    max_size: int = 256,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """Convert an image or file path to a raw 1x3xHxW PyTorch tensor with values in [0, 1].

    The returned tensor is marked as raw (unnormalized).

    Args:
        image_or_path: PIL Image instance or file path.
        max_size: Maximum dimension for resizing if path is provided.
        device: Target compute device. Defaults to active get_device().

    Returns:
        torch.Tensor of shape [1, 3, H, W] in range [0.0, 1.0], float32.
    """
    if isinstance(image_or_path, (str, Path)):
        img = load_image(image_or_path, max_size=max_size)
    elif isinstance(image_or_path, Image.Image):
        img = image_or_path.convert("RGB")
        target_size = calculate_proportional_size(img.size, max_size=max_size)
        if target_size != img.size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
    else:
        raise TypeError(f"Unsupported image type: {type(image_or_path)}")

    # Convert to float tensor [3, H, W] in [0, 1]
    tensor = TF.to_tensor(img)  # Shape: (3, H, W), float32, [0, 1]
    tensor = tensor.unsqueeze(0)  # Shape: (1, 3, H, W)

    # Attach normalization state flag to distinguish raw vs normalized
    tensor._is_imagenet_normalized = False  # type: ignore[attr-defined]

    target_dev = get_device(device)
    tensor = tensor.to(target_dev)
    # Ensure attribute survives moving device
    tensor._is_imagenet_normalized = False  # type: ignore[attr-defined]

    return tensor


def is_normalized(tensor: torch.Tensor) -> bool:
    """Check if a tensor has already undergone ImageNet normalization.

    Detects via explicit flag attribute `_is_imagenet_normalized` or negative value inspection.

    Args:
        tensor: PyTorch tensor to inspect.

    Returns:
        bool indicating whether tensor is normalized.
    """
    if getattr(tensor, "_is_imagenet_normalized", False) is True:
        return True

    # ImageNet normalized tensors have values with negative numbers (e.g. min around -2.11)
    # whereas raw tensors strictly reside in [0.0, 1.0].
    if tensor.min().item() < -0.05:
        return True

    return False


def normalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a raw [1, 3, H, W] image tensor using ImageNet mean and std.

    Prevents double normalization: raises DoubleNormalizationError if input is already normalized.

    Args:
        tensor: Raw PyTorch image tensor of shape [1, 3, H, W] with values in [0, 1].

    Returns:
        Normalized torch.Tensor with ImageNet statistics and _is_imagenet_normalized=True.

    Raises:
        DoubleNormalizationError: If tensor was already normalized.
        ValueError: If tensor values are outside expected [0, 1] bounds or invalid shape.
    """
    if is_normalized(tensor):
        raise DoubleNormalizationError(
            "Double normalization detected! The tensor has already been normalized with ImageNet statistics."
        )

    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected tensor of shape [B, 3, H, W], got {list(tensor.shape)}")

    # Sanity check range for raw tensor (allowing tiny numerical tolerance)
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    if t_min < -1e-4 or t_max > 1.0 + 1e-4:
        raise ValueError(
            f"Expected raw image tensor values in [0, 1], but got range [{t_min:.4f}, {t_max:.4f}]. "
            "Tensor might be in [0, 255] or already modified."
        )

    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)

    normalized = (tensor - mean) / std
    normalized._is_imagenet_normalized = True  # type: ignore[attr-defined]
    return normalized


def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization, converting back to raw values in [0, 1].

    Args:
        tensor: Normalized PyTorch tensor of shape [1, 3, H, W] (or unnormalized if already in [0, 1]).

    Returns:
        Raw torch.Tensor clamped to [0.0, 1.0] with _is_imagenet_normalized=False.
    """
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected tensor of shape [B, 3, H, W], got {list(tensor.shape)}")

    # If tensor is not normalized, clamp and return copy
    if not is_normalized(tensor):
        raw = torch.clamp(tensor.clone(), 0.0, 1.0)
        raw._is_imagenet_normalized = False  # type: ignore[attr-defined]
        return raw

    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)

    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0.0, 1.0)
    denorm._is_imagenet_normalized = False  # type: ignore[attr-defined]
    return denorm


def tensor_to_image(
    tensor: torch.Tensor,
    is_normalized_tensor: Optional[bool] = None
) -> Image.Image:
    """Convert a PyTorch tensor (1x3xHxW or 3xHxW) to a PIL RGB image for preview.

    Automatically handles denormalization if tensor is normalized.

    Args:
        tensor: Input PyTorch tensor.
        is_normalized_tensor: Optional explicit flag. If None, auto-detected.

    Returns:
        PIL.Image.Image in RGB mode.
    """
    t = tensor.detach().cpu()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    elif t.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {list(t.shape)}")

    # Determine if denormalization is needed
    should_denorm = is_normalized_tensor if is_normalized_tensor is not None else is_normalized(t)
    if should_denorm:
        t = denormalize_imagenet(t)
    else:
        t = torch.clamp(t, 0.0, 1.0)

    # Convert [1, 3, H, W] -> [H, W, 3] uint8
    t_sq = t.squeeze(0).permute(1, 2, 0)
    arr = (t_sq.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_image(
    image_or_tensor: Union[Image.Image, torch.Tensor],
    output_path: Union[str, Path],
    is_normalized_tensor: Optional[bool] = None
) -> str:
    """Save a PIL Image or PyTorch tensor to disk.

    Args:
        image_or_tensor: PIL Image or PyTorch tensor.
        output_path: Destination file path.
        is_normalized_tensor: Flag indicating if tensor is ImageNet normalized.

    Returns:
        Resolved string path of saved image.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_or_tensor, torch.Tensor):
        img = tensor_to_image(image_or_tensor, is_normalized_tensor=is_normalized_tensor)
    elif isinstance(image_or_tensor, Image.Image):
        img = image_or_tensor.convert("RGB")
    else:
        raise TypeError(f"Unsupported type to save: {type(image_or_tensor)}")

    img.save(dest)
    return str(dest.resolve())


def create_dataset_preview(
    content_paths: List[Union[str, Path]],
    style_paths: List[Union[str, Path]],
    output_path: Union[str, Path] = "outputs/dataset_preview.png",
    max_size: int = 256
) -> str:
    """Create a side-by-side grid preview image of content and style datasets.

    Args:
        content_paths: List of content image paths.
        style_paths: List of style image paths.
        output_path: Where to save the preview image.
        max_size: Max resize limit used for NST.

    Returns:
        Path of saved preview plot.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    n_content = len(content_paths)
    n_style = len(style_paths)
    total_cols = max(n_content, n_style)

    fig, axes = plt.subplots(2, total_cols, figsize=(4 * total_cols, 8))
    fig.patch.set_facecolor("#1e1e1e")

    # Row 0: Content images
    for i in range(total_cols):
        ax = axes[0, i] if total_cols > 1 else axes[0]
        if i < n_content:
            p = Path(content_paths[i])
            img = load_image(p, max_size=max_size)
            tensor = image_to_tensor(p, max_size=max_size)
            ax.imshow(img)
            ax.set_title(
                f"Content {i+1}: {p.stem}\nResized: {img.width}x{img.height}\nTensor: {list(tensor.shape)}",
                color="white",
                fontsize=10,
                pad=8
            )
        ax.axis("off")

    # Row 1: Style images
    for i in range(total_cols):
        ax = axes[1, i] if total_cols > 1 else axes[1]
        if i < n_style:
            p = Path(style_paths[i])
            img = load_image(p, max_size=max_size)
            tensor = image_to_tensor(p, max_size=max_size)
            ax.imshow(img)
            ax.set_title(
                f"Style {i+1}: {p.stem}\nResized: {img.width}x{img.height}\nTensor: {list(tensor.shape)}",
                color="white",
                fontsize=10,
                pad=8
            )
        ax.axis("off")

    plt.suptitle(
        f"Neural Style Transfer Dataset Preview (Max {max_size}px, BCHW 1x3xHxW)",
        color="white",
        fontsize=14,
        y=0.98
    )
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(out_file.resolve())

