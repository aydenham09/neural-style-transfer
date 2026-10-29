"""Neural Style Transfer (NST) Optimization Engine.

Implements the iterative pixel-optimization algorithm formulated by
Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge (2015/2016).

Core Architecture & Contract:
1. Target Image as Parameter:
   - The weights of VGG-19 remain strictly frozen throughout optimization.
   - The image itself is the parameter tensor being optimized:
     target_image = content_tensor.clone().detach().requires_grad_(True)
2. Normalization Contract:
   - target_image is maintained in raw RGB float space [0.0, 1.0].
   - Normalization via ImageNet statistics ((x - mean) / std) occurs
     exactly once inside VGG19FeatureExtractor.forward().
   - After each optimizer step, pixels are clamped to [0.0, 1.0] in-place
     under torch.no_grad() without denormalization overhead.
3. Feature Target Freezing:
   - Reference content features (conv4_2) and style Gram matrices
     (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1) are computed once
     under torch.no_grad() and detached as constant targets.
4. Numerical Stability & Diagnostics:
   - Strict monitoring of loss values and gradient norms for NaN/Inf.
   - Hardware-synchronized timing (torch.cuda.synchronize() when CUDA is active).
   - SHA-256 parameter hashing verifying zero drift in frozen VGG weights.
5. Tracking & Artifact Generation:
   - Initial image, periodic progress checkpoints (every 50 steps), final stylized image.
   - Dual loss plots: raw loss components and weighted total loss.
   - Side-by-side comparison grid (Content - Style - Result).
   - Complete configuration and metrics JSON artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.loss import NSTLoss
from src.model import VGG19FeatureExtractor, build_vgg19_extractor
from src.utils import (
    calculate_proportional_size,
    get_device,
    image_to_tensor,
    load_image,
    save_image,
    tensor_to_image,
)


def compute_vgg_checksum(model: nn.Module) -> str:
    """Compute SHA-256 checksum across all parameter tensors to verify immutability."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for param in model.parameters():
            hasher.update(param.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def compute_file_hash(path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_tensor_hash(tensor: torch.Tensor) -> str:
    """Compute SHA-256 hash of a PyTorch tensor's memory buffer."""
    hasher = hashlib.sha256()
    hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


class StyleTransferOptimizer:
    """Iterative Neural Style Transfer Optimization Engine using Adam."""

    def __init__(
        self,
        content_path: Union[str, Path],
        style_path: Union[str, Path],
        output_dir: Union[str, Path],
        max_size: int = 256,
        num_steps: int = 300,
        lr: float = 0.02,
        content_weight: float = 1.0,
        style_weight: float = 1e5,
        tv_weight: float = 0.0,
        save_progress_every: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        extractor: Optional[VGG19FeatureExtractor] = None,
        progress_callback: Optional[Any] = None,
    ):
        """Initialize the NST optimization pipeline.

        Args:
            content_path: Path to content landscape image.
            style_path: Path to style reference painting.
            output_dir: Output folder for this specific run.
            max_size: Maximum resolution dimension in pixels (default: 256).
            num_steps: Total Adam optimization steps (default: 300).
            lr: Adam learning rate (default: 0.02).
            content_weight: alpha weighting factor (default: 1.0).
            style_weight: beta weighting factor (default: 1e5).
            tv_weight: gamma Total Variation regularization factor (default: 0.0).
            save_progress_every: Checkpoint saving cadence (default: 50).
            device: Explicit compute device (defaults to auto-detected get_device()).
            checkpoint_dir: Directory for cached VGG-19 weights.
            extractor: Pre-instantiated and cached VGG19FeatureExtractor (optional).
            progress_callback: Callable(step, total_steps, metrics_dict) for UI updates.
        """
        self.content_path = Path(content_path)
        self.style_path = Path(style_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_size = int(max_size)
        self.num_steps = int(num_steps)
        self.lr = float(lr)
        self.content_weight = float(content_weight)
        self.style_weight = float(style_weight)
        self.tv_weight = float(tv_weight)
        self.save_progress_every = int(save_progress_every)

        self.device = get_device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cached_extractor = extractor
        self.progress_callback = progress_callback

        # Validate input paths
        if not self.content_path.is_file():
            raise FileNotFoundError(f"Content image not found: {self.content_path}")
        if not self.style_path.is_file():
            raise FileNotFoundError(f"Style image not found: {self.style_path}")

        # Subdirectory for intermediate progress images
        self.progress_dir = self.output_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    def _sync_device(self) -> None:
        """Synchronize hardware device stream for precise profiling."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif hasattr(torch, "mps") and hasattr(torch.mps, "synchronize") and self.device.type == "mps":
            torch.mps.synchronize()

    def run(self) -> Dict[str, Any]:
        """Execute the end-to-end Neural Style Transfer optimization loop.

        Returns:
            Dictionary containing run summary metrics, configuration, and verification results.
        """
        print("=" * 76)
        print("   NEURAL STYLE TRANSFER (NST) OPTIMIZATION ENGINE")
        print("=" * 76)
        print(f"Device:           {self.device} ({torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU / Accelerator'})")
        print(f"Content Image:    {self.content_path.name}")
        print(f"Style Reference:  {self.style_path.name}")
        print(f"Resolution Limit: Max {self.max_size}px")
        print(f"Optimization:     Adam (lr={self.lr}, steps={self.num_steps})")
        print(f"Weights:          content={self.content_weight}, style={self.style_weight:.1e}, tv={self.tv_weight}")
        print(f"Output Directory: {self.output_dir}")
        print("-" * 76)

        # 1. Load VGG-19 Pretrained Extractor
        print("\n[Step 1/5] Initializing Frozen VGG-19 Feature Extractor...")
        if self.cached_extractor is not None:
            extractor = self.cached_extractor
        else:
            extractor = build_vgg19_extractor(checkpoint_dir=self.checkpoint_dir, device=self.device)
        vgg_checksum_before = compute_vgg_checksum(extractor)
        print(f"  VGG-19 Parameters Frozen: {all(not p.requires_grad for p in extractor.parameters())}")
        print(f"  VGG-19 Checksum (Before): {vgg_checksum_before[:16]}...")

        # 2. Load Content and Style Images
        print("\n[Step 2/5] Loading and Preprocessing Input Images...")
        content_tensor = image_to_tensor(self.content_path, max_size=self.max_size, device=self.device)
        style_tensor = image_to_tensor(self.style_path, max_size=self.max_size, device=self.device)

        content_file_hash = compute_file_hash(self.content_path)
        style_file_hash = compute_file_hash(self.style_path)
        content_tensor_hash = compute_tensor_hash(content_tensor)
        style_tensor_hash = compute_tensor_hash(style_tensor)

        _, _, c_h, c_w = content_tensor.shape
        _, _, s_h, s_w = style_tensor.shape
        print(f"  Content Tensor: {list(content_tensor.shape)} (range [{content_tensor.min():.3f}, {content_tensor.max():.3f}])")
        print(f"  Content SHA-256 (File):   {content_file_hash[:16]}...")
        print(f"  Content SHA-256 (Tensor): {content_tensor_hash[:16]}...")
        print(f"  Style Tensor:   {list(style_tensor.shape)} (range [{style_tensor.min():.3f}, {style_tensor.max():.3f}])")
        print(f"  Style SHA-256 (File):     {style_file_hash[:16]}...")
        print(f"  Style SHA-256 (Tensor):   {style_tensor_hash[:16]}...")

        # 3. Precompute Reference Features ONCE without Gradient
        print("\n[Step 3/5] Precomputing Reference Target Features (Single Forward Pass)...")
        with torch.no_grad():
            content_target_feat = extractor.get_content_feature(content_tensor)
            style_target_feats = extractor.get_style_features(style_tensor)

        print(f"  Content target feature (conv4_2): shape {list(content_target_feat.shape)}")
        print(f"  Style target layers ({len(style_target_feats)} layers): {list(style_target_feats.keys())}")

        # Initialize Composite Loss Manager
        nst_loss = NSTLoss(
            content_weight=self.content_weight,
            style_weight=self.style_weight,
            tv_weight=self.tv_weight,
        ).to(self.device)
        nst_loss.set_content_target(content_target_feat)
        nst_loss.set_style_targets(style_target_feats)

        # 4. Initialize Target Image from Content Copy
        print("\n[Step 4/5] Initializing Target Image and Adam Optimizer...")
        # Contract: target_image is a detached copy of content requiring gradients
        target_image = content_tensor.clone().detach().requires_grad_(True)

        # Save step 0 (initial image)
        initial_img_path = self.output_dir / "step_0000_initial.png"
        save_image(target_image, initial_img_path, is_normalized_tensor=False)
        print(f"  Initial target image saved to: {initial_img_path}")

        # Initialize Adam Optimizer
        optimizer = torch.optim.Adam([target_image], lr=self.lr)

        # 5. Iterative Optimization Loop
        print("\n[Step 5/5] Running Adam Optimization Loop...")
        print("-" * 76)
        print(f"{'Step':<6} | {'Total Loss':<12} | {'Content (Raw)':<14} | {'Style (Raw)':<14} | {'Grad Norm':<10} | {'Time/step':<10}")
        print("-" * 76)

        history: List[Dict[str, Any]] = []
        early_stopped = False
        stop_reason = ""

        # Synchronize and start wall-clock timer
        self._sync_device()
        start_time = time.perf_counter()

        for step in range(1, self.num_steps + 1):
            step_start = time.perf_counter()

            # Zero optimizer gradients
            optimizer.zero_grad()

            # Forward pass through frozen extractor
            # target_image is RGB [0, 1]; extractor normalizes internally
            target_features = extractor(target_image)

            # Compute composite NST loss
            total_loss, metrics = nst_loss(target_features, target_image)

            # Numerical Sanity Check: Loss must be finite
            if not torch.isfinite(total_loss):
                early_stopped = True
                stop_reason = f"Non-finite loss encountered at step {step}: total_loss={total_loss.item()}"
                print(f"\n[ALERT] Early stopping: {stop_reason}")
                break

            # Backward pass
            total_loss.backward()

            # Numerical Sanity Check: Gradient must be finite
            if target_image.grad is None:
                early_stopped = True
                stop_reason = f"Target image gradient is None at step {step}"
                print(f"\n[ALERT] Early stopping: {stop_reason}")
                break

            if not torch.isfinite(target_image.grad).all():
                early_stopped = True
                stop_reason = f"Non-finite gradients (NaN/Inf) detected at step {step}"
                print(f"\n[ALERT] Early stopping: {stop_reason}")
                break

            grad_norm = target_image.grad.norm().item()

            # Optimizer Step
            optimizer.step()

            # Contractual pixel clamping to [0.0, 1.0] under torch.no_grad()
            with torch.no_grad():
                target_image.clamp_(0.0, 1.0)

            # Profiling step time
            self._sync_device()
            step_time_ms = (time.perf_counter() - step_start) * 1000.0

            # Record metrics history
            step_record = {
                "step": step,
                "total_loss": float(metrics["total_loss"]),
                "content_loss": float(metrics["content_loss"]),
                "style_loss": float(metrics["style_loss"]),
                "tv_loss": float(metrics["tv_loss"]),
                "weighted_content": float(metrics["weighted_content"]),
                "weighted_style": float(metrics["weighted_style"]),
                "weighted_tv": float(metrics["weighted_tv"]),
                "grad_norm": float(grad_norm),
                "step_time_ms": float(step_time_ms),
            }
            history.append(step_record)

            # Console logging
            if step == 1 or step % 25 == 0 or step == self.num_steps:
                print(
                    f"{step:<6d} | "
                    f"{metrics['total_loss']:<12.4e} | "
                    f"{metrics['content_loss']:<14.6f} | "
                    f"{metrics['style_loss']:<14.6e} | "
                    f"{grad_norm:<10.4f} | "
                    f"{step_time_ms:<8.1f} ms"
                )

            # Save intermediate progress checkpoint every `save_progress_every` steps
            if step % self.save_progress_every == 0 or step == self.num_steps:
                progress_path = self.progress_dir / f"step_{step:04d}.png"
                save_image(target_image, progress_path, is_normalized_tensor=False)

            # Optional UI progress callback
            if self.progress_callback is not None:
                try:
                    self.progress_callback(step, self.num_steps, metrics)
                except Exception:
                    pass

        # End of optimization profiling
        self._sync_device()
        total_runtime_sec = time.perf_counter() - start_time
        print("-" * 76)

        # 6. Evaluate Exact Post-Update Metrics on Final Clamped Image
        # Ensures metrics in JSON and logs strictly match the saved final image
        with torch.no_grad():
            final_features = extractor(target_image)
            final_total_loss, final_loss_dict = nst_loss(final_features, target_image)

        final_post_metrics = {
            "step": len(history),
            "total_loss": float(final_total_loss.item()),
            "content_loss": float(final_loss_dict["content_loss"]),
            "style_loss": float(final_loss_dict["style_loss"]),
            "tv_loss": float(final_loss_dict["tv_loss"]),
            "weighted_content": float(final_loss_dict["weighted_content"]),
            "weighted_style": float(final_loss_dict["weighted_style"]),
            "weighted_tv": float(final_loss_dict["weighted_tv"]),
            "grad_norm": history[-1]["grad_norm"] if history else 0.0,
            "step_time_ms": history[-1]["step_time_ms"] if history else 0.0,
            "style_breakdown": {k: float(v) for k, v in final_loss_dict.get("style_breakdown", {}).items()},
        }

        # 7. Verify VGG Weights Remain Strictly Unchanged
        vgg_checksum_after = compute_vgg_checksum(extractor)
        vgg_drift_detected = (vgg_checksum_before != vgg_checksum_after)
        print(f"\n[Verification] VGG-19 Weight Freezing Check:")
        print(f"  Checksum Before: {vgg_checksum_before}")
        print(f"  Checksum After:  {vgg_checksum_after}")
        print(f"  Weights Identical: {not vgg_drift_detected} (Zero Drift: Pass)")

        if vgg_drift_detected:
            raise RuntimeError("CRITICAL ERROR: VGG-19 parameter weights changed during optimization!")

        # 8. Save Final Stylized Result
        final_image_path = self.output_dir / "final_stylized.png"
        save_image(target_image, final_image_path, is_normalized_tensor=False)
        print(f"\nFinal stylized image saved to: {final_image_path}")

        # Compute Quantized PNG Metrics (evaluating impact of 8-bit uint8 quantization from disk)
        with torch.no_grad():
            quantized_tensor = image_to_tensor(final_image_path, max_size=self.max_size, device=self.device)
            quantized_features = extractor(quantized_tensor)
            quantized_total_loss, quantized_loss_dict = nst_loss(quantized_features, quantized_tensor)

        quantized_png_metrics = {
            "total_loss": float(quantized_total_loss.item()),
            "content_loss": float(quantized_loss_dict["content_loss"]),
            "style_loss": float(quantized_loss_dict["style_loss"]),
            "tv_loss": float(quantized_loss_dict["tv_loss"]),
            "weighted_content": float(quantized_loss_dict["weighted_content"]),
            "weighted_style": float(quantized_loss_dict["weighted_style"]),
            "weighted_tv": float(quantized_loss_dict["weighted_tv"]),
            "quantization_delta_total_loss": float(quantized_total_loss.item() - final_total_loss.item()),
            "quantization_delta_content_loss": float(quantized_loss_dict["content_loss"] - final_loss_dict["content_loss"]),
            "quantization_delta_style_loss": float(quantized_loss_dict["style_loss"] - final_loss_dict["style_loss"]),
        }
        print(
            f"[Quantization Analysis] Float Loss: {final_total_loss.item():.4f} | "
            f"PNG uint8 Loss: {quantized_total_loss.item():.4f} | "
            f"Delta: {quantized_png_metrics['quantization_delta_total_loss']:+.6f}"
        )

        # 9. Generate Visual Comparison Grid (Content - Style - Result)
        comparison_path = self.output_dir / "comparison_content_style_result.png"
        self._generate_comparison_grid(
            content_path=self.content_path,
            style_path=self.style_path,
            result_tensor=target_image,
            output_path=comparison_path,
            history=history,
            final_loss=final_post_metrics["total_loss"],
        )
        print(f"Comparison grid saved to:      {comparison_path}")

        # 10. Generate Loss History Plots (Raw and Weighted)
        loss_plot_path = self.output_dir / "loss_history.png"
        self._generate_loss_plots(history=history, output_path=loss_plot_path)
        print(f"Loss history plot saved to:    {loss_plot_path}")

        # 11. Compile Configuration and Metrics JSON
        initial_metrics = history[0] if history else {}
        avg_step_time_ms = total_runtime_sec / len(history) * 1000.0 if history else 0.0

        summary = {
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "os": os.name,
                "platform": sys.platform,
                "python_version": sys.version.split()[0],
                "pytorch_version": torch.__version__,
                "device": str(self.device),
                "device_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            },
            "input_hashes": {
                "content_file_sha256": content_file_hash,
                "style_file_sha256": style_file_hash,
                "content_tensor_sha256": content_tensor_hash,
                "style_tensor_sha256": style_tensor_hash,
            },
            "configuration": {
                "content_image": str(self.content_path),
                "style_image": str(self.style_path),
                "output_dir": str(self.output_dir),
                "max_size": self.max_size,
                "content_dimensions": [c_w, c_h],
                "style_dimensions": [s_w, s_h],
                "optimizer": "Adam",
                "learning_rate": self.lr,
                "num_steps": self.num_steps,
                "content_weight": self.content_weight,
                "style_weight": self.style_weight,
                "tv_weight": self.tv_weight,
                "save_progress_every": self.save_progress_every,
            },
            "execution": {
                "total_runtime_seconds": round(total_runtime_sec, 3),
                "avg_step_time_ms": round(avg_step_time_ms, 2),
                "steps_completed": len(history),
                "early_stopped": early_stopped,
                "stop_reason": stop_reason,
                "vgg_weights_frozen_verified": not vgg_drift_detected,
                "vgg_checksum_before": vgg_checksum_before,
                "vgg_checksum_after": vgg_checksum_after,
            },
            "initial_metrics": initial_metrics,
            "final_metrics": final_post_metrics,
            "final_metrics_quantized_png": quantized_png_metrics,
            "final_metrics_pre_update_step300": history[-1] if history else {},
            "metrics_summary": {
                "initial_total_loss": initial_metrics.get("total_loss"),
                "final_total_loss": final_post_metrics.get("total_loss"),
                "final_total_loss_float": final_post_metrics.get("total_loss"),
                "final_total_loss_quantized_png": quantized_png_metrics.get("total_loss"),
                "quantization_delta_total_loss": quantized_png_metrics.get("quantization_delta_total_loss"),
                "total_loss_reduction_pct": (
                    round((1.0 - final_post_metrics.get("total_loss", 0) / initial_metrics["total_loss"]) * 100, 2)
                    if initial_metrics.get("total_loss", 0) > 0 else 0.0
                ),
                "initial_style_loss": initial_metrics.get("style_loss"),
                "final_style_loss": final_post_metrics.get("style_loss"),
                "final_style_loss_float": final_post_metrics.get("style_loss"),
                "final_style_loss_quantized_png": quantized_png_metrics.get("style_loss"),
                "quantization_delta_style_loss": quantized_png_metrics.get("quantization_delta_style_loss"),
                "style_loss_reduction_pct": (
                    round((1.0 - final_post_metrics.get("style_loss", 0) / initial_metrics["style_loss"]) * 100, 2)
                    if initial_metrics.get("style_loss", 0) > 0 else 0.0
                ),
                "initial_content_loss": initial_metrics.get("content_loss"),
                "final_content_loss": final_post_metrics.get("content_loss"),
                "final_content_loss_float": final_post_metrics.get("content_loss"),
                "final_content_loss_quantized_png": quantized_png_metrics.get("content_loss"),
                "quantization_delta_content_loss": quantized_png_metrics.get("quantization_delta_content_loss"),
            },
            "artifacts": {
                "initial_image": str(initial_img_path),
                "final_image": str(final_image_path),
                "comparison_grid": str(comparison_path),
                "loss_plot": str(loss_plot_path),
                "progress_folder": str(self.progress_dir),
            },
            "history": history,
        }

        json_path = self.output_dir / "run_config_and_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics & config JSON saved to: {json_path}")
        print("=" * 76)

        return summary

    def _generate_comparison_grid(
        self,
        content_path: Path,
        style_path: Path,
        result_tensor: torch.Tensor,
        output_path: Path,
        history: List[Dict[str, Any]],
        final_loss: Optional[float] = None,
    ) -> None:
        """Create a side-by-side presentation figure: Content, Style, and Final Result."""
        content_img = load_image(content_path, max_size=self.max_size)
        style_img = load_image(style_path, max_size=self.max_size)
        result_img = tensor_to_image(result_tensor, is_normalized_tensor=False)

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.patch.set_facecolor("#121212")

        # 1. Content
        axes[0].imshow(content_img)
        axes[0].set_title(
            f"1. Content Landscape\n{content_path.stem}\n({content_img.width}x{content_img.height} px)",
            color="#E0E0E0",
            fontsize=12,
            pad=10,
        )
        axes[0].axis("off")

        # 2. Style
        axes[1].imshow(style_img)
        axes[1].set_title(
            f"2. Style Reference\n{style_path.stem}\n({style_img.width}x{style_img.height} px)",
            color="#E0E0E0",
            fontsize=12,
            pad=10,
        )
        axes[1].axis("off")

        # 3. Result
        axes[2].imshow(result_img)
        resolved_loss = final_loss if final_loss is not None else (history[-1]["total_loss"] if history else 0.0)
        axes[2].set_title(
            f"3. Stylized Output (Adam)\nSteps: {len(history)} | Loss: {resolved_loss:.2e}\n({result_img.width}x{result_img.height} px)",
            color="#00E676",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        axes[2].axis("off")

        plt.suptitle(
            f"Neural Style Transfer — Baseline Experiment (Alpha={self.content_weight}, Beta={self.style_weight:.0e}, LR={self.lr})",
            color="#FFFFFF",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def _generate_loss_plots(
        self,
        history: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Create high-quality dual plots: Raw unweighted losses & Weighted total loss."""
        if not history:
            return

        steps = [entry["step"] for entry in history]
        raw_content = [entry["content_loss"] for entry in history]
        raw_style = [entry["style_loss"] for entry in history]
        raw_tv = [entry["tv_loss"] for entry in history]

        weighted_content = [entry["weighted_content"] for entry in history]
        weighted_style = [entry["weighted_style"] for entry in history]
        total_loss = [entry["total_loss"] for entry in history]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor("#181818")

        for ax in axes:
            ax.set_facecolor("#222222")
            ax.grid(True, linestyle="--", alpha=0.3, color="#666666")
            ax.tick_params(colors="#CCCCCC")
            for spine in ax.spines.values():
                spine.set_color("#444444")

        # Left Subplot: Raw Losses
        ax1 = axes[0]
        ax1.plot(steps, raw_content, color="#4FC3F7", linewidth=2.0, label="Raw Content Loss (conv4_2 MSE)")
        ax1.set_xlabel("Optimization Step", color="#E0E0E0", fontsize=11)
        ax1.set_ylabel("Raw Content Loss", color="#4FC3F7", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="#4FC3F7")

        # Twin axis for raw style loss because scales differ by orders of magnitude
        ax1_style = ax1.twinx()
        ax1_style.plot(steps, raw_style, color="#FFB74D", linewidth=2.0, linestyle="--", label="Raw Style Loss (Gram MSE)")
        ax1_style.set_ylabel("Raw Style Loss", color="#FFB74D", fontsize=11)
        ax1_style.tick_params(axis="y", labelcolor="#FFB74D")
        ax1_style.spines["right"].set_color("#FFB74D")
        ax1_style.spines["left"].set_color("#4FC3F7")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_style.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor="#2A2A2A", edgecolor="#444444", labelcolor="#E0E0E0")
        ax1.set_title("Raw Unweighted Losses vs. Step", color="#FFFFFF", fontsize=12, pad=10)

        # Right Subplot: Weighted & Total Loss (Log Scale)
        ax2 = axes[1]
        ax2.plot(steps, total_loss, color="#00E676", linewidth=2.5, label="Total Weighted Loss")
        ax2.plot(steps, weighted_style, color="#FF9100", linewidth=1.8, linestyle=":", label=f"Weighted Style (Beta={self.style_weight:.0e})")
        ax2.plot(steps, weighted_content, color="#29B6F6", linewidth=1.8, linestyle="--", label=f"Weighted Content (Alpha={self.content_weight})")

        ax2.set_yscale("log")
        ax2.set_xlabel("Optimization Step", color="#E0E0E0", fontsize=11)
        ax2.set_ylabel("Weighted Loss (Log Scale)", color="#E0E0E0", fontsize=11)
        ax2.legend(loc="upper right", facecolor="#2A2A2A", edgecolor="#444444", labelcolor="#E0E0E0")
        ax2.set_title("Weighted Loss Trajectory (Logarithmic Scale)", color="#FFFFFF", fontsize=12, pad=10)

        plt.suptitle(
            f"NST Optimization Convergence Profile — Adam Optimizer (lr={self.lr})",
            color="#FFFFFF",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def run_single_transfer(
    content_path: Union[str, Path] = "data/content/landscape_mountain.jpg",
    style_path: Union[str, Path] = "data/style/style_van_gogh.jpg",
    output_dir: Union[str, Path] = "outputs/experiments/exp01_mountain_van_gogh_adam",
    max_size: int = 256,
    num_steps: int = 300,
    lr: float = 0.02,
    content_weight: float = 1.0,
    style_weight: float = 1e5,
    tv_weight: float = 0.0,
    save_progress_every: int = 50,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_dir: Union[str, Path] = "checkpoints",
) -> Dict[str, Any]:
    """Helper entry point to instantiate and execute a single NST experiment run."""
    optimizer = StyleTransferOptimizer(
        content_path=content_path,
        style_path=style_path,
        output_dir=output_dir,
        max_size=max_size,
        num_steps=num_steps,
        lr=lr,
        content_weight=content_weight,
        style_weight=style_weight,
        tv_weight=tv_weight,
        save_progress_every=save_progress_every,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    return optimizer.run()


def main():
    """CLI Entry Point for running Neural Style Transfer."""
    parser = argparse.ArgumentParser(description="Neural Style Transfer Pixel Optimization")
    parser.add_argument("--content", type=str, default="data/content/landscape_mountain.jpg", help="Content image path")
    parser.add_argument("--style", type=str, default="data/style/style_van_gogh.jpg", help="Style reference image path")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments/exp01_mountain_van_gogh_adam", help="Output directory")
    parser.add_argument("--max-size", type=int, default=256, help="Maximum image dimension in px")
    parser.add_argument("--steps", type=int, default=300, help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=0.02, help="Adam learning rate")
    parser.add_argument("--content-weight", type=float, default=1.0, help="Content loss weight (alpha)")
    parser.add_argument("--style-weight", type=float, default=1e5, help="Style loss weight (beta)")
    parser.add_argument("--tv-weight", type=float, default=0.0, help="TV loss weight (gamma)")
    parser.add_argument("--save-every", type=int, default=50, help="Progress save cadence (steps)")
    parser.add_argument("--device", type=str, default=None, help="Explicit compute device ('cuda', 'cpu')")

    args = parser.parse_args()

    run_single_transfer(
        content_path=args.content,
        style_path=args.style,
        output_dir=args.output_dir,
        max_size=args.max_size,
        num_steps=args.steps,
        lr=args.lr,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        save_progress_every=args.save_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()

