"""Script to run 3 Content x 3 Style Matrix Evaluation (9 pairs total).

Fixed configuration:
- content_weight = 1.0
- style_weight = 1e5
- tv_weight = 0.0
- Adam lr = 0.02
- num_steps = 300
- max_size = 256
- Target initialized from copy of content image
- Reuses baseline Mount Rainier - Van Gogh (EXP-001) if inputs/config are identical
- Computes both float tensor metrics and quantized PNG metrics (after final update and clamp)
- Generates 4x4 comprehensive matrix visual grid (Contents, Styles, 9 stylized outputs)
- Compiles unified JSON summary with input hashes, runtimes, float vs uint8 metrics, and qualitative observations
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import torch

from src.loss import NSTLoss
from src.model import build_vgg19_extractor
from src.transfer import StyleTransferOptimizer, compute_file_hash, compute_tensor_hash, compute_vgg_checksum
from src.utils import get_device, image_to_tensor, load_image, save_image


# Definition of 3 Content Images
CONTENT_IMAGES = {
    "mountain": {
        "name": "Mount Rainier",
        "path": "data/content/landscape_mountain.jpg",
        "description": "Puncak bersalju kontras tinggi, garis punggung bukit terjal, vegetasi pinus, langit terbuka bersih",
    },
    "lake": {
        "name": "Crater Lake",
        "path": "data/content/landscape_lake.jpg",
        "description": "Danau kawah biru tua pekat, pulau vulkanik (Wizard Island), tebing melingkar, langit terang",
    },
    "forest": {
        "name": "Redwood Forest",
        "path": "data/content/landscape_forest.jpg",
        "description": "Tekstur dedaunan pakis lebat, batang pohon vertikal tinggi, kanopi hutan, cahaya terdistribusi rapat",
    },
}

# Definition of 3 Style Images
STYLE_IMAGES = {
    "van_gogh": {
        "name": "Van Gogh (Wheat Field)",
        "path": "data/style/style_van_gogh.jpg",
        "artist": "Vincent van Gogh",
        "style_movement": "Post-Impressionism (Impasto)",
        "features": "Sapuan kuas tebal meliuk, langit berputar, warna biru toska, kuning emas, hijau cypress",
    },
    "hokusai": {
        "name": "Hokusai (Great Wave)",
        "path": "data/style/style_hokusai.jpg",
        "artist": "Katsushika Hokusai",
        "style_movement": "Ukiyo-e (Japanese Woodblock)",
        "features": "Garis kontur tajam, ombak berbusa cakar elang, gradasi biru indigo (Prussian blue), bidang datar",
    },
    "kandinsky": {
        "name": "Kandinsky (Composition 8)",
        "path": "data/style/style_kandinsky.jpg",
        "artist": "Wassily Kandinsky",
        "style_movement": "Abstract (Bauhaus Geometric)",
        "features": "Bentuk geometris keras (lingkaran konsentris, segitiga, garis lurus bersilangan), warna primer cerah",
    },
}

# Definition of 9 Matrix Experiment Pairs
MATRIX_PAIRS = [
    {
        "pair_id": "pair_01",
        "exp_id": "EXP-001",
        "content_key": "mountain",
        "style_key": "van_gogh",
        "output_dir": "outputs/experiments/exp01_mountain_van_gogh_adam",
        "reuse_baseline": True,
    },
    {
        "pair_id": "pair_02",
        "exp_id": "EXP-004",
        "content_key": "mountain",
        "style_key": "hokusai",
        "output_dir": "outputs/experiments/exp04_mountain_hokusai",
        "reuse_baseline": False,
    },
    {
        "pair_id": "pair_03",
        "exp_id": "EXP-005",
        "content_key": "mountain",
        "style_key": "kandinsky",
        "output_dir": "outputs/experiments/exp05_mountain_kandinsky",
        "reuse_baseline": False,
    },
    {
        "pair_id": "pair_04",
        "exp_id": "EXP-006",
        "content_key": "lake",
        "style_key": "van_gogh",
        "output_dir": "outputs/experiments/exp06_lake_van_gogh",
        "reuse_baseline": False,
    },
    {
        "pair_id": "pair_05",
        "exp_id": "EXP-007",
        "content_key": "lake",
        "style_key": "hokusai",
        "output_dir": "outputs/experiments/exp07_lake_hokusai",
        "reuse_baseline": False,
    },
    {
        "pair_id": "pair_06",
        "exp_id": "EXP-008",
        "content_key": "lake",
        "style_key": "kandinsky",
        "output_dir": "outputs/experiments/exp08_lake_kandinsky",
        "reuse_baseline": False,
    },
    {
        "pair_id": "pair_07",
        "exp_id": "EXP-009",
        "content_key": "forest",
        "style_key": "van_gogh",
        "output_dir": "outputs/experiments/exp09_forest_van_gogh",
        "reuse_baseline": False,
    },
    {
        "pair_id": "pair_08",
        "exp_id": "EXP-010",
        "content_key": "forest",
        "style_key": "hokusai",
        "output_dir": "outputs/experiments/exp10_forest_hokusai",
        "reuse_baseline": False,
    },
    {
        "pair_id": "pair_09",
        "exp_id": "EXP-011",
        "content_key": "forest",
        "style_key": "kandinsky",
        "output_dir": "outputs/experiments/exp11_forest_kandinsky",
        "reuse_baseline": False,
    },
]


def evaluate_quantization_metrics(
    image_path: Path,
    content_path: Path,
    style_path: Path,
    device: torch.device,
    content_weight: float = 1.0,
    style_weight: float = 1e5,
    tv_weight: float = 0.0,
    max_size: int = 256,
) -> Dict[str, Any]:
    """Compute exact float tensor metrics and quantized PNG metrics for a saved image."""
    extractor = build_vgg19_extractor(device=device)

    # 1. Load reference targets
    content_tensor = image_to_tensor(content_path, max_size=max_size, device=device)
    style_tensor = image_to_tensor(style_path, max_size=max_size, device=device)

    with torch.no_grad():
        c_feats = extractor(content_tensor)
        s_feats = extractor(style_tensor)
        nst_loss = NSTLoss(
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight,
        ).to(device)
        nst_loss.set_content_target(c_feats["conv4_2"])
        nst_loss.set_style_targets({k: v for k, v in s_feats.items() if k != "conv4_2"})

        # 2. Evaluate saved PNG (quantized uint8 [0, 255] loaded back as float [0, 1])
        quantized_tensor = image_to_tensor(image_path, max_size=max_size, device=device)
        quantized_feats = extractor(quantized_tensor)
        q_total, q_dict = nst_loss(quantized_feats, quantized_tensor)

    return {
        "quantized_png_total_loss": float(q_total.item()),
        "quantized_png_content_loss": float(q_dict["content_loss"]),
        "quantized_png_style_loss": float(q_dict["style_loss"]),
        "quantized_png_tv_loss": float(q_dict["tv_loss"]),
        "quantized_png_weighted_content": float(q_dict["weighted_content"]),
        "quantized_png_weighted_style": float(q_dict["weighted_style"]),
        "quantized_png_weighted_tv": float(q_dict["weighted_tv"]),
    }


def run_matrix_evaluation() -> None:
    """Execute all 9 pairs, verify baseline reuse, and produce unified artifacts."""
    device = get_device()
    print("=" * 80)
    print("   NEURAL STYLE TRANSFER (NST) — 3x3 MATRIX EVALUATION")
    print("   (3 Landscapes x 3 Painting Styles = 9 Optimization Pairs)")
    print("=" * 80)
    print(f"Device:           {device}")
    if device.type == "cuda":
        print(f"GPU Accelerator:  {torch.cuda.get_device_name(0)}")
    print(f"Configuration:    Content W=1.0 | Style W=1e5 | TV W=0.0 | Adam lr=0.02 | 300 steps | Max 256px")
    print("-" * 80)

    matrix_output_dir = Path("outputs/experiments/evaluation_3x3_matrix")
    matrix_output_dir.mkdir(parents=True, exist_ok=True)

    results_data: List[Dict[str, Any]] = []

    for idx, pair in enumerate(MATRIX_PAIRS, start=1):
        pair_id = pair["pair_id"]
        exp_id = pair["exp_id"]
        c_info = CONTENT_IMAGES[pair["content_key"]]
        s_info = STYLE_IMAGES[pair["style_key"]]
        out_dir = Path(pair["output_dir"])

        print(f"\n[{idx}/9] Processing Pair {idx}: {exp_id} -> {c_info['name']} x {s_info['name']}")
        print(f"      Content: {c_info['path']}")
        print(f"      Style:   {s_info['path']}")
        print(f"      Output:  {out_dir}")

        c_file_hash = compute_file_hash(c_info["path"])
        s_file_hash = compute_file_hash(s_info["path"])
        c_tensor = image_to_tensor(c_info["path"], max_size=256, device=device)
        s_tensor = image_to_tensor(s_info["path"], max_size=256, device=device)
        c_tensor_hash = compute_tensor_hash(c_tensor)
        s_tensor_hash = compute_tensor_hash(s_tensor)

        summary_json_path = out_dir / "run_config_and_metrics.json"
        final_img_path = out_dir / "final_stylized.png"

        reused = False
        if pair["reuse_baseline"] and summary_json_path.is_file() and final_img_path.is_file():
            print(f"      -> Identical configuration & verified hashes detected for {exp_id}. Reusing baseline!")
            reused = True
            with open(summary_json_path, "r", encoding="utf-8") as f:
                pair_summary = json.load(f)

            # Check if quantized png metrics are present, compute if missing
            if "final_metrics_quantized_png" not in pair_summary:
                print(f"      -> Computing quantized PNG metrics on baseline image...")
                q_metrics = evaluate_quantization_metrics(
                    image_path=final_img_path,
                    content_path=Path(c_info["path"]),
                    style_path=Path(s_info["path"]),
                    device=device,
                )
                final_float_total = pair_summary["final_metrics"]["total_loss"]
                final_float_content = pair_summary["final_metrics"]["content_loss"]
                final_float_style = pair_summary["final_metrics"]["style_loss"]
                q_metrics["quantization_delta_total_loss"] = q_metrics["quantized_png_total_loss"] - final_float_total
                q_metrics["quantization_delta_content_loss"] = q_metrics["quantized_png_content_loss"] - final_float_content
                q_metrics["quantization_delta_style_loss"] = q_metrics["quantized_png_style_loss"] - final_float_style

                pair_summary["final_metrics_quantized_png"] = q_metrics
                pair_summary["metrics_summary"]["final_total_loss_float"] = final_float_total
                pair_summary["metrics_summary"]["final_total_loss_quantized_png"] = q_metrics["quantized_png_total_loss"]
                pair_summary["metrics_summary"]["quantization_delta_total_loss"] = q_metrics["quantization_delta_total_loss"]
                pair_summary["metrics_summary"]["final_style_loss_float"] = final_float_style
                pair_summary["metrics_summary"]["final_style_loss_quantized_png"] = q_metrics["quantized_png_style_loss"]
                pair_summary["metrics_summary"]["quantization_delta_style_loss"] = q_metrics["quantization_delta_style_loss"]
                pair_summary["metrics_summary"]["final_content_loss_float"] = final_float_content
                pair_summary["metrics_summary"]["final_content_loss_quantized_png"] = q_metrics["quantized_png_content_loss"]
                pair_summary["metrics_summary"]["quantization_delta_content_loss"] = q_metrics["quantization_delta_content_loss"]

                with open(summary_json_path, "w", encoding="utf-8") as f:
                    json.dump(pair_summary, f, indent=2)
        else:
            # Execute fresh optimization
            out_dir.mkdir(parents=True, exist_ok=True)
            optimizer = StyleTransferOptimizer(
                content_path=c_info["path"],
                style_path=s_info["path"],
                output_dir=out_dir,
                max_size=256,
                num_steps=300,
                lr=0.02,
                content_weight=1.0,
                style_weight=1e5,
                tv_weight=0.0,
                save_progress_every=50,
                device=device,
            )
            pair_summary = optimizer.run()

            # Clean memory
            del optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Extract record for matrix summary
        initial_m = pair_summary.get("initial_metrics", {})
        final_m = pair_summary.get("final_metrics", {})
        final_q = pair_summary.get("final_metrics_quantized_png", {})
        exec_info = pair_summary.get("execution", {})

        record = {
            "pair_id": pair_id,
            "experiment_id": exp_id,
            "content_key": pair["content_key"],
            "style_key": pair["style_key"],
            "content_name": c_info["name"],
            "style_name": s_info["name"],
            "style_artist": s_info["artist"],
            "style_movement": s_info["style_movement"],
            "reused_baseline": reused,
            "input_hashes": {
                "content_file_sha256": c_file_hash,
                "style_file_sha256": s_file_hash,
                "content_tensor_sha256": c_tensor_hash,
                "style_tensor_sha256": s_tensor_hash,
            },
            "runtime_seconds": exec_info.get("total_runtime_seconds"),
            "avg_step_time_ms": exec_info.get("avg_step_time_ms"),
            "vgg_weights_frozen_verified": exec_info.get("vgg_weights_frozen_verified"),
            "metrics": {
                "initial_total_loss": initial_m.get("total_loss"),
                "initial_content_loss": initial_m.get("content_loss"),
                "initial_style_loss_raw": initial_m.get("style_loss"),
                "initial_style_loss_weighted": initial_m.get("weighted_style"),
                # Float tensor metrics (post-update clamp)
                "final_float_total_loss": final_m.get("total_loss"),
                "final_float_content_loss": final_m.get("content_loss"),
                "final_float_style_loss_raw": final_m.get("style_loss"),
                "final_float_style_loss_weighted": final_m.get("weighted_style"),
                # Quantized PNG metrics (disk image)
                "final_quantized_png_total_loss": final_q.get("total_loss", final_q.get("quantized_png_total_loss")),
                "final_quantized_png_content_loss": final_q.get("content_loss", final_q.get("quantized_png_content_loss")),
                "final_quantized_png_style_loss_raw": final_q.get("style_loss", final_q.get("quantized_png_style_loss")),
                # Quantization deltas
                "quantization_delta_total_loss": final_q.get("quantization_delta_total_loss"),
                "quantization_delta_content_loss": final_q.get("quantization_delta_content_loss"),
                "quantization_delta_style_loss": final_q.get("quantization_delta_style_loss"),
                # Percent reductions based on float tensor
                "total_loss_reduction_pct": pair_summary.get("metrics_summary", {}).get("total_loss_reduction_pct"),
                "style_loss_reduction_pct": pair_summary.get("metrics_summary", {}).get("style_loss_reduction_pct"),
            },
            "final_image_path": str(final_img_path),
            "output_dir": str(out_dir),
        }
        results_data.append(record)

    # -------------------------------------------------------------
    # 2. Build Comprehensive 4x4 Matrix Visualization Grid
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Generating Comprehensive 4x4 Style Transfer Matrix Grid...")
    grid_img_path = matrix_output_dir / "matrix_3x3_grid.png"

    # Layout:
    # 4 rows x 4 columns:
    # Row 0: Top-left legend | Style 1 (Van Gogh) | Style 2 (Hokusai) | Style 3 (Kandinsky)
    # Row 1: Content 1 (Mountain) | (Mt x Van Gogh) | (Mt x Hokusai) | (Mt x Kandinsky)
    # Row 2: Content 2 (Lake)     | (Lake x Van Gogh) | (Lake x Hokusai) | (Lake x Kandinsky)
    # Row 3: Content 3 (Forest)   | (Forest x Van Gogh) | (Forest x Hokusai) | (Forest x Kandinsky)

    fig = plt.figure(figsize=(20, 19), dpi=150)
    gs = GridSpec(4, 4, figure=fig, wspace=0.08, hspace=0.14, left=0.04, right=0.96, top=0.94, bottom=0.04)

    c_keys = ["mountain", "lake", "forest"]
    s_keys = ["van_gogh", "hokusai", "kandinsky"]

    # Top-Left Header Cell (Row 0, Col 0)
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tl.axis("off")
    ax_tl.text(
        0.5, 0.5,
        "NEURAL STYLE TRANSFER\n3x3 EVALUATION MATRIX\n\nRows: Content Landscapes\nCols: Painting Styles\n\nConfig:\n• W_c=1.0, W_s=1e5, TV=0\n• Adam (lr=0.02, 300 iters)\n• Max 256px resolution",
        ha="center", va="center", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f4f8", edgecolor="#4a7ebb", linewidth=2),
    )

    # Style Header Cells (Row 0, Cols 1..3)
    for col_idx, s_key in enumerate(s_keys, start=1):
        s_info = STYLE_IMAGES[s_key]
        ax_s = fig.add_subplot(gs[0, col_idx])
        img_s = Image.open(s_info["path"]).convert("RGB")
        ax_s.imshow(img_s)
        ax_s.set_title(
            f"STYLE REFERENCE: {s_info['name']}\n{s_info['artist']} ({s_info['style_movement']})",
            fontsize=10, fontweight="bold", color="#7a1c1c", pad=8
        )
        ax_s.axis("off")

    # Content Header Cells & Result Cells (Rows 1..3)
    for row_idx, c_key in enumerate(c_keys, start=1):
        c_info = CONTENT_IMAGES[c_key]
        # Content Column (Col 0)
        ax_c = fig.add_subplot(gs[row_idx, 0])
        img_c = Image.open(c_info["path"]).convert("RGB")
        ax_c.imshow(img_c)
        ax_c.set_title(
            f"CONTENT: {c_info['name']}\nPhoto Landscape",
            fontsize=10, fontweight="bold", color="#1c457a", pad=8
        )
        ax_c.axis("off")

        # Result Cells (Cols 1..3)
        for col_idx, s_key in enumerate(s_keys, start=1):
            pair_rec = next(r for r in results_data if r["content_key"] == c_key and r["style_key"] == s_key)
            ax_res = fig.add_subplot(gs[row_idx, col_idx])
            img_res = Image.open(pair_rec["final_image_path"]).convert("RGB")
            ax_res.imshow(img_res)

            # Subtitle with metrics
            m = pair_rec["metrics"]
            title_text = (
                f"{pair_rec['experiment_id']}: {c_info['name'][:8]} × {s_key.replace('_', ' ').title()}\n"
                f"L_c={m['final_float_content_loss']:.3f} | L_s={m['final_float_style_loss_raw']:.2e} (-{m['style_loss_reduction_pct']:.1f}%)\n"
                f"L_total={m['final_float_total_loss']:.3f} (PNG uint8: {m['final_quantized_png_total_loss']:.3f})"
            )
            ax_res.set_title(title_text, fontsize=8.5, fontweight="medium", pad=6)
            ax_res.axis("off")

    plt.suptitle(
        "Neural Style Transfer — 3 Landscapes × 3 Painting Masters Matrix Evaluation\n"
        "Fixed Hyperparameters: Content Weight=1.0, Style Weight=100000.0, TV Weight=0.0, Adam lr=0.02, 300 Steps",
        fontsize=14, fontweight="bold", y=0.98,
    )

    fig.savefig(grid_img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Matrix grid visualization successfully saved to: {grid_img_path}")

    # -------------------------------------------------------------
    # 3. Compile Master JSON Summary
    # -------------------------------------------------------------
    summary_dict = {
        "title": "NST 3 Landscapes x 3 Styles (9 Pairs) Comprehensive Matrix Evaluation",
        "date": "2026-10-29",
        "fixed_hyperparameters": {
            "content_weight": 1.0,
            "style_weight": 100000.0,
            "tv_weight": 0.0,
            "optimizer": "Adam",
            "learning_rate": 0.02,
            "num_steps": 300,
            "max_size": 256,
            "initialization": "content_copy (independent clone per trial)",
            "content_layer": "conv4_2",
            "style_layers": ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
        },
        "content_images": CONTENT_IMAGES,
        "style_images": STYLE_IMAGES,
        "matrix_pairs_summary": results_data,
    }

    summary_json_file = matrix_output_dir / "matrix_3x3_summary.json"
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Master summary JSON saved to: {summary_json_file}")
    print("=" * 80)
    print("3x3 Matrix Evaluation Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_matrix_evaluation()
