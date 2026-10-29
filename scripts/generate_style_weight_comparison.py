"""Generate Style Weight Comparison Grid, Loss Curves, and Consolidated Metrics.

Compares style weights 1e4, 1e5, and 1e6 on the Mount Rainier x Van Gogh pair.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.utils import load_image

OUTPUT_COMPARISON_DIR = Path("outputs/experiments/style_weight_comparison")
OUTPUT_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

TRIALS = [
    {
        "id": "EXP-002",
        "style_weight_label": "1e4 (10,000)",
        "style_weight": 1e4,
        "dir": Path("outputs/experiments/exp02_mountain_van_gogh_style1e4"),
        "color": "#4FC3F7",
    },
    {
        "id": "EXP-001",
        "style_weight_label": "1e5 (100,000) [Baseline]",
        "style_weight": 1e5,
        "dir": Path("outputs/experiments/exp01_mountain_van_gogh_adam"),
        "color": "#00E676",
    },
    {
        "id": "EXP-003",
        "style_weight_label": "1e6 (1,000,000)",
        "style_weight": 1e6,
        "dir": Path("outputs/experiments/exp03_mountain_van_gogh_style1e6"),
        "color": "#FF9100",
    },
]


def load_trial_data():
    loaded = []
    for t in TRIALS:
        json_path = t["dir"] / "run_config_and_metrics.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        img_path = t["dir"] / "final_stylized.png"
        loaded.append({
            **t,
            "data": data,
            "final_image_path": img_path,
        })
    return loaded


def generate_five_panel_grid(trials_data):
    content_path = Path("data/content/landscape_mountain.jpg")
    style_path = Path("data/style/style_van_gogh.jpg")

    content_img = load_image(content_path, max_size=256)
    style_img = load_image(style_path, max_size=256)

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    fig.patch.set_facecolor("#121212")

    # 1. Content
    axes[0].imshow(content_img)
    axes[0].set_title(
        f"1. Content Landscape\nMount Rainier\n({content_img.width}x{content_img.height} px)",
        color="#E0E0E0",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[0].axis("off")

    # 2. Style
    axes[1].imshow(style_img)
    axes[1].set_title(
        f"2. Style Reference\nVan Gogh (Wheat Field)\n({style_img.width}x{style_img.height} px)",
        color="#E0E0E0",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[1].axis("off")

    # 3-5. Results for 1e4, 1e5, 1e6
    for idx, t in enumerate(trials_data):
        ax = axes[idx + 2]
        res_img = Image.open(t["final_image_path"]).convert("RGB")
        ax.imshow(res_img)

        final_m = t["data"]["final_metrics"]
        title_text = (
            f"{idx + 3}. Result (\u03b2 = {t['style_weight']:.0e})\n"
            f"Loss: {final_m['total_loss']:.3f} | Style: {final_m['style_loss']:.2e}\n"
            f"Content: {final_m['content_loss']:.3f}"
        )
        ax.set_title(
            title_text,
            color=t["color"],
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.axis("off")

    plt.suptitle(
        "Neural Style Transfer — Style Weight Experiment Comparison (\u03b1=1.0, \u03b2 \u2208 {10\u2074, 10\u2075, 10\u2076}, Adam lr=0.02, 300 Steps)",
        color="#FFFFFF",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    out_path = OUTPUT_COMPARISON_DIR / "grid_content_style_three_results.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved 5-panel comparison grid to: {out_path}")


def generate_comparative_loss_plots(trials_data):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.patch.set_facecolor("#181818")

    for ax in axes:
        ax.set_facecolor("#222222")
        ax.grid(True, linestyle="--", alpha=0.3, color="#666666")
        ax.tick_params(colors="#CCCCCC")
        for spine in ax.spines.values():
            spine.set_color("#444444")

    # 1. Total Weighted Loss (Log scale)
    ax1 = axes[0]
    for t in trials_data:
        steps = [e["step"] for e in t["data"]["history"]]
        total_losses = [e["total_loss"] for e in t["data"]["history"]]
        ax1.plot(steps, total_losses, color=t["color"], linewidth=2.2, label=f"\u03b2={t['style_weight']:.0e} ({t['id']})")
    ax1.set_yscale("log")
    ax1.set_title("Total Weighted Loss (\u03b1\u00b7L_c + \u03b2\u00b7L_s) [Log Scale]", color="#FFFFFF", fontsize=12, pad=10)
    ax1.set_xlabel("Step", color="#E0E0E0", fontsize=11)
    ax1.set_ylabel("Weighted Total Loss", color="#E0E0E0", fontsize=11)
    ax1.legend(loc="upper right", facecolor="#2A2A2A", edgecolor="#444444", labelcolor="#E0E0E0")

    # 2. Raw Style Loss (Gram MSE, Log scale)
    ax2 = axes[1]
    for t in trials_data:
        steps = [e["step"] for e in t["data"]["history"]]
        style_losses = [e["style_loss"] for e in t["data"]["history"]]
        ax2.plot(steps, style_losses, color=t["color"], linewidth=2.2, label=f"\u03b2={t['style_weight']:.0e} ({t['id']})")
    ax2.set_yscale("log")
    ax2.set_title("Raw Style Loss (Gram Matrix MSE) [Log Scale]", color="#FFFFFF", fontsize=12, pad=10)
    ax2.set_xlabel("Step", color="#E0E0E0", fontsize=11)
    ax2.set_ylabel("Raw Style Loss", color="#E0E0E0", fontsize=11)
    ax2.legend(loc="upper right", facecolor="#2A2A2A", edgecolor="#444444", labelcolor="#E0E0E0")

    # 3. Raw Content Loss (conv4_2 MSE, Linear scale)
    ax3 = axes[2]
    for t in trials_data:
        steps = [e["step"] for e in t["data"]["history"]]
        content_losses = [e["content_loss"] for e in t["data"]["history"]]
        ax3.plot(steps, content_losses, color=t["color"], linewidth=2.2, label=f"\u03b2={t['style_weight']:.0e} ({t['id']})")
    ax3.set_title("Raw Content Loss (conv4_2 MSE)", color="#FFFFFF", fontsize=12, pad=10)
    ax3.set_xlabel("Step", color="#E0E0E0", fontsize=11)
    ax3.set_ylabel("Raw Content Loss", color="#E0E0E0", fontsize=11)
    ax3.legend(loc="upper left", facecolor="#2A2A2A", edgecolor="#444444", labelcolor="#E0E0E0")

    plt.suptitle(
        "Convergence Dynamics Across Style Weights (\u03b2 \u2208 {10\u2074, 10\u2075, 10\u2076})",
        color="#FFFFFF",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    out_path = OUTPUT_COMPARISON_DIR / "comparative_loss_plots.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved comparative loss plots to: {out_path}")


def generate_consolidated_summary(trials_data):
    content_hash = trials_data[0]["data"]["input_hashes"]["content_file_sha256"]
    style_hash = trials_data[0]["data"]["input_hashes"]["style_file_sha256"]
    content_t_hash = trials_data[0]["data"]["input_hashes"]["content_tensor_sha256"]
    style_t_hash = trials_data[0]["data"]["input_hashes"]["style_tensor_sha256"]

    # Verify input hashes are identical across all trials
    for t in trials_data:
        ih = t["data"]["input_hashes"]
        assert ih["content_file_sha256"] == content_hash, "Content file hash mismatch!"
        assert ih["style_file_sha256"] == style_hash, "Style file hash mismatch!"
        assert ih["content_tensor_sha256"] == content_t_hash, "Content tensor hash mismatch!"
        assert ih["style_tensor_sha256"] == style_t_hash, "Style tensor hash mismatch!"

    table = []
    for t in trials_data:
        d = t["data"]
        init_m = d["initial_metrics"]
        final_m = d["final_metrics"]
        exec_info = d["execution"]
        table.append({
            "experiment_id": t["id"],
            "style_weight": t["style_weight"],
            "style_to_content_ratio": t["style_weight"] / d["configuration"]["content_weight"],
            "total_runtime_seconds": exec_info["total_runtime_seconds"],
            "avg_step_time_ms": exec_info["avg_step_time_ms"],
            "initial_total_loss": init_m["total_loss"],
            "final_total_loss_post_update": final_m["total_loss"],
            "total_loss_reduction_pct": d["metrics_summary"]["total_loss_reduction_pct"],
            "initial_content_loss": init_m["content_loss"],
            "final_content_loss_post_update": final_m["content_loss"],
            "initial_style_loss": init_m["style_loss"],
            "final_style_loss_post_update": final_m["style_loss"],
            "style_loss_reduction_pct": d["metrics_summary"]["style_loss_reduction_pct"],
            "vgg_weights_frozen_verified": exec_info["vgg_weights_frozen_verified"],
            "output_directory": str(t["dir"]),
        })

    summary = {
        "title": "Style Weight Comparison Experiment Summary",
        "date": "2026-10-29",
        "fixed_hyperparameters": {
            "content_image": "data/content/landscape_mountain.jpg",
            "style_image": "data/style/style_van_gogh.jpg",
            "content_weight": 1.0,
            "tv_weight": 0.0,
            "optimizer": "Adam",
            "learning_rate": 0.02,
            "steps": 300,
            "max_size": 256,
            "content_layer": "conv4_2",
            "style_layers": ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
        },
        "verified_input_hashes": {
            "content_file_sha256": content_hash,
            "style_file_sha256": style_hash,
            "content_tensor_sha256": content_t_hash,
            "style_tensor_sha256": style_t_hash,
            "all_trials_used_identical_inputs": True,
        },
        "trials_comparison": table,
        "qualitative_evaluation": {
            "style_weight_1e4": {
                "content_preservation": "Sangat tinggi (hampir identik dengan foto asli)",
                "style_texture_strength": "Sangat lemah, transfer gaya terbatas pada pergeseran rona warna tipis di langit",
                "artifacts": "Sangat minim artefak, tetapi gaya Van Gogh belum terserap secara artistik",
            },
            "style_weight_1e5": {
                "content_preservation": "Tinggi (garis punggung bukit, gletser, dan batas vegetasi sangat jelas)",
                "style_texture_strength": "Moderat (palet warna biru toska, kuning gandum, dan hijau cypress terserap baik; tekstur impasto mulai terbentuk)",
                "artifacts": "Grain halus pada langit terbuka (karena TV=0), bentuk semantik tetap proporsional",
            },
            "style_weight_1e6": {
                "content_preservation": "Mulai terdegradasi (puncak gunung dan awan melebur ke dalam pola sapuan abstrak, batas objek mulai kabur)",
                "style_texture_strength": "Sangat kuat (sapuan kuas tebal dan vorteks berputar khas Van Gogh mendominasi seluruh kanvas)",
                "artifacts": "Distorsi struktural lokal dan fragmentasi warna; kontur gunung kehilangan beberapa detail fisik",
            },
        },
        "interim_recommendation": {
            "recommended_weight": 1e5,
            "reasoning": "Rasio beta/alpha = 1e5 memberikan titik temu optimal (sweet spot) antara keterjagaan bentuk semantik lanskap gunung dan penyerapan karakteristik visual maestro Van Gogh tanpa degradasi topologi ekstrem seperti pada 1e6 atau under-styling seperti pada 1e4.",
        },
    }

    out_json = OUTPUT_COMPARISON_DIR / "comparison_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved consolidated summary JSON to: {out_json}")


if __name__ == "__main__":
    trials = load_trial_data()
    generate_five_panel_grid(trials)
    generate_comparative_loss_plots(trials)
    generate_consolidated_summary(trials)
    print("All style weight comparison artifacts generated successfully!")

