"""Test the 3 Recommended Demonstration Pairs and verify end-to-end artifact generation.

Recommended Pairs:
1. Laut / Pantai x Hokusai (landscape_beach_waves.jpg x style_hokusai.jpg) -> DEMO-REC-01
2. Sawah / Padang Rumput x Van Gogh (landscape_rice_terrace.jpg x style_van_gogh.jpg) -> DEMO-REC-02
3. Gunung / Danau x Kandinsky (landscape_mountain.jpg x style_kandinsky.jpg) -> DEMO-REC-03 (EXP-005 baseline)

Verifies:
- Execution using cached VGG extractor.
- Generation of final PNG and metrics JSON.
- Evaluation of float tensor vs quantized PNG uint8 metrics.
- Readiness for interactive download in the Streamlit demo.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from src.model import build_vgg19_extractor
from src.transfer import StyleTransferOptimizer
from src.utils import get_device

RECOMMENDED_TRIALS = [
    {
        "trial_id": "DEMO-REC-01",
        "category": "Laut / Pantai Berombak × Hokusai (The Great Wave)",
        "content_path": "data/content/landscape_beach_waves.jpg",
        "style_path": "data/style/style_hokusai.jpg",
        "output_dir": "outputs/demo_showcases/rec01_beach_waves_hokusai",
        "steps": 300,
        "style_weight": 1e5,
    },
    {
        "trial_id": "DEMO-REC-02",
        "category": "Sawah / Padang Berundak × Van Gogh (Wheat Field)",
        "content_path": "data/content/landscape_rice_terrace.jpg",
        "style_path": "data/style/style_van_gogh.jpg",
        "output_dir": "outputs/demo_showcases/rec02_rice_terrace_van_gogh",
        "steps": 300,
        "style_weight": 1e5,
    },
    {
        "trial_id": "DEMO-REC-03",
        "category": "Gunung × Kandinsky (Composition 8) — Eksplorasi Abstrak",
        "content_path": "data/content/landscape_mountain.jpg",
        "style_path": "data/style/style_kandinsky.jpg",
        "output_dir": "outputs/demo_showcases/rec03_mountain_kandinsky",
        "steps": 300,
        "style_weight": 1e5,
    },
]


def run_recommended_tests():
    device = get_device()
    print("=" * 80)
    print("   TESTING 3 RECOMMENDED DEMO SHOWCASE PAIRS")
    print("=" * 80)
    print(f"Device: {device}")

    # Build cached extractor once to simulate Streamlit @st.cache_resource
    print("\n[Setup] Building cached VGG-19 extractor...")
    t0 = time.perf_counter()
    cached_extractor = build_vgg19_extractor(device=device)
    print(f"[Setup] VGG-19 Extractor built in {time.perf_counter() - t0:.2f} s")

    results = []

    for trial in RECOMMENDED_TRIALS:
        t_id = trial["trial_id"]
        cat = trial["category"]
        c_path = Path(trial["content_path"])
        s_path = Path(trial["style_path"])
        out_dir = Path(trial["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n---> Running {t_id}: {cat}")
        print(f"     Content: {c_path}")
        print(f"     Style:   {s_path}")

        optimizer = StyleTransferOptimizer(
            content_path=c_path,
            style_path=s_path,
            output_dir=out_dir,
            max_size=256,
            num_steps=trial["steps"],
            lr=0.02,
            content_weight=1.0,
            style_weight=trial["style_weight"],
            tv_weight=0.0,
            save_progress_every=50,
            device=device,
            extractor=cached_extractor,
        )

        summary = optimizer.run()

        # Check final image and JSON
        final_img = Path(summary["artifacts"]["final_image"])
        final_json = out_dir / "run_config_and_metrics.json"

        assert final_img.is_file(), f"Final image not found: {final_img}"
        assert final_json.is_file(), f"Final JSON not found: {final_json}"

        with Image.open(final_img) as img:
            w, h = img.size

        # Simulate download byte reading
        with open(final_img, "rb") as f:
            png_bytes = f.read()

        with open(final_json, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        m_sum = json_data.get("metrics_summary", {})
        exec_info = json_data.get("execution", {})

        record = {
            "trial_id": t_id,
            "category": cat,
            "runtime_seconds": exec_info.get("total_runtime_seconds"),
            "resolution": f"{w}x{h}",
            "png_file_size_kb": len(png_bytes) / 1024.0,
            "initial_total_loss": m_sum.get("initial_total_loss"),
            "final_float_total_loss": m_sum.get("final_total_loss_float"),
            "final_quantized_png_total_loss": m_sum.get("final_total_loss_quantized_png"),
            "quantization_delta_total_loss": m_sum.get("quantization_delta_total_loss"),
            "final_content_loss": m_sum.get("final_content_loss_float"),
            "final_style_loss": m_sum.get("final_style_loss_float"),
            "style_loss_reduction_pct": m_sum.get("style_loss_reduction_pct"),
            "final_image_path": str(final_img),
        }
        results.append(record)

        print(f"[{t_id} Completed] Runtime: {record['runtime_seconds']} s | Style Reduksi: {record['style_loss_reduction_pct']}% | PNG Size: {record['png_file_size_kb']:.1f} KB")

    summary_file = Path("outputs/demo_showcases/recommended_pairs_test_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"All 3 Recommended Pairs Tested and Verified! Summary saved to {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_recommended_tests()

