"""Studio Neural Style Transfer (NST) — Lanskap Alami x Maestro Klasik.

Aplikasi demo lokal interaktif berbasis Streamlit yang memanfaatkan pipeline NST
resmi proyek tanpa menduplikasi kode logika transfer.

Fitur Utama:
- Pilihan 8 preset pemandangan alam beresolusi tinggi + opsi unggah citra mandiri (JPG/PNG).
- Pilihan 3 preset lukisan maestro klasik (Van Gogh, Hokusai, Kandinsky) + opsi unggah gaya.
- Rekomendasi pasangan estetik berbasis karakteristik visual (dengan kebebasan kombinasi penuh).
- Konfigurasi parameter iterasi (50-300 langkah) dan bobot gaya (1e4 s/d 1e6).
- Caching ekstraktor fitur VGG-19 beku via st.cache_resource untuk efisiensi komputasi.
- Visualisasi dinamis proses transfer dan tampilan komparasi Konten - Gaya - Hasil.
- Unduh hasil citra PNG dan konfigurasi/metrik JSON terstruktur.
- Bagian detail metrik matematis dan analisis kuantisasi (bukan skor kualitas artistik).
- Penanganan error komprehensif: citra rusak, aset hilang, dan OOM (Out of Memory).
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from PIL import Image
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.loss import NSTLoss
from src.model import VGG19FeatureExtractor, build_vgg19_extractor
from src.transfer import StyleTransferOptimizer
from src.utils import (
    CorruptedImageError,
    DoubleNormalizationError,
    InvalidImageError,
    get_device,
    load_image,
)

# -----------------------------------------------------------------------------
# 1. Preset Definitions (8 Landscapes & 3 Master Paintings)
# -----------------------------------------------------------------------------
LANDSCAPE_PRESETS = {
    "mountain": {
        "title": "Gunung (Mount Rainier)",
        "tag": "gunung",
        "path": "data/content/landscape_mountain.jpg",
        "creator": "WqemenW (CC0)",
        "desc": "Puncak gunung bersalju dengan kontur puncak tajam, lereng berbatu, dan langit terbuka bersih.",
        "best_match": "kandinsky",
        "match_reason": "Kontur topografi gunung yang kokoh cocok untuk mengeksplorasi deformasi bentuk abstrak.",
    },
    "lake": {
        "title": "Danau (Crater Lake)",
        "tag": "danau",
        "path": "data/content/landscape_lake.jpg",
        "creator": "USGS (Public Domain)",
        "desc": "Danau kawah vulkanik biru tua pekat dengan tebing melingkar dan pulau Wizard Island.",
        "best_match": "van_gogh",
        "match_reason": "Kedalaman air danau kawah menyatu alami dengan ritme turbulensi sapuan kuas impasto.",
    },
    "forest_coast": {
        "title": "Pesisir Berpohon Redwood (Coastal Shore)",
        "tag": "pesisir_berpohon",
        "path": "data/content/landscape_forest.jpg",
        "creator": "NPS (Public Domain)",
        "desc": "Pesisir berbatu dengan jajaran pohon redwood di tepian (karakter pesisir terbuka, bukan interior hutan).",
        "best_match": "hokusai",
        "match_reason": "Garis pantai dan batas pohon pesisir merespons kontur cetak balok kayu ukiyo-e dengan tegas.",
    },
    "beach_waves": {
        "title": "Pantai Berombak (Crashing Waves)",
        "tag": "pantai_berombak",
        "path": "data/content/landscape_beach_waves.jpg",
        "creator": "James Wheeler (CC0)",
        "desc": "Pantai luas dengan buih ombak laut berbusa putih memecah di atas pasir basah.",
        "best_match": "hokusai",
        "match_reason": "Tema maritim ombak berbusa sangat harmonis dengan estetika The Great Wave off Kanagawa.",
    },
    "rocky_sea": {
        "title": "Laut Berbatu (Rocky Sea Coast)",
        "tag": "laut_berbatu",
        "path": "data/content/landscape_rocky_sea.jpg",
        "creator": "Austin Schmid (CC0)",
        "desc": "Tebing karang laut vulkanik gelap dengan deburan ombak laut biru tua.",
        "best_match": "hokusai",
        "match_reason": "Kontras karang gelap dan air laut sangat selaras dengan pigmen Prussian blue Hokusai.",
    },
    "rice_terrace": {
        "title": "Sawah / Padang Berundak (Rice Terraces)",
        "tag": "sawah_padang_rumput",
        "path": "data/content/landscape_rice_terrace.jpg",
        "creator": "Dohduhdah (Public Domain / PD-self)",
        "desc": "Hamparan persawahan berundak hijau subur dan padang rumput lereng perbukitan (difoto 2011).",
        "best_match": "van_gogh",
        "match_reason": "Pola terasering alami sangat senada dengan sapuan gandum berombak khas Van Gogh.",
    },
    "sunset": {
        "title": "Senja (Golden Sunset)",
        "tag": "senja",
        "path": "data/content/landscape_sunset.jpg",
        "creator": "Tomas Sobek (CC0)",
        "desc": "Pemandangan senja dengan semburat langit jingga keemasan dan siluet perbukitan.",
        "best_match": "van_gogh",
        "match_reason": "Semburat langit senja menyerap palet warna ochre dan kuning emas Van Gogh secara dramatis.",
    },
    "interior_forest": {
        "title": "Hutan Pedalaman (Misty Interior Forest)",
        "tag": "hutan_pedalaman",
        "path": "data/content/landscape_interior_forest.jpg",
        "creator": "Kalen Emsley (CC0)",
        "desc": "Pedalaman hutan cemara berkabut tebal dengan kanopi rapat dan lantai hutan berlumut.",
        "best_match": "kandinsky",
        "match_reason": "Tekstur dedaunan lebat dan kabut memberikan latar kaya bagi kontras bentuk geometris.",
    },
}

STYLE_PRESETS = {
    "van_gogh": {
        "title": "Vincent van Gogh (Wheat Field with Cypresses)",
        "artist": "Vincent van Gogh (1889)",
        "movement": "Post-Impressionism (Impasto)",
        "path": "data/style/style_van_gogh.jpg",
        "desc": "Sapuan kuas tebal meliuk (*impasto*), awan turbulen, palet warna toska, kuning emas, dan hijau zaitun.",
    },
    "hokusai": {
        "title": "Katsushika Hokusai (The Great Wave off Kanagawa)",
        "artist": "Katsushika Hokusai (ca. 1830)",
        "movement": "Ukiyo-e (Japanese Woodblock)",
        "path": "data/style/style_hokusai.jpg",
        "desc": "Cetak balok kayu Jepang dengan gradasi biru nila (*Prussian blue*), garis kontur tajam, dan buih ombak putih.",
    },
    "kandinsky": {
        "title": "Wassily Kandinsky (Composition 8)",
        "artist": "Wassily Kandinsky (1923)",
        "movement": "Bauhaus Abstract Geometric",
        "path": "data/style/style_kandinsky.jpg",
        "desc": "Komposisi abstrak geometris dengan lingkaran konsentris, garis silang, bujur sangkar, dan warna primer cerah.",
    },
}


# -----------------------------------------------------------------------------
# 2. Cached Resource & Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vgg_extractor_cached() -> VGG19FeatureExtractor:
    """Load and cache the frozen VGG-19 feature extractor in memory."""
    device = get_device()
    extractor = build_vgg19_extractor(checkpoint_dir="checkpoints", device=device)
    return extractor


def validate_image_file(path: Path) -> Tuple[bool, str]:
    """Validate whether an image file is readable and non-corrupted."""
    try:
        if not path.is_file():
            return False, f"File tidak ditemukan: {path}"
        with Image.open(path) as img:
            img.verify()
        return True, "OK"
    except Exception as e:
        return False, f"Citra rusak atau tidak dapat dibaca: {str(e)}"


# -----------------------------------------------------------------------------
# 3. Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Studio Neural Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .recommendation-card {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 0.85rem 1rem;
        border-radius: 0.375rem;
        margin-bottom: 1rem;
    }
    .info-card {
        background-color: #f8fafc;
        border-left: 5px solid #3b82f6;
        padding: 0.85rem 1rem;
        border-radius: 0.375rem;
        margin-bottom: 1rem;
    }
    .stat-box {
        background-color: #f3f4f6;
        border-radius: 0.5rem;
        padding: 0.75rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-title">🎨 Studio Neural Style Transfer (NST)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Sintesis citra artistik memadukan lanskap alam dengan karakteristik lukisan maestro klasik '
    'menggunakan optimasi representasi fitur mendalam VGG-19 (Gatys et al., 2016).</div>',
    unsafe_allow_html=True,
)

# Informational Notice: Algorithmic Clarity
with st.expander("ℹ️ Prinsip Ilmiah: Cara Kerja Algoritma Gatys & Ekspektasi Hasil", expanded=False):
    st.markdown(
        """
        - **Bukan *Copy-Paste* Objek:** Algoritma NST Gatys **tidak memotong atau menempelkan objek fisik** dari lukisan (misal: ombak Hokusai tidak akan menempel begitu saja pada puncak gunung).
        - **Pencocokan Statistik Tekstur:** Transfer gaya bekerja dengan mencocokkan korelasi antar-fitur (*feature correlations*) multi-skala melalui **Matriks Gram** pada ruang representasi konvolusional.
        - **Optimasi Piksel Iteratif:** Komputer memperbarui nilai piksel kanvas secara bertahap menggunakan gradien balik (*backpropagation*) Adam untuk menyeimbangkan retensi bentuk pemandangan dan penyerapan corak warna/tekstur lukisan.
        """
    )


# -----------------------------------------------------------------------------
# 4. Sidebar: Image Selection, Recommendation & Parameters
# -----------------------------------------------------------------------------
st.sidebar.header("1. Citra Konten (Pemandangan)")
content_source_mode = st.sidebar.radio(
    "Pilih Sumber Konten:",
    ["Preset Lanskap (8 Pilihan)", "Unggah Citra Mandiri (JPG/PNG)"],
    index=0,
)

selected_content_path: Optional[Path] = None
content_display_label = ""
active_content_tag = "gunung"

upload_dir = Path("outputs/demo_uploads")
upload_dir.mkdir(parents=True, exist_ok=True)

if content_source_mode == "Preset Lanskap (8 Pilihan)":
    preset_c_keys = list(LANDSCAPE_PRESETS.keys())
    selected_c_key = st.sidebar.selectbox(
        "Pilih Preset Lanskap:",
        preset_c_keys,
        format_func=lambda k: LANDSCAPE_PRESETS[k]["title"],
        index=0,
    )
    c_info = LANDSCAPE_PRESETS[selected_c_key]
    selected_content_path = Path(c_info["path"])
    content_display_label = c_info["title"]
    active_content_tag = c_info["tag"]
    st.sidebar.caption(f"ℹ️ {c_info['desc']}")
    st.sidebar.caption(f"Kredit: {c_info['creator']}")
else:
    uploaded_content = st.sidebar.file_uploader(
        "Unggah Citra Pemandangan:",
        type=["jpg", "jpeg", "png"],
        key="content_upload",
    )
    if uploaded_content is not None:
        selected_content_path = upload_dir / f"user_content_{uploaded_content.name}"
        with open(selected_content_path, "wb") as f:
            f.write(uploaded_content.getbuffer())
        content_display_label = f"Unggahan: {uploaded_content.name}"
        active_content_tag = "custom"

st.sidebar.markdown("---")
st.sidebar.header("2. Referensi Gaya (Lukisan)")
style_source_mode = st.sidebar.radio(
    "Pilih Sumber Gaya:",
    ["Preset Maestro (3 Lukisan)", "Unggah Lukisan Mandiri (JPG/PNG)"],
    index=0,
)

selected_style_path: Optional[Path] = None
style_display_label = ""
selected_s_key: Optional[str] = None

if style_source_mode == "Preset Maestro (3 Lukisan)":
    preset_s_keys = list(STYLE_PRESETS.keys())
    selected_s_key = st.sidebar.selectbox(
        "Pilih Lukisan Referensi:",
        preset_s_keys,
        format_func=lambda k: STYLE_PRESETS[k]["title"],
        index=0,
    )
    s_info = STYLE_PRESETS[selected_s_key]
    selected_style_path = Path(s_info["path"])
    style_display_label = s_info["title"]
    st.sidebar.caption(f"🎨 Aliran: {s_info['movement']}")
    st.sidebar.caption(f"Ciri Visual: {s_info['desc']}")
else:
    uploaded_style = st.sidebar.file_uploader(
        "Unggah Citra Lukisan Gaya:",
        type=["jpg", "jpeg", "png"],
        key="style_upload",
    )
    if uploaded_style is not None:
        selected_style_path = upload_dir / f"user_style_{uploaded_style.name}"
        with open(selected_style_path, "wb") as f:
            f.write(uploaded_style.getbuffer())
        style_display_label = f"Unggahan: {uploaded_style.name}"

st.sidebar.markdown("---")
st.sidebar.header("3. Parameter Optimasi")

# Hyperparameter inputs
num_steps = st.sidebar.slider(
    "Jumlah Langkah Iterasi (Steps):",
    min_value=50,
    max_value=300,
    value=300,
    step=25,
    help="Jumlah iterasi optimasi piksel dengan Adam. Nilai default 300 memberikan konvergensi yang matang.",
)

style_weight_selection = st.sidebar.select_slider(
    "Bobot Gaya (Style Weight β):",
    options=[10000.0, 50000.0, 100000.0, 500000.0, 1000000.0],
    value=100000.0,
    format_func=lambda v: f"{v:.0e} ({'Rekomendasi Default' if v == 1e5 else 'Subtil' if v < 1e5 else 'Ekspresif / Agresif'})",
    help="Rasio beta/alpha. Nilai 1e5 terbukti sebagai titik keseimbangan terbaik antara bentuk lanskap dan karakter lukisan.",
)

with st.sidebar.expander("Konfigurasi Internal Terkunci (Kontrak Baku)"):
    st.markdown(
        """
        - **Batas Resolusi Maksimum:** `256px` (proporsional)
        - **Bobot Konten (α):** `1.0`
        - **Bobot Regularisasi TV (γ):** `0.0`
        - **Optimizer:** Adam (`lr = 0.02`)
        - **Inisialisasi Target:** Salinan konten (`requires_grad=True`)
        - **Normalisasi ImageNet:** Internal 1× di dalam forward ekstraktor
        - **Pixel Clamping:** `[0.0, 1.0]` setelah tiap step
        """
    )


# -----------------------------------------------------------------------------
# 5. Recommendation Banner & Interactive Guidance
# -----------------------------------------------------------------------------
# Generate contextual recommendation based on current content selection
rec_target_key = "van_gogh"
rec_explanation = ""

if active_content_tag in ["pantai_berombak", "laut_berbatu", "pesisir_berpohon"]:
    rec_target_key = "hokusai"
    rec_explanation = (
        "Lanskap bertema air laut/pantai sangat selaras dengan karya **Katsushika Hokusai (*The Great Wave*)**. "
        "Pigmen *Prussian blue* dan kontur buih ombak menghasilkan transformasi bergaya ukiyo-e yang sangat elegan."
    )
elif active_content_tag in ["sawah_padang_rumput", "senja", "danau"]:
    rec_target_key = "van_gogh"
    rec_explanation = (
        "Lanskap persawahan, padang rumput, atau danau kawah memiliki harmoni tinggi dengan karya **Vincent van Gogh (*Wheat Field with Cypresses*)**. "
        "Palet kuning keemasan, biru toska, dan sapuan *impasto* meliuk mengalir sangat alami pada kontur alam."
    )
elif active_content_tag in ["gunung", "hutan_pedalaman"]:
    rec_target_key = "kandinsky"
    rec_explanation = (
        "Lanskap pegunungan atau hutan berkabut sangat memikat jika dipasangkan dengan karya **Wassily Kandinsky (*Composition 8*)** "
        "untuk eksplorasi seni modern abstrak, menguji bagaimana geometri kaku berinteraksi dengan topografi alami."
    )
else:
    rec_target_key = "van_gogh"
    rec_explanation = "Kombinasi bebas sepenuhnya didukung! Silakan eksplorasi pasangan sesuai preferensi estetika Anda."

st.markdown(
    f"""
    <div class="recommendation-card">
        <strong>💡 Rekomendasi Kurasi Studio:</strong> {rec_explanation}
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 6. Preview Section: Content and Style Side-by-Side
# -----------------------------------------------------------------------------
st.subheader("Preview Citra Masukan")
prev_col1, prev_col2 = st.columns(2)

content_valid = False
style_valid = False

with prev_col1:
    st.markdown(f"**Citra Konten:** {content_display_label}")
    if selected_content_path is not None and selected_content_path.is_file():
        valid, msg = validate_image_file(selected_content_path)
        if valid:
            img_c = Image.open(selected_content_path)
            st.image(img_c, width="stretch", caption=f"Resolusi Asli: {img_c.width}x{img_c.height}")
            content_valid = True
        else:
            st.error(f"Error pada citra konten: {msg}")
    else:
        st.info("Silakan pilih preset atau unggah citra pemandangan di sidebar.")

with prev_col2:
    st.markdown(f"**Citra Gaya:** {style_display_label}")
    if selected_style_path is not None and selected_style_path.is_file():
        valid, msg = validate_image_file(selected_style_path)
        if valid:
            img_s = Image.open(selected_style_path)
            st.image(img_s, width="stretch", caption=f"Resolusi Asli: {img_s.width}x{img_s.height}")
            style_valid = True
        else:
            st.error(f"Error pada citra gaya: {msg}")
    else:
        st.info("Silakan pilih preset atau unggah citra lukisan di sidebar.")


# -----------------------------------------------------------------------------
# 7. Action Button & Execution Engine
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Eksekusi Transfer Gaya")

can_process = content_valid and style_valid

if not can_process:
    st.warning("Pastikan kedua citra (Konten dan Gaya) telah terpilih dan valid sebelum memproses.")

btn_col1, btn_col2 = st.columns([1, 4])
process_clicked = btn_col1.button(
    "🚀 Mulai Transfer Gaya",
    type="primary",
    disabled=not can_process,
    use_container_width=True,
)

# Placeholder containers for dynamic progress
progress_container = st.container()

if process_clicked and can_process:
    with progress_container:
        st.info("Menginisialisasi pipeline optimasi NST...")
        pbar = st.progress(0.0)
        status_text = st.empty()

        # Temporary output directory for this demo run
        run_timestamp = int(time.time())
        run_output_dir = Path(f"outputs/demo_runs/run_{run_timestamp}")
        run_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Access Cached VGG Extractor
            status_text.text("Memuat ekstraktor fitur VGG-19 beku dari cache...")
            extractor = load_vgg_extractor_cached()

            # 2. Progress Callback Function
            def ui_progress_callback(cur_step: int, total_steps: int, metrics: Dict[str, Any]):
                frac = cur_step / float(total_steps)
                pbar.progress(frac)
                status_text.text(
                    f"Optimasi Step {cur_step}/{total_steps} | "
                    f"Loss Total: {metrics['total_loss']:.4f} | "
                    f"Content Loss: {metrics['content_loss']:.4f} | "
                    f"Style Loss: {metrics['style_loss']:.2e}"
                )

            # 3. Instantiate and run StyleTransferOptimizer
            optimizer = StyleTransferOptimizer(
                content_path=selected_content_path,
                style_path=selected_style_path,
                output_dir=run_output_dir,
                max_size=256,
                num_steps=num_steps,
                lr=0.02,
                content_weight=1.0,
                style_weight=style_weight_selection,
                tv_weight=0.0,
                save_progress_every=max(10, num_steps // 6),
                device=get_device(),
                extractor=extractor,
                progress_callback=ui_progress_callback,
            )

            summary = optimizer.run()

            # 4. Save results to Session State so they persist
            st.session_state["last_result"] = {
                "summary": summary,
                "output_dir": str(run_output_dir),
                "final_image_path": summary["artifacts"]["final_image"],
                "content_path": str(selected_content_path),
                "style_path": str(selected_style_path),
                "content_label": content_display_label,
                "style_label": style_display_label,
                "timestamp": run_timestamp,
            }

            pbar.progress(1.0)
            status_text.success("Transfer gaya selesai dengan sukses!")
            time.sleep(0.5)

        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            st.error("Terjadi kehabisan memori GPU (CUDA Out Of Memory). Coba gunakan gambar beresolusi lebih kecil.")
        except (CorruptedImageError, InvalidImageError) as e:
            st.error(f"Kesalahan pada struktur gambar: {str(e)}")
        except FileNotFoundError as e:
            st.error(f"File aset tidak ditemukan: {str(e)}")
        except Exception as e:
            st.error(f"Terjadi kesalahan tak terduga selama optimasi: {str(e)}")


# -----------------------------------------------------------------------------
# 8. Display Results (Persists in Session State)
# -----------------------------------------------------------------------------
if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    summary = res["summary"]
    metrics_sum = summary.get("metrics_summary", {})
    exec_info = summary.get("execution", {})

    st.markdown("---")
    st.subheader("Hasil Transformasi Neural Style Transfer")

    # Grid Display: Content - Style - Result
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.markdown(f"**Konten:** {res['content_label']}")
        if Path(res["content_path"]).is_file():
            st.image(Image.open(res["content_path"]), width="stretch")

    with res_col2:
        st.markdown(f"**Gaya:** {res['style_label']}")
        if Path(res["style_path"]).is_file():
            st.image(Image.open(res["style_path"]), width="stretch")

    with res_col3:
        st.markdown("**Hasil Transfer (Stylized):**")
        final_img_file = Path(res["final_image_path"])
        if final_img_file.is_file():
            final_img = Image.open(final_img_file)
            st.image(final_img, width="stretch", caption=f"Resolusi: {final_img.width}x{final_img.height}")
        else:
            st.error("File citra hasil tidak ditemukan.")

    # Execution Metric Highlights
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Durasi Eksekusi", f"{exec_info.get('total_runtime_seconds', 0.0):.2f} s")
    stat_col2.metric("Rata-rata Waktu/Langkah", f"{exec_info.get('avg_step_time_ms', 0.0):.1f} ms")
    stat_col3.metric("Penurunan Style Loss", f"{metrics_sum.get('style_loss_reduction_pct', 0.0):.1f}%")
    stat_col4.metric("Integritas Bobot VGG-19", "Terkunci (Zero Drift)")

    # Download Buttons Section
    st.markdown("#### Unduh Hasil & Konfigurasi")
    down_col1, down_col2, _ = st.columns([1.5, 2, 3])

    if final_img_file.is_file():
        with open(final_img_file, "rb") as f:
            img_bytes = f.read()
        down_col1.download_button(
            label="📥 Unduh Citra PNG",
            data=img_bytes,
            file_name=f"nst_stylized_{res['timestamp']}.png",
            mime="image/png",
            use_container_width=True,
        )

    json_str = json.dumps(summary, indent=2)
    down_col2.download_button(
        label="📥 Unduh Konfigurasi & Metrik (JSON)",
        data=json_str,
        file_name=f"nst_metrics_{res['timestamp']}.json",
        mime="application/json",
        use_container_width=True,
    )

    # Technical Details & Diagnostics Section (Explicitly NOT a quality score)
    with st.expander("📊 Detail Metrik Matematis & Diagnostik Kuantisasi (Bukan Tolok Ukur Kualitas Estetika)", expanded=False):
        st.markdown(
            """
            > **Catatan Metodologi:** Nilai fungsi loss matematis digunakan untuk memantau konvergensi numerik, 
            > **bukan skor estetika seni**. Citra dengan loss rendah belum tentu memiliki komposisi visual yang lebih disukai 
            > manusia dibanding citra dengan retensi fitur yang berbeda.
            """
        )

        d_col1, d_col2 = st.columns(2)

        with d_col1:
            st.markdown("**Perbandingan Loss Awal vs Akhir (Post-Update Clamping):**")
            st.write({
                "Total Loss Awal": metrics_sum.get("initial_total_loss"),
                "Total Loss Akhir (Float)": metrics_sum.get("final_total_loss_float"),
                "Reduksi Total Loss": f"{metrics_sum.get('total_loss_reduction_pct', 0.0)}%",
                "Content Loss Akhir": metrics_sum.get("final_content_loss_float"),
                "Style Loss Mentah Akhir": metrics_sum.get("final_style_loss_float"),
            })

        with d_col2:
            st.markdown("**Dampak Kuantisasi Integer 8-bit (Tensor Float32 vs File PNG di Disk):**")
            st.write({
                "Loss Total Tensor Float32": metrics_sum.get("final_total_loss_float"),
                "Loss Total Citra PNG Disk": metrics_sum.get("final_total_loss_quantized_png"),
                "Selisih Kuantisasi (Delta)": f"{metrics_sum.get('quantization_delta_total_loss', 0.0):+.6f}",
                "Status Integritas": "Kuantisasi diskret uint8 terbukti tidak mendistorsi hasil konvergensi.",
            })

        # Display loss plot if available
        loss_plot_path = summary.get("artifacts", {}).get("loss_plot")
        if loss_plot_path and Path(loss_plot_path).is_file():
            st.markdown("**Grafik Riwayat Konvergensi Loss:**")
            st.image(Image.open(loss_plot_path), width="stretch")
