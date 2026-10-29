# Neural Style Transfer (NST): Lanskap Alami × Maestro Klasik

Proyek Generative AI berbasis **Neural Style Transfer (NST)** menggunakan Python dan PyTorch untuk mentransformasi foto pemandangan alam menjadi karya seni bercorak lukisan maestro klasik:
1. **Vincent van Gogh** — *Wheat Field with Cypresses* (Post-Impressionism / Impasto)
2. **Katsushika Hokusai** — *The Great Wave off Kanagawa* (Ukiyo-e / Cetak Balok Kayu)
3. **Wassily Kandinsky** — *Composition 8* (Bauhaus Abstract Geometric)

Proyek ini dilengkapi **Demo Studio Interaktif Berbasis Streamlit (`app.py`)**, engine optimasi Adam piksel-level modular (`src/transfer.py`), ekstraktor fitur VGG-19 beku mandiri (`src/model.py`), 8 preset pemandangan alam beresolusi tinggi, serta sistem evaluasi komparatif matematis dan perseptual.

---

## 🧠 Konsep & Filosofi Arsitektur

### 1. Peran Pretrained Model VGG-19 (ImageNet)
> **Penting untuk Dipahami:**  
> Proyek ini **TIDAK** melatih jaringan neural convolutional (CNN) dari awal (*from scratch*).  
> Sebaliknya, kita memanfaatkan **VGG-19** yang telah dilatih pada dataset **ImageNet** (1,2 juta gambar, 1000 kelas).

- **Mengapa VGG-19?**  
  Filter konvolusi VGG-19 telah mempelajari hierarki representasi visual dunia nyata secara berjenjang:
  - *Layer awal (`conv1_1`, `conv2_1`):* Menangkap fitur frekuensi tinggi, tepi (*edges*), garis halus, dan warna lokal.
  - *Layer menengah (`conv3_1`, `conv4_1`):* Menangkap ornamen permukaan dan pola tekstur yang lebih kompleks.
  - *Layer dalam (`conv4_2`, `conv5_1`):* Menangkap semantik tingkat tinggi (*high-level layout*), seperti siluet gunung, bentangan danau, atau cakrawala.
- **Bobot Dibekukan (*Frozen Weights*):**  
  Seluruh parameter bobot ($W$) dan bias ($b$) dari VGG-19 dikunci secara ketat (`param.requires_grad = False`). Model berfungsi murni sebagai **ekstraktor fitur statis**. Checksum SHA-256 diverifikasi sebelum dan sesudah optimasi untuk memastikan *zero drift*.

### 2. Apa yang Sebenarnya Dioptimasi?
Dalam formulasi orisinal Gatys et al. (2015/2016):
- Yang dioptimasi **bukanlah bobot model**, melainkan **nilai piksel citra kanvas target itu sendiri** ($I_{\text{target}}$).
- Kanvas target diinisialisasi dari salinan independen citra konten (`target = content.clone().detach().requires_grad_(True)`).
- Melalui *backpropagation* menggunakan optimizer Adam (`lr = 0.02`), gradien dari fungsi loss gabungan diarahkan langsung untuk memperbarui piksel target:
  $$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{content}} + \beta \mathcal{L}_{\text{style}} + \gamma \mathcal{L}_{\text{tv}}$$
- Clamping piksel ke rentang $[0.0, 1.0]$ dilakukan secara in-place di bawah `torch.no_grad()` setelah setiap update langkah.

### 3. Kontrak Normalisasi ImageNet Baku
- **Input Ekstraktor:** Wajib berupa raw RGB float $[0.0, 1.0]$.
- **Normalisasi:** Dilakukan tepat **satu kali** di dalam `VGG19FeatureExtractor.forward()` menggunakan statistik ImageNet ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$). Tidak ada tebakan berbasis rentang nilai numerik.

---

## 🎨 Demo Studio Interaktif (Streamlit)

Proyek ini menyediakan antarmuka web interaktif yang siap dijalankan secara lokal:

```powershell
# Menjalankan studio demo NST
.\.venv\Scripts\streamlit.exe run app.py
```

### Fitur-Fitur Studio:
- **Koleksi Preset Luas:** Pilihan **8 preset pemandangan alam** (Gunung, Danau, Pesisir Berpohon, Pantai Berombak, Laut Berbatu, Sawah/Padang Berundak, Senja, Hutan Pedalaman) serta **3 lukisan maestro klasik**.
- **Unggah Citra Mandiri:** Dukungan drag-and-drop untuk file JPG/PNG kustom baik untuk konten maupun gaya.
- **Rekomendasi Cerdas:** Sistem memandu rekomendasi pasangan estetik (misal: Laut/Pantai $\rightarrow$ Hokusai; Sawah/Padang $\rightarrow$ Van Gogh; Gunung/Danau $\rightarrow$ Kandinsky) tanpa membatasi kebebasan kombinasi pengguna.
- **Kontrol Parameter:** Slider langkah iterasi (50 s/d 300) dan pengatur bobot gaya $\beta$ ($10^4$ s/d $10^6$).
- **Caching Model Ekstraktor:** VGG-19 di-cache menggunakan `st.cache_resource` sehingga startup optimasi instan.
- **Visualisasi Dinamis & Download:** Progres real-time, komparasi Konten - Gaya - Hasil, unduh hasil citra PNG resolusi penuh, serta ekspor konfigurasi/metrik JSON.
- **Detail Teknis Objektif:** Menampilkan metrik loss matematis dan analisis galat kuantisasi integer 8-bit tanpa klaim subyektif "skor kualitas".
- **Tahan Banting (*Robust Error Handling*):** Penanganan anggun terhadap citra rusak, aset hilang, dan proteksi OOM (CUDA Out Of Memory).
- **Status Verifikasi:** Fitur unduh citra PNG dan konfigurasi/metrik JSON telah diperiksa secara manual dan berhasil. Probe server headless terkonfirmasi `HTTP 200 OK`. Fitur interaktif antarmuka browser lainnya belum diperiksa secara manual via browser GUI.

---

## 📂 Struktur Direktori Proyek

```text
neural-style-transfer/
├── app.py                            # Aplikasi Studio Demo Interaktif Streamlit
├── requirements.txt                  # Dependensi Python utama
├── checkpoints/                      # Direktori cache bobot VGG-19 (vgg19-dcbb9e9d.pth)
├── data/
│   ├── content/                      # 8 Foto Lanskap Pemandangan Alami (CC0 / Public Domain)
│   │   ├── landscape_mountain.jpg          # Mount Rainier (Gunung)
│   │   ├── landscape_lake.jpg              # Crater Lake (Danau)
│   │   ├── landscape_forest.jpg            # Pesisir Berpohon Redwood (Coastal Shore)
│   │   ├── landscape_beach_waves.jpg       # Pantai Berombak (Crashing Waves)
│   │   ├── landscape_rocky_sea.jpg         # Laut Berbatu (Rocky Sea Coast)
│   │   ├── landscape_rice_terrace.jpg      # Sawah / Padang Berundak (Rice Terraces)
│   │   ├── landscape_sunset.jpg            # Senja Keemasan (Golden Sunset)
│   │   └── landscape_interior_forest.jpg   # Hutan Pedalaman Berkabut (Misty Forest)
│   ├── style/                        # 3 Lukisan Referensi Maestro (Public Domain / CC0)
│   │   ├── style_van_gogh.jpg              # Van Gogh (Wheat Field with Cypresses)
│   │   ├── style_hokusai.jpg               # Hokusai (The Great Wave off Kanagawa)
│   │   └── style_kandinsky.jpg             # Kandinsky (Composition 8)
│   └── metadata/
│       ├── dataset_metadata.yaml     # Metadata dimensi, lisensi, kredit, dan URL resmi
│       └── styles_info.yaml          # Dokumentasi referensi karakteristik visual (diselaraskan baseline 1e5 & conv1_1..conv5_1)
├── src/                              # Modul Kode Inti PyTorch
│   ├── __init__.py                   # Package exports
│   ├── utils.py                      # Device detection, loader, validator, normalizer, visualizer
│   ├── model.py                      # VGG19FeatureExtractor mandiri dengan frozen weights
│   ├── loss.py                       # GramMatrix, ContentLoss, StyleLoss, TVLoss, NSTLoss
│   └── transfer.py                   # StyleTransferOptimizer: engine optimasi piksel Adam
├── scripts/                          # Skrip Otomasi dan Evaluasi
│   ├── download_dataset.py           # Pengunduh dataset awal
│   ├── download_additional_presets.py# Pengunduh 5 preset pemandangan tambahan
│   ├── run_3x3_matrix_evaluation.py  # Eksekusi evaluasi matriks lengkap 9 pasangan
│   ├── generate_style_weight_comparison.py # Eksekusi komparasi bobot gaya (1e4, 1e5, 1e6)
│   ├── test_demo_recommendations.py  # Pengujian 3 pasangan rekomendasi showcase demo
│   └── probe_streamlit_server.py     # Verifikasi headless HTTP server Streamlit
├── tests/                            # Test Suite Komprehensif (30 Unit Tests Aktif)
│   ├── test_preprocessing.py         # 9 tests pra-pemrosesan citra
│   ├── test_model.py                 # 8 tests ekstraktor VGG-19 & normalisasi
│   ├── test_loss.py                  # 7 tests Gram matrix, Content, Style, & TV Loss
│   ├── test_transfer.py              # 2 tests engine optimasi & artefak
│   └── test_demo_workflow.py         # 4 tests alur demo Streamlit & download
├── outputs/                          # Hasil Eksperimen & Visualisasi (Diabaikan git)
│   ├── experiments/                  # Output tiap batch eksperimen
│   │   ├── exp01_mountain_van_gogh_adam/
│   │   ├── exp02_mountain_van_gogh_style1e4/
│   │   ├── exp03_mountain_van_gogh_style1e6/
│   │   ├── exp04_mountain_hokusai/ s/d exp11_forest_kandinsky/
│   │   ├── style_weight_comparison/
│   │   └── evaluation_3x3_matrix/
│   └── demo_showcases/               # Output hasil pengujian pasangan rekomendasi demo
├── experiments/                      # Dokumentasi & Log Eksperimen
│   ├── experiments_log.md            # Rekapitulasi tabel seluruh percobaan
│   ├── exp01_mountain_van_gogh_adam.md
│   ├── exp02_exp03_style_weight_comparison.md
│   └── exp04_exp11_matrix_3x3_evaluation.md
└── docs/
    └── HANDOVER_SUMMARY.md           # Ringkasan serah terima proyek komprehensif
```

---

## ⚙️ Lingkungan Komputasi & Dependensi

- **Sistem Operasi:** Windows 11 (AMD64 / `win32`)
- **Akselerasi GPU:** **NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)** dengan CUDA 12.1 / 12.3
- **Python Runtime:** Python 3.11.6 di virtual environment [`.venv/`](file:///d:/neural-style-transfer/.venv)
- **Pustaka Utama:** PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Streamlit 1.66.1, Pillow 13.0.0, NumPy 2.4.6, Matplotlib 3.11.2, PyYAML 6.0.3

---

## 🚀 Panduan Eksekusi

### 1. Aktivasi Virtual Environment (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Menjalankan Seluruh Unit Test (30 Tests)
```powershell
python -m unittest discover tests
```

### 3. Menjalankan Studio Demo Streamlit
```powershell
streamlit run app.py
```
Akses aplikasi melalui browser pada alamat default: `http://localhost:8501`.

### 4. Menjalankan Optimasi NST Mandiri (CLI)
```powershell
python src/transfer.py --content data/content/landscape_mountain.jpg --style data/style/style_van_gogh.jpg --output-dir outputs/experiments/my_run --steps 300 --lr 0.02 --content-weight 1.0 --style-weight 100000.0 --tv-weight 0.0
```

---

## 📊 Ringkasan Temuan Eksperimen

1. **Trade-off Bobot Gaya ($\beta$):**
   - $\beta = 10^4$: Bentuk gunung utuh sempurna ($\mathcal{L}_c = 0.028$), namun gaya lukisan sangat lemah (*under-styled*).
   - $\beta = 10^5$: **Titik temu terbaik (*sweet spot*)** untuk foto pemandangan realistis; siluet lanskap tetap kokoh sementara sapuan kuas dan palet warna maestro terserap proporsional.
   - $\beta = 10^6$: Penyerapan gaya sangat agresif (reduksi gaya $94.5\%$), namun memicu distorsi struktural topografi gunung ($\mathcal{L}_c = 2.038$).
2. **Karakteristik 3 Aliran Gaya:**
   - *Van Gogh (Impasto):* Sangat ramah terhadap struktur lanskap ($\mathcal{L}_c \approx 0.23$), mengalirkan tekstur kuas meliuk pada langit dan air.
   - *Hokusai (Ukiyo-e):* Mengubah lanskap laut dan pegunungan menjadi karya cetak kayu dramatis dengan pigmen Prussian blue dan kontur buih ombak tajam.
   - *Kandinsky (Abstrak Geometris):* Menciptakan karya modern yang berani, namun menuntut penalti konten tinggi ($\mathcal{L}_c \approx 1.57 - 1.66$) akibat benturan bentuk sintetis dengan kurva alami.
3. **Analisis Kuantisasi Integer 8-bit:**
   - Kuantisasi uint8 pada file PNG disk hanya menimbulkan selisih loss marginal ($\Delta \mathcal{L}_{\text{total}} \le +0.0020$, deviasi $< 0.09\%$), membuktikan file PNG tersimpan mencerminkan hasil optimasi memori GPU secara presisi.
