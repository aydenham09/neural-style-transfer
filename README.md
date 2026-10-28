# Neural Style Transfer (NST): Landscape to Classical Art Paintings

Proyek Generative AI berbasis **Neural Style Transfer (NST)** menggunakan Python dan PyTorch untuk mentransformasi foto lanskap/pemandangan alam menjadi karya seni dengan karakteristik lukisan maestro klasik:
1. **Vincent van Gogh** — *Wheat Field with Cypresses* (Post-Impressionism)
2. **Katsushika Hokusai** — *The Great Wave off Kanagawa* (Ukiyo-e)
3. **Wassily Kandinsky** — *Composition 8* (Abstract Art)

---

## 🧠 Konsep & Filosofi Arsitektur

### 1. Peran Pretrained Model VGG-19 (ImageNet)
> **Penting untuk Dipahami:**  
> Proyek ini **TIDAK** melatih jaringan neural convolutional (CNN) dari nol (*from scratch*).  
> Sebaliknya, kita memanfaatkan **VGG-19** yang telah dilatih sebelumnya (*pre-trained*) pada dataset raksasa **ImageNet** (1,2 juta gambar, 1000 kelas objek).

- **Mengapa VGG-19?**  
  Filter konvolusi VGG-19 telah belajar mengenali hierarki representasi visual dunia nyata secara natural:
  - *Layer awal (conv1, conv2):* Mendeteksi tepi (*edges*), garis, tekstur mikro, dan warna lokal.
  - *Layer menengah (conv3, conv4):* Mendeteksi pola permukaan, ornamen, dan struktur tekstur yang lebih kompleks.
  - *Layer dalam (conv5):* Menangkap semantik tingkat tinggi (*high-level semantic layout*), seperti keberadaan gunung, pohon, atau langit tanpa mempedulikan tekstur piksel mentah.
- **Bobot Dibekukan (*Frozen Weights*):**  
  Seluruh parameter bobot ($W$) dan bias ($b$) dari VGG-19 diset `requires_grad = False`. Model ini murni berfungsi sebagai **ekstraktor fitur statis** (*frozen feature extractor*). Jaringan tidak diperbarui sama sekali.

### 2. Apa yang Sebenarnya Dioptimasi?
Dalam Neural Style Transfer klasik (Gatys et al., 2015/2016):
- Yang dioptimasi **bukanlah bobot model**, melainkan **nilai piksel dari gambar itu sendiri** ($I_{\text{target}}$).
- Gambar target diinisialisasi (misalnya dari salinan gambar konten atau random Gaussian noise) dan diperlakukan sebagai parameter tensor yang memerlukan gradien (`requires_grad = True`).
- Melalui *backpropagation* (menggunakan optimizer seperti **L-BFGS** atau **Adam**), gradien dari *loss function* diarahkan langsung untuk memodifikasi piksel gambar target hingga meminimalkan selisih konten dan selisih gaya secara simultan.

### 3. Komponen Kustom yang Kita Implementasikan Sendiri
Kami membangun seluruh komponen inti secara modular dari dasar (bukan mengandalkan black-box wrapper atau kode tutorial generik):
- ✅ **Image Preprocessing & Postprocessing:** Transformasi resolusi, standardisasi dimensi BCHW, normalisasi ImageNet ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$), denormalisasi balik, dan clamping piksel $[0, 1]$.
- ✅ **Gram Matrix:** Perhitungan matriks korelasi fitur antar-kanal ($G_{ij} = \sum_k F_{ik} F_{jk}$) untuk menangkap tekstur gaya lukisan tanpa bergantung pada posisi spasial objek.
- ✅ **Content Loss:** Formulasi Mean Squared Error (MSE) antara feature map konten dan target pada layer representasi semantik (misal `conv4_2`).
- ✅ **Style Loss:** Formulasi bobot multi-skala MSE antara Gram Matrix gaya dan target di berbagai kedalaman layer konvolusi.
- ✅ **Total Variation (TV) Regularization:** Regularisasi penalti spasial untuk meredam noise bintik frekuensi tinggi agar hasil lukisan lebih halus dan koheren.
- ✅ **Optimization Engine:** Loop optimasi modular yang mendukung L-BFGS dengan *closure* dan Adam, dilengkapi *checkpointing* serta *loss tracking*.

---

## 📂 Struktur Direktori Proyek

```text
NST/
├── .gitignore                    # Pengecualian environment, dataset gambar, model weights, & outputs
├── README.md                     # Dokumentasi arsitektur, panduan, dan konsep NST
├── requirements.txt              # Daftar pustaka Python yang dibutuhkan
├── data/
│   ├── content/                  # Tempat menyimpan foto pemandangan alam (diabaikan oleh git)
│   ├── style/                    # Tempat menyimpan 3 lukisan referensi gaya (diabaikan oleh git)
│   └── metadata/
│       └── styles_info.yaml      # Deskripsi karakteristik visual tiap gaya lukisan
├── src/                          # Modul kode implementasi PyTorch mandiri
│   ├── __init__.py               # Package marker
│   ├── utils.py                  # Load/save image, device selector (MPS/CUDA/CPU), visualizer
│   ├── model.py                  # Ekstraktor fitur VGG-19 terpilih & frozen
│   ├── loss.py                   # Gram Matrix, ContentLoss, StyleLoss, & TotalVariationLoss
│   └── transfer.py               # Engine loop optimasi piksel (L-BFGS & Adam)
├── checkpoints/                  # Tempat penyimpanan intermediate tensor / state (opsional)
├── outputs/                      # Hasil render lukisan dan plot loss (diabaikan oleh git)
├── experiments/                  # Catatan tracking eksperimen & konfigurasi
│   ├── EXPERIMENT_TEMPLATE.md    # Template laporan setiap run eksperimen
│   └── experiments_log.md        # Rekap tabel eksperimen
└── docs/
    └── INTERACTION_LOG_TEMPLATE.md # Template log interaksi & keputusan vibe coding
```

---

## ⚙️ Persiapan & Lingkungan Komputasi

### Status Perangkat Terverifikasi
- **Sistem Operasi:** macOS Darwin (Apple Silicon `arm64`)
- **Python:** Python 3.13.x
- **Akselerasi Perangkat:** **Apple Silicon MPS (Metal Performance Shaders)** terdeteksi aktif (`torch.backends.mps.is_available() == True`).

### Aktivasi Virtual Environment
```bash
# Mengaktifkan virtual environment proyek
source .venv/bin/activate

# Memeriksa kesiapan PyTorch & akselerasi MPS
python -c "import torch; print('PyTorch:', torch.__version__, '| Akselerasi MPS Aktif:', torch.backends.mps.is_available())"
```

---

## 🧭 Rencana Tahapan Kerja (Roadmap)

1. **Fase 1: Persiapan Dasar (Saat Ini - Selesai)**
   - Audit folder dan proteksi git
   - Setup virtual environment, requirements, gitignore, dan template eksperimen
   - Perumusan arsitektur modular dan pemahaman konsep pretrained VGG-19
2. **Fase 2: Utilitas Gambar & Pra-pemrosesan (`src/utils.py`)**
   - Loader gambar dengan penyesuaian rasio aspek
   - Normalisasi ImageNet dan konversi tensor PyTorch
   - Post-processing (denormalisasi dan visualisasi Matplotlib)
3. **Fase 3: Ekstraktor Fitur VGG-19 (`src/model.py`)**
   - Download selektif bobot VGG-19 pretrained ImageNet
   - Pembekuan seluruh gradien (`param.requires_grad = False`)
   - Pemotongan layer yang tidak dibutuhkan (hanya mempertahankan `features` sampai `conv5_1`)
4. **Fase 4: Fungsi Loss Kustom & Gram Matrix (`src/loss.py`)**
   - Implementasi Gram Matrix yang dinormalisasi
   - Content Loss layer (representasi semantik)
   - Multi-layer Style Loss
   - Total Variation Loss
5. **Fase 5: Engine Optimasi & Pipeline Transfer (`src/transfer.py`)**
   - Rekonstruksi loop optimasi L-BFGS & Adam
   - Pengujian terhadap 3 gaya (Van Gogh, Hokusai, Kandinsky)
   - Evaluasi visual, tuning rasio $\alpha / \beta$, dan dokumentasi hasil

