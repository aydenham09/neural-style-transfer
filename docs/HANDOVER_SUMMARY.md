# Ringkasan Serah Terima (Handover Summary) — Tahap Fungsi Loss Kustom, Ekstraktor Fitur, & Preprocessing

Dokumen ini merangkum status proyek Neural Style Transfer (NST), penambahan file, panduan eksekusi di perangkat lokal, serta catatan teknis penting terkini.

---

## 1. Daftar File Proyek

### File Utama Kode (`src/`):
- `src/utils.py`: Modul utilitas untuk:
  - Deteksi device akselerasi komputasi (`get_device()`) dengan prioritas CUDA $\rightarrow$ MPS $\rightarrow$ CPU.
  - Resize proporsional dengan batas maksimal 256 piksel tanpa memperbesar gambar kecil (`load_image()`, `calculate_proportional_size()`).
  - Konversi citra ke tensor PyTorch `1x3xHxW` bernilai `0.0 - 1.0` (`image_to_tensor()`).
  - Normalisasi ImageNet dengan proteksi normalisasi ganda (`normalize_imagenet()`, `is_normalized()`).
  - Denormalisasi balik dan rekonstruksi citra untuk preview/simpan (`denormalize_imagenet()`, `tensor_to_image()`, `save_image()`).
  - Pembuat visualisasi grid dataset (`create_dataset_preview()`).
  - Exception kustom (`CorruptedImageError`, `InvalidImageError`, `DoubleNormalizationError`).
- `src/model.py`: Ekstraktor fitur VGG-19 pretrained ImageNet mandiri:
  - Menggunakan `VGG19_Weights.IMAGENET1K_V1` dengan cache lokal di `checkpoints/`.
  - Mengambil convolutional features saja, dipangkas sampai layer `conv5_1` (indeks 29, 30 layer total).
  - Mode evaluasi ketat (`eval()`) dan pembekuan seluruh parameter (`param.requires_grad = False`).
  - Menonaktifkan in-place activation pada semua `nn.ReLU` (`inplace = False`) demi integritas autograd.
  - Representasi Konten dari layer `conv4_2` (post-ReLU / `relu4_2`, indeks 22).
  - Representasi Gaya dari layer `conv1_1` (idx 1), `conv2_1` (idx 6), `conv3_1` (idx 11), `conv4_1` (idx 20), `conv5_1` (idx 29).
  - Fungsi proteksi normalisasi ImageNet yang kuat (`is_tensor_imagenet_normalized`), mendeteksi dari distribusi nilai numerik sehingga tahan terhadap operasi tensor yang menghilangkan atribut Python.
  - Aliran gradien dipastikan mencapai tensor citra input yang dioptimasi tanpa `torch.no_grad()`.
- `src/loss.py`: Modul fungsi loss kustom dan korelasi tekstur:
  - `gram_matrix`: Perhitungan perkalian matriks saluran $G = F F^T / (C \cdot H \cdot W)$ dengan invariansi skala resolusi dan verifikasi simetris $G = G^T$.
  - `ContentLoss`: Perhitungan MSE berfaktor $0.5$ terhadap fitur konten pada `conv4_2`.
  - `SingleLayerStyleLoss` & `StyleLoss`: Perhitungan loss gaya multi-skala berbobot pada 5 layer konvolusi, mendukung ukuran citra konten dan gaya yang berbeda tanpa kendala dimensi.
  - `TotalVariationLoss`: Regularisasi penalti spasial piksel tetangga untuk meredam noise frekuensi tinggi.
  - `NSTLoss`: Composite loss manager terpadu yang menggabungkan $\alpha \mathcal{L}_{\text{content}} + \beta \mathcal{L}_{\text{style}} + \gamma \mathcal{L}_{\text{tv}}$ beserta metriks breakdown.
- `src/__init__.py`: Mengekspor modul dan fungsi-fungsi kunci dari `src.utils`, `src.model`, dan `src.loss`.

### Skrip Validasi & Data (`scripts/`):
- `scripts/download_dataset.py`: Mengunduh 3 gambar lanskap dan 3 lukisan maestro dengan header User-Agent resmi dan jeda inter-request.
- `scripts/verify_extractor.py`: Skrip verifikasi fitur VGG-19, pembekuan bobot, dan aliran gradien input.
- `scripts/verify_loss.py`: Skrip verifikasi fungsi loss, simetri Gram Matrix, TV loss, dan backpropagation end-to-end.

### Pengujian Otomatis (`tests/`):
- `tests/test_preprocessing.py`: 9 unit test preprocessing.
- `tests/test_model.py`: 8 unit test model VGG-19.
- `tests/test_loss.py`: 7 unit test Gram Matrix dan seluruh varian loss.
- **Total:** **24 unit test aktif, seluruhnya berstatus OK (100% Pass) dalam ~2.6 detik**.

### Dokumentasi & Metadata:
- `docs/HANDOVER_SUMMARY.md`: Dokumen serah terima ini.
- `docs/session_02_dataset_preprocessing.md`: Log Sesi 02 (Dataset & Preprocessing).
- `docs/session_03_vgg19_feature_extractor.md`: Log Sesi 03 (Ekstraktor Fitur VGG-19).
- `docs/session_04_loss_functions.md`: Log Sesi 04 (Fungsi Loss Kustom).
- `data/metadata/dataset_metadata.yaml`: Metadata resolusi, lisensi, dan URL resmi citra.
- `data/metadata/styles_info.yaml`: Karakteristik visual 3 lukisan maestro (Van Gogh, Hokusai, Kandinsky).

---

## 2. Cara Menjalankan

### Aktivasi Virtual Environment
```bash
# Di macOS / Linux:
source .venv/bin/activate

# Di Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### Menjalankan Seluruh Unit Test (24 Tests)
```bash
python -m unittest discover tests
```

### Menjalankan Verifikasi Ekstraktor VGG-19
```bash
python scripts/verify_extractor.py
```

### Menjalankan Verifikasi Fungsi Loss & Gram Matrix
```bash
python scripts/verify_loss.py
```

---

## 3. Lingkungan Komputasi & Akselerasi Perangkat

- **Sistem Operasi:** macOS Darwin (Apple Silicon `arm64`)
- **Akselerasi Aktif:** **`mps` (`Apple Metal Performance Shaders`)** — terdeteksi otomatis dan aktif pada seluruh pipeline.
- **CPU Fallback:** Mendukung fallback otomatis ke `cpu`.
- **Python Runtime:** Python 3.13.15
- **PyTorch:** 2.14.0 | **Torchvision:** 0.29.0

---

## 4. Rencana Tahapan Berikutnya (Fase 5)
- **Implementasi `src/transfer.py` (Engine Optimasi NST & Loop Transfer Gaya):**
  1. Mekanisme inisialisasi gambar target yang dapat dipilih (`content`, `random_noise`, atau `style`).
  2. Implementasi optimizer **L-BFGS** dengan fungsi `closure()` yang mengevaluasi loss berulang kali per langkah optimasi quasi-Newton.
  3. Pilihan alternatif optimizer **Adam** untuk eksperimen perbandingan kecepatan dan kestabilan.
  4. Penjagaan batas piksel $[0.0, 1.0]$ via `clamp_(0.0, 1.0)` di setiap iterasi.
  5. Pencatatan riwayat loss (*loss history tracking*), progress bar `tqdm`, dan penyimpanan gambar hasil intermediate ke `outputs/`.
