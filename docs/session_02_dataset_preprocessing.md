# Log Interaksi & Pengembangan NST — Sesi 02

Dokumen ini mencatat kelanjutan pekerjaan Neural Style Transfer (NST) pada tahap penyiapan environment perangkat baru, dataset gambar (konten & gaya), modul preprocessing PyTorch, unit test, dan validasi visual.

---

## Ringkasan Sesi

- **ID Sesi / Tanggal:** `2026-10-28 - Sesi 02`
- **Fokus / Tujuan:** Audit & inisialisasi environment Windows, pengadaan dataset resmi (3 pemandangan alam, 3 maestro seni klasik), implementasi modul preprocessing citra (`src/utils.py`), penanganan pencegahan normalisasi ganda, pengujian unit komprehensif, dan pembuatan visualisasi preview.
- **Status:** [x] Selesai

---

## 1. Prompt / Arahan Pengguna
> Melanjutkan project NST pemandangan dari repo. Baca README, log interaksi, dan kode yang sudah ada tanpa menimpa pekerjaan sebelumnya atau mengubah identitas Git.  
> Cek lingkungan perangkat ini dan siapkan virtual environment sendiri dari dependensi project (jangan menganggap path/device dari komputer sebelumnya berlaku).  
> Kerjakan dataset dan preprocessing:
> - 3 foto lanskap: gunung, danau, hutan.
> - 3 referensi lukisan: Van Gogh (*Wheat Field with Cypresses*), Hokusai (*The Great Wave*), Kandinsky (*Composition 8*).
> - Utamakan sumber resmi dan verifikasi izin/lisensi. Catat di `data/metadata/`.
> - Buat pembaca RGB, resize proporsional maks 256 px tanpa memperbesar gambar kecil, konversi tensor 1×3×H×W bernilai 0–1, normalisasi ImageNet dan konversi balik untuk preview. Bedakan tensor mentah dan ternormalisasi agar tidak terjadi normalisasi ganda. Konten dan gaya boleh berbeda ukuran.
> - Uji pemuatan, ukuran tensor, rentang nilai, normalisasi–denormalisasi, penyimpanan, dan penanganan file rusak. Tampilkan preview. Gunakan akselerasi yang benar-benar tersedia di perangkat ini.
> - Simpan log sesi dan ringkasan serah terima (file berubah, cara menjalankan, kendala aktual). Jangan komit atau push otomatis.

---

## 2. Audit Lingkungan Perangkat (Environment Discovery)
Berbeda dengan catatan komputer sebelumnya pada README (yang mencatat macOS Apple Silicon arm64 dengan MPS):
- **Sistem Operasi Aktual:** Windows 10/11 x64 (Build kernel NT)
- **CPU:** Intel Core Ultra (Family 6 Model 170, 22 logical cores)
- **GPU Akselerasi:** Tidak terpasang kartu grafis NVIDIA (`nvidia-smi` not found / No CUDA Toolkit); MPS tidak didukung di Windows.
- **PyTorch Device Aktif:** `cpu` (`torch.device("cpu")`), diverifikasi via `torch.cuda.is_available() == False` dan `torch.backends.mps.is_available() == False`.
- **Python Runtime:** Python 3.12.4 (`C:\Program Files\Python312\python.exe`)
- **Virtual Environment Proyek:** Dibuat baru di `.venv/` dengan seluruh dependensi `requirements.txt` terinstal (`torch==2.14.1+cpu`, `torchvision==0.29.1`, `Pillow==13.0.0`, `numpy==2.5.4`, `matplotlib==3.11.2`, `pyyaml==6.0.3`, `tqdm==4.70.1`).

---

## 3. Keputusan Desain & Arsitektur Preprocessing (`src/utils.py`)
1. **Device Selector Dinamis (`get_device`):**
   - Mendeteksi ketersediaan secara berurutan: CUDA -> MPS -> CPU fallback.
   - Di perangkat saat ini, secara otomatis dan aman menggunakan `cpu`.
2. **Resize Proporsional Tanpa Upscaling (`calculate_proportional_size`):**
   - Jika $\max(W, H) \le 256$, ukuran asli dipertahankan secara utuh (tidak diperbesar).
   - Jika $\max(W, H) > 256$, diskalakan proporsional sehingga sisi terpanjang tepat bernilai 256 px, menjaga aspek rasio lanskap tanpa distorsi.
   - Interpolasi menggunakan `PIL.Image.Resampling.LANCZOS` untuk kualitas rekonstruksi tinggi.
3. **Standarisasi Dimensi Tensor (`image_to_tensor`):**
   - Format: `[B, C, H, W]` = `[1, 3, H, W]`, tipe data `torch.float32`.
   - Rentang nilai piksel mentah (*raw*): strictly $[0.0, 1.0]$.
   - Ukuran konten dan gaya diperbolehkan berbeda (misal: gunung $144 \times 256$, Van Gogh $204 \times 256$).
4. **Pencegahan Normalisasi Ganda (Double Normalization Guard):**
   - Disediakan fungsi pengecekan `is_normalized(tensor)`.
   - Menggunakan penanda atribut `tensor._is_imagenet_normalized = True/False` serta inspeksi nilai minimum ($< -0.05$ mengindikasikan normalisasi ImageNet karena mean subtraction).
   - Fungsi `normalize_imagenet(tensor)` memvalidasi rentang $[0, 1]$ dan melempar exception `DoubleNormalizationError` jika tensor sudah pernah dinormalisasi.
5. **Denormalisasi & Rekonstruksi Preview (`denormalize_imagenet` & `tensor_to_image`):**
   - Mengembalikan nilai fitur dari distribusi ImageNet $(\mu, \sigma)$ kembali ke $[0.0, 1.0]$ via $I = (I_{\text{norm}} \times \sigma) + \mu$, di-clamp ke $[0, 1]$, dan dikonversi ke array NumPy `uint8` $[0, 255]$ mode RGB.
6. **Penanganan Error Robust:**
   - Exception custom: `CorruptedImageError`, `InvalidImageError`, `DoubleNormalizationError`.
   - File tidak ada melempar `FileNotFoundError`. File 0 byte, file teks/sampah biner yang menyamar sebagai jpg, atau header truncated ditangkap dan melempar `CorruptedImageError`.

---

## 4. Dataset yang Diunduh & Metadata Resmi

Seluruh dataset diunduh langsung dari repositori museum dan arsip institusi publik resmi:

| Kategori | File | Judul Karya / Foto | Pembuat / Sumber | Dimensi Asli | Lisensi |
|---|---|---|---|---|---|
| **Content** | `data/content/landscape_mountain.jpg` | View of Mount Rainier | WqemenW / Wikimedia Commons | $3520 \times 1980$ | CC0 1.0 Universal Public Domain |
| **Content** | `data/content/landscape_lake.jpg` | Crater Lake from Rim | U.S. Geological Survey (USGS) | $3275 \times 2160$ | Public Domain (U.S. Govt) |
| **Content** | `data/content/landscape_forest.jpg` | Coastal Redwood Forest (REDW9339) | National Park Service (NPS) Archive | $2960 \times 2014$ | Public Domain (U.S. Govt) |
| **Style** | `data/style/style_van_gogh.jpg` | Wheat Field with Cypresses (1889) | Vincent van Gogh / The Met Museum | $4000 \times 3184$ | Met Open Access (CC0 / Public Domain) |
| **Style** | `data/style/style_hokusai.jpg` | Under the Wave off Kanagawa (The Great Wave) | Katsushika Hokusai (ca. 1830) / The Met & LoC | $3859 \times 2594$ | Public Domain |
| **Style** | `data/style/style_kandinsky.jpg` | Composition 8 (Komposition 8) (1923) | Wassily Kandinsky / Guggenheim Museum | $3911 \times 2849$ | Public Domain |

Metadata lengkap tersimpan di `data/metadata/dataset_metadata.yaml`.

---

## 5. Kendala Aktual & Solusi (Troubleshooting)

1. **Kendala 1: HTTP Error 410 & 429 pada Scraping API The Met**
   - *Error:* `HTTP Error 410: Gone` pada endpoint pencarian REST The Met dan `HTTP Error 429: Too Many Requests` saat scraping HTML.
   - *Penyebab:* Endpoint pencarian publik The Met telah diperbarui/dibatasi rate limit ketat untuk bot.
   - *Solusi:* Menggunakan endpoint object direct ID yang valid (`DP-42549-001.jpg` untuk Van Gogh Object 436535) serta CDN Wikimedia Commons yang menyimpan digital scan master resolusi tinggi dari The Met dan Library of Congress dengan header User-Agent resmi.
2. **Kendala 2: HTTP Error 400 pada Wikimedia Thumbnail API**
   - *Error:* `HTTP Error 400: Use thumbnail sizes listed on https://w.wiki/GHai` saat meminta arbitrary thumbnail URL.
   - *Penyebab:* Kebijakan baru Wikimedia mewajibkan pemanggilan ukuran thumbnail baku atau unduhan file master langsung.
   - *Solusi:* Menggunakan algoritma hash MD5 MediaWiki (`commons/<hash[0]>/<hash[0:2]>/filename`) untuk mengunduh master image asli tanpa batasan resolusi.
3. **Kendala 3: Matplotlib Writable Directory Warning di Sandbox**
   - *Error:* `C:\Users\Ghaisani\.matplotlib is not a writable directory` saat render visualisasi.
   - *Solusi:* Menyetel backend `matplotlib.use("Agg")` untuk rendering headless non-interaktif sehingga penulisan cache dialihkan ke temp directory tanpa error.

---

## 6. Hasil Pengujian Unit (`tests/test_preprocessing.py`)
Hasil eksekusi test runner:
```text
Ran 9 tests in 5.380s
OK
- test_01_device_acceleration: Passed (cpu detected)
- test_02_proportional_resizing_rules: Passed (max 256 px, small image no enlargement)
- test_03_load_image_rgb_and_dimensions: Passed (all 6 images RGB mode & verified)
- test_04_tensor_conversion_and_range: Passed (1x3xHxW, float32, strictly [0, 1])
- test_05_different_sizes_content_and_style: Passed (content & style maintain aspect ratios)
- test_06_imagenet_normalization_and_prevention: Passed (DoubleNormalizationError raised correctly)
- test_07_denormalization_and_reconstruction: Passed (diff < 1e-5)
- test_08_tensor_to_image_and_save: Passed (roundtrip save & reload verified)
- test_09_corrupted_and_missing_file_handling: Passed (FileNotFoundError & CorruptedImageError handled)
```

---

## 7. Rencana Langkah Berikutnya (Next Action Items)
1. [ ] **Fase 3: Model Fitur VGG-19 (`src/model.py`)** — Memuat arsitektur VGG-19, memotong layer yang tidak dibutuhkan (hingga `conv5_1`), dan membekukan seluruh parameter bobot (`param.requires_grad = False`).
2. [ ] **Fase 4: Modul Loss Kustom (`src/loss.py`)** — Implementasi Gram Matrix ternormalisasi, Content Loss, Multi-layer Style Loss, dan Total Variation Regularization.
3. [ ] **Fase 5: Engine Optimasi NST (`src/transfer.py`)** — Optimasi L-BFGS/Adam untuk mentransfer gaya lukisan maestro ke foto lanskap.

