# Ringkasan Serah Terima (Handover Summary) — Tahap Dataset & Preprocessing

Dokumen ini merangkum status proyek Neural Style Transfer (NST), perubahan file, panduan eksekusi di perangkat lokal, serta catatan teknis penting.

---

## 1. Daftar File yang Berubah & Ditambahkan

### File Baru:
- `src/utils.py`: Modul utilitas inti untuk:
  - Deteksi device akselerasi komputasi (`get_device()`)
  - Resize proporsional dengan batas maksimal 256 piksel tanpa memperbesar gambar kecil (`load_image()`, `calculate_proportional_size()`)
  - Konversi citra ke tensor PyTorch `1x3xHxW` bernilai `0.0 - 1.0` (`image_to_tensor()`)
  - Normalisasi ImageNet dengan proteksi normalisasi ganda (`normalize_imagenet()`, `is_normalized()`)
  - Denormalisasi balik dan rekonstruksi citra untuk preview/simpan (`denormalize_imagenet()`, `tensor_to_image()`, `save_image()`)
  - Pembuat visualisasi grid dataset (`create_dataset_preview()`)
  - Exception kustom (`CorruptedImageError`, `InvalidImageError`, `DoubleNormalizationError`)
- `scripts/download_dataset.py`: Skrip mandiri untuk mengunduh dataset konten lanskap dan lukisan maestro dari sumber resmi dan mengekspor metadata YAML.
- `data/metadata/dataset_metadata.yaml`: Rekapitulasi metadata lengkap untuk 3 foto konten dan 3 lukisan gaya (judul, pembuat, URL sumber & unduhan, lisensi, dimensi asli, format, dan ukuran).
- `tests/test_preprocessing.py`: Unit test suite (9 test cases) yang mencakup seluruh skenario validasi, ukuran tensor, rentang nilai, normalisasi/denormalisasi, proteksi file rusak, dan roundtrip penyimpanan.
- `docs/session_02_dataset_preprocessing.md`: Log sesi pengembangan dan interaksi Sesi 02 sesuai format template proyek.
- `docs/HANDOVER_SUMMARY.md`: Ringkasan serah terima ini.
- `outputs/dataset_preview.png`: Gambar grid komparasi visual dataset konten dan gaya (dihasilkan otomatis oleh skrip pengujian/preview).

### File Diubah:
- `src/__init__.py`: Mengekspor fungsi-fungsi utilitas utama dari `src.utils`.

### File Gambar (Diabaikan oleh Git via `.gitignore`):
- `data/content/landscape_mountain.jpg` (Gunung - Mount Rainier, CC0)
- `data/content/landscape_lake.jpg` (Danau - Crater Lake, Public Domain USGS)
- `data/content/landscape_forest.jpg` (Hutan - Redwood Forest, Public Domain NPS)
- `data/style/style_van_gogh.jpg` (Van Gogh - Wheat Field with Cypresses, The Met Open Access)
- `data/style/style_hokusai.jpg` (Hokusai - The Great Wave off Kanagawa, Public Domain)
- `data/style/style_kandinsky.jpg` (Kandinsky - Composition 8, Public Domain Guggenheim)

---

## 2. Cara Menjalankan

### Aktivasi Virtual Environment
```powershell
# Di Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### Menjalankan Unit Test Preprocessing
```powershell
python -m unittest tests/test_preprocessing.py
```
Seluruh 9 pengujian akan dijalankan, memvalidasi pemuatan gambar, ukuran tensor `1x3xHxW`, rentang nilai `[0, 1]`, normalisasi ImageNet, proteksi `DoubleNormalizationError`, serta simulasi file rusak/kosong/truncated.

### Menghasilkan Preview Dataset
```powershell
python -c "from src.utils import create_dataset_preview; create_dataset_preview(['data/content/landscape_mountain.jpg', 'data/content/landscape_lake.jpg', 'data/content/landscape_forest.jpg'], ['data/style/style_van_gogh.jpg', 'data/style/style_hokusai.jpg', 'data/style/style_kandinsky.jpg'], output_path='outputs/dataset_preview.png')"
```
Hasil visualisasi tersimpan di `outputs/dataset_preview.png`.

### Mengunduh Ulang Dataset (Jika Diperlukan)
```powershell
python scripts/download_dataset.py
```

---

## 3. Lingkungan Komputasi & Akselerasi Perangkat

- **OS:** Windows x64
- **CPU:** Intel Core Ultra (22 logical threads)
- **Akselerasi Aktif:** `cpu` (`torch.device("cpu")`). Tidak ada GPU diskrit NVIDIA/CUDA atau Apple MPS yang terpasang pada environment ini. Seluruh pipeline telah disetel otomatis mendeteksi dan memanfaatkan CPU secara optimal.

---

## 4. Kendala & Catatan Penanganan (Troubleshooting)

1. **Akses Jaringan Sandbox:**  
   Proses instalasi dependensi pip dan pengunduhan gambar memerlukan izin jaringan (bypass sandbox) yang dijalankan secara terisolasi pada langkah unduhan saja, sementara proses pengujian, preprocessing, dan validasi dijalankan secara aman di dalam sandbox lokal.
2. **Rate Limit / Kebijakan API Museum & Arsip:**  
   The Met memberlakukan batasan ketat pada scraping HTML dan menonaktifkan endpoint search lama (HTTP 410). Seluruh aset diunduh menggunakan direct object image URL yang valid (`DP-42549-001.jpg`) dan CDN Wikimedia Commons master hash MD5 untuk menjamin integritas file master tanpa kompresi pihak ketiga.
3. **Penyimpanan Matplotlib Cache:**  
   Backend diatur secara terprogram ke `Agg` agar tidak memerlukan akses direktori konfigurasi GUI/interaktif saat dijalankan di lingkungan otomasi/sandbox.

