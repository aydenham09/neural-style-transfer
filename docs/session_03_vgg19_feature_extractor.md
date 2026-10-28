# Log Interaksi & Pengembangan NST — Sesi 03

Dokumen ini mencatat kelanjutan pekerjaan Neural Style Transfer (NST) pada tahap implementasi ekstraktor fitur konvolusional VGG-19 pretrained ImageNet, pembekuan bobot, pemotongan arsitektur, proteksi normalisasi ImageNet, unit test komprehensif, dan verifikasi aliran gradien input.

---

## Ringkasan Sesi

- **ID Sesi / Tanggal:** `2026-10-28 - Sesi 03`
- **Fokus / Tujuan:** Membangun ekstraktor fitur VGG-19 pretrained ImageNet (`src/model.py`), mengonfigurasi pembekuan bobot & mode `eval()`, memotong layer hingga `conv5_1`, menonaktifkan in-place ReLU (`inplace=False`), menjamin normalisasi ImageNet tepat satu kali (tanpa ketergantungan atribut yang bisa hilang), memverifikasi bahwa gradien mengalir ke tensor citra yang dioptimasi sementara parameter VGG tetap beku, serta memperbarui test suite dan ringkasan serah terima.
- **Status:** [x] Selesai

---

## 1. Prompt / Arahan Pengguna
> Lanjut project NST ini. Baca README, docs/HANDOVER_SUMMARY.md, log sesi terbaru, dan src/utils.py dulu supaya sesuai pekerjaan sebelumnya.  
> Buat ekstraktor fitur VGG-19 pretrained di src/model.py:
> - Gunakan VGG19_Weights.IMAGENET1K_V1 dan cache unduhan di checkpoints/.
> - Ambil bagian features saja, gunakan eval(), dan bekukan seluruh parameter.
> - Content dari conv4_2; style dari conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
> - Gunakan fitur setelah ReLU secara konsisten, dokumentasikan nama dan indeks layer. Atur ReLU inplace=False.
> - Normalisasi ImageNet tepat sekali. Periksa fungsi preprocessing yang sudah ada; jangan hanya mengandalkan atribut penanda tensor karena bisa hilang setelah operasi tensor.
> - Gunakan device yang tersedia di perangkat ini.  
> Uji memakai satu gambar pemandangan pada ukuran maksimum 256. Tampilkan bentuk fitur dan pastikan nilainya finite. Periksa gradien dapat mencapai tensor input, sementara bobot VGG tetap beku. Jangan membungkus forward gambar yang dioptimalkan dengan torch.no_grad().  
> Jelaskan bahwa bobot berasal dari pretrained ImageNet, sedangkan pipeline kita implementasikan sendiri. Catat hasil aktual, error jika ada, versi dependensi, dan cara menjalankan. Perbarui ringkasan serah terima.  
> Belum membuat Gram Matrix, loss, atau loop NST. Jangan mengubah identitas Git, commit, atau push otomatis. Setelah selesai, rangkum perubahan yang siap diperiksa.

---

## 2. Audit Lingkungan Perangkat & Dependensi Aktual
- **Sistem Operasi:** macOS Darwin Kernel (Apple Silicon `arm64`, macOS 27.x)
- **Akselerasi Komputasi:** **Apple Silicon MPS (`torch.backends.mps.is_available() == True`)**
- **Python Runtime:** Python 3.13.15
- **Versi Dependensi Utama:**
  - `torch`: 2.14.0
  - `torchvision`: 0.29.0
  - `Pillow`: 12.3.0
  - `numpy`: 2.5.3
  - `matplotlib`: 3.11.1
  - `PyYAML`: 6.0.3
- **Cache Unduhan Bobot:** `checkpoints/checkpoints/vgg19-dcbb9e9d.pth` (548 MB)

---

## 3. Keputusan Desain & Arsitektur (`src/model.py`)

### A. Pretrained ImageNet vs Implementasi Pipeline Mandiri
- **Bobot Pretrained:** Parameter konvolusi VGG-19 diambil dari `VGG19_Weights.IMAGENET1K_V1` yang telah dilatih pada dataset raksasa ImageNet (1.000 kelas). Bobot ini **dibekukan secara mutlak** (`requires_grad = False`) dan model disetel pada mode `eval()`.
- **Implementasi Mandiri:**
  1. Pemotongan layer selektif: Menghapus classification head (FC layers) dan lapisan konvolusi setelah `conv5_1` (mengurangi beban memori dan komputasi GPU).
  2. Penyetelan `inplace = False` pada seluruh modul `nn.ReLU` untuk mencegah modifikasi memori tensor saat proses autograd backpropagation.
  3. Pemeriksaan normalisasi ImageNet berbasis distribusi nilai numerik (anti double-normalization tanpa ketergantungan atribut volatile).
  4. Penyaluran gradien kembali ke tensor piksel gambar input (`input_tensor.grad`) tanpa memodifikasi bobot VGG.

### B. Pemetaan Layer & Indeks Konvolusi (Post-ReLU)
| Canonical Name | Alias Post-ReLU | Indeks di `vgg.features` | Modul PyTorch | Peran di NST | Output Dimensi (Input $144 \times 256$) |
|---|---|---|---|---|---|
| `conv1_1` | `relu1_1` | 1 | `ReLU(inplace=False)` | **Style Layer 1** | `[1, 64, 144, 256]` |
| `conv2_1` | `relu2_1` | 6 | `ReLU(inplace=False)` | **Style Layer 2** | `[1, 128, 72, 128]` |
| `conv3_1` | `relu3_1` | 11 | `ReLU(inplace=False)` | **Style Layer 3** | `[1, 256, 36, 64]` |
| `conv4_1` | `relu4_1` | 20 | `ReLU(inplace=False)` | **Style Layer 4** | `[1, 512, 18, 32]` |
| `conv4_2` | `relu4_2` | 22 | `ReLU(inplace=False)` | **CONTENT LAYER** | `[1, 512, 18, 32]` |
| `conv5_1` | `relu5_1` | 29 | `ReLU(inplace=False)` | **Style Layer 5** | `[1, 512, 9, 16]` |

*Catatan:* Layer di atas indeks 29 (`conv5_2` sampai `maxpool5`) dipangkas secara permanen karena tidak diperlukan dalam formulasi loss NST Gatys et al.

### C. Normalisasi ImageNet yang Kuat & Tahan Operasi Tensor
Penanda atribut Python seperti `tensor._is_imagenet_normalized = True` sering kali hilang secara diam-diam (*dropped*) oleh engine C++ PyTorch saat tensor mengalami operasi seperti `.clone()`, `.detach()`, `.requires_grad_()`, atau aritmatika tensor.  
Untuk mengatasi hal ini, fungsi `is_tensor_imagenet_normalized()` mengombinasikan:
1. Pengecekan atribut eksplisit (jika ada).
2. Fallback inspeksi rentang numerik:
   - Citra RGB mentah berada dalam rentang $[0.0, 1.0]$.
   - Citra yang dinormalisasi ImageNet mengalami pergeseran rata-rata $(\mu = [0.485, 0.456, 0.406])$ dan pembagian simpangan baku $(\sigma = [0.229, 0.224, 0.225])$, sehingga nilai piksel bernilai negatif (minimum mencapai $\approx -2.11$) dan melampaui $1.0$ (maksimum mencapai $\approx +2.64$).
   - Jika $\min < -0.05$ atau $\max > 1.5$, tensor dipastikan telah berada di ruang normalisasi ImageNet, sehingga terhindar dari normalisasi berulang.

---

## 4. Hasil Pengujian Aktual & Verifikasi

### Eksekusi Unit Test Suite (`python -m unittest discover tests`)
```text
[Model Test Setup] Running on compute device: mps
Verified weight cache in: checkpoints/checkpoints/vgg19-dcbb9e9d.pth
Verified 26 parameters are strictly frozen.
Verified 13 ReLUs all have inplace=False.
Attribute survived arithmetic? False
Landscape image input shape: [1, 3, 144, 256]
  Feature 'conv1_1': shape=[1, 64, 144, 256], finite=True
  Feature 'conv2_1': shape=[1, 128, 72, 128], finite=True
  Feature 'conv3_1': shape=[1, 256, 36, 64], finite=True
  Feature 'conv4_1': shape=[1, 512, 18, 32], finite=True
  Feature 'conv4_2': shape=[1, 512, 18, 32], finite=True
  Feature 'conv5_1': shape=[1, 512, 9, 16], finite=True
Input image gradient norm: 6.880593 (Gradient successfully reached input image)
Verified: all VGG weights have grad == None (completely frozen during backprop).
Verified CPU fallback execution and gradient propagation.

[Test Setup] Running tests on compute device: mps
Verified active device: mps
Ran 17 tests in 1.640s
OK
```

### Eksekusi Skrip Verifikasi Mandiri (`python scripts/verify_extractor.py`)
- Input Foto Lanskap: `data/content/landscape_mountain.jpg` $\rightarrow$ Shape: `[1, 3, 144, 256]` (maksimum 256 px).
- Seluruh nilai fitur bernilai *finite* (`torch.isfinite() == True`, tidak ada NaN/Inf).
- Autograd backward pass:
  - `input_tensor.grad` ada dan finite dengan grad norm $6.889860$.
  - Seluruh bobot VGG tetap `param.grad is None`.
- Proteksi double-normalization terbukti aktif meski atribut Python di-strip oleh `.clone() + 0.0`.

---

## 5. Kendala Aktual & Solusi (Troubleshooting)
1. **Kendala 1: Perbedaan Objek Device MPS pada Assertion Test**
   - *Error:* `AssertionError: device(type='mps', index=0) != device(type='mps')`.
   - *Penyebab:* PyTorch pada macOS Silicon terkadang melampirkan indeks device eksplisit (`mps:0`) pada tensor hasil komputasi sub-layer, sedangkan instansiasi `torch.device("mps")` tidak memiliki indeks spesifik.
   - *Solusi:* Membandingkan `tensor.device.type == expected_device.type` pada assertion test.
2. **Kendala 2: Atribut `self.training` pada Module Pembungkus**
   - *Error:* `self.features.eval()` hanya menyetel submodule `features`, sedangkan pembungkus kelas `VGG19FeatureExtractor` default-nya `training=True`.
   - *Solusi:* Memanggil `self.eval()` pada konstruktor kelas `VGG19FeatureExtractor` agar seluruh level hierarki modul berada dalam mode evaluasi.
3. **Kendala 3: HTTP Error 429 pada Unduhan Dataset Wikimedia**
   - *Error:* `HTTP Error 429: Too Many Requests` saat mengunduh gambar danau dan hutan.
   - *Penyebab:* Header User-Agent generic dibatasi oleh rate limiter Wikimedia.
   - *Solusi:* Memperbarui header User-Agent menjadi `NeuralStyleTransferBot/1.0` dengan jeda waktu inter-request $1.5$ detik di `scripts/download_dataset.py`. Seluruh 6 gambar dataset resmi kini tersimpan lengkap.

