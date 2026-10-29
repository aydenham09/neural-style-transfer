# Laporan Eksperimen: EXP-001 (Mountain x Van Gogh — Baseline Adam)

Dokumen ini mencatat konfigurasi, metrik kuantitatif, analisis kurva loss, serta evaluasi kualitatif dari eksperimen baseline Neural Style Transfer (NST) pertama.

---

## Metadata Eksperimen

- **ID Eksperimen:** `EXP-001`
- **Tanggal & Waktu:** 2026-10-28 22:02 WIB
- **Tujuan / Hipotesis:** Menguji pipeline optimasi piksel baseline menggunakan optimizer Adam (lr=0.02, 300 steps) dengan inisialisasi target dari salinan citra konten (`landscape_mountain.jpg`) dan referensi gaya lukisan Vincent van Gogh (*Wheat Field with Cypresses*). Memverifikasi pembekuan bobot VGG-19, penegakan kontrak normalisasi ImageNet di dalam ekstraktor, dan clamping piksel $[0, 1]$.
- **Perangkat Komputasi:** NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.1 / PyTorch 2.5.1)
- **Status Eksperimen:** **Selesai Sukses (100% Convergence)**

---

## 1. Input Data & Referensi Gaya

| Komponen | File / Sumber | Resolusi Awal | Resolusi Input NST (Max 256px) | Karakteristik Visual |
|---|---|---|---|---|
| **Content Image** | `data/content/landscape_mountain.jpg` (Mount Rainier) | 3520 x 1980 | **256 x 144** | Puncak gunung bersalju, lereng berhutan hijau gelap, langit biru bersih |
| **Style Image** | `data/style/style_van_gogh.jpg` (*Wheat Field with Cypresses*) | 4000 x 3184 | **256 x 204** | Sapuan kuas tebal meliuk (*impasto*), langit berawan turbulen, pohon cypress hijau zaitun, gandum emas |

---

## 2. Konfigurasi Hiperparameter

```yaml
experiment_id: EXP-001
image:
  max_size: 256
  init_method: "content_copy"
  content_shape: [1, 3, 144, 256]
  style_shape: [1, 3, 204, 256]

weights:
  content_weight (alpha): 1.0
  style_weight (beta): 100000.0 (1e5)
  tv_weight (gamma): 0.0 # Tanpa penalti TV pada baseline ini
  style_to_content_ratio: 100000.0

layers:
  content_layer: "conv4_2" (post-ReLU, index 22)
  style_layers:
    conv1_1: 0.2 (index 1)
    conv2_1: 0.2 (index 6)
    conv3_1: 0.2 (index 11)
    conv4_1: 0.2 (index 20)
    conv5_1: 0.2 (index 29)

optimizer:
  type: "Adam"
  learning_rate: 0.02
  steps: 300
  pixel_clamping: [0.0, 1.0] (in-place under torch.no_grad())
```

---

## 3. Profil Eksekusi & Metrik Kuantitatif

- **Durasi Eksekusi Total:** **3.712 detik** (300 iterasi)
- **Rata-rata Waktu per Step:** **12.37 ms/step** (stabil ~11.6 ms pada steady-state)
- **Sinkronisasi Perangkat:** `torch.cuda.synchronize()` aktif sebelum dan sesudah pengukuran waktu
- **Verifikasi Bobot VGG-19:** 
  - SHA-256 Sebelum: `4ab1c9682b08a6c52c0c043020a710debec2c6257641b3db95081c365897341e`
  - SHA-256 Sesudah: `4ab1c9682b08a6c52c0c043020a710debec2c6257641b3db95081c365897341e`
  - **Hasil: Zero Drift (Bobot VGG-19 100% Tidak Berubah)**

### Tabel Dinamika Loss:

| Step | Content Loss ($\mathcal{L}_c$) | Weighted Content | Style Loss ($\mathcal{L}_s$) | Weighted Style | Total Loss ($\mathcal{L}_{\text{total}}$) | Grad Norm | Waktu Step |
|---|---|---|---|---|---|---|---|
| **1** | 0.000000 | 0.0000 | 3.1027e-05 | 3.1027 | 3.1027 | 0.6842 | 141.0 ms |
| **50** | 0.244488 | 0.2445 | 1.0416e-05 | 1.0416 | 1.2861 | 0.1568 | 12.1 ms |
| **100** | 0.222994 | 0.2230 | 9.0881e-06 | 0.9088 | 1.1318 | 0.2044 | 11.9 ms |
| **150** | 0.220191 | 0.2202 | 8.7404e-06 | 0.8740 | 1.0942 | 0.2049 | 12.1 ms |
| **200** | 0.220282 | 0.2203 | 8.5474e-06 | 0.8547 | 1.0750 | 0.2115 | 11.8 ms |
| **250** | 0.216248 | 0.2162 | 8.3465e-06 | 0.8347 | 1.0509 | 0.2076 | 12.3 ms |
| **300 (Final)** | 0.226895 | 0.2269 | 8.3606e-06 | 0.8361 | **1.0630** | 0.2712 | 11.6 ms |

### Ringkasan Penurunan Loss:
- **Penurunan Total Loss:** **65.74%** (dari $3.1027 \rightarrow 1.0630$)
- **Penurunan Style Loss:** **73.05%** (dari $3.1027 \times 10^{-5} \rightarrow 8.3606 \times 10^{-6}$)
- **Perilaku Content Loss:** Berawal dari $0.0000$ (karena inisialisasi identik dengan konten) lalu naik ke $\approx 0.22$ pada iterasi awal saat piksel bergulir menyesuaikan tekstur gaya, dan bertahan stabil di rentang $0.21 - 0.23$ hingga akhir optimasi.

---

## 4. Evaluasi Kualitatif & Analisis Visual

> **Peringatan Metodologis:** Evaluasi kualitas transfer citra seni **tidak boleh disimpulkan semata-mata dari kurva penurunan loss numerik**. Nilai loss yang rendah dapat terjadi bersamaan dengan distorsi bentuk semantik atau dominasi artefak frekuensi tinggi jika regularisasi tidak memadai.

### 1. Keterjagaan Bentuk Semantik (Content Preservation):
- **Skor Kualitatif:** **Sangat Baik (High Fidelity)**
- **Observasi:**
  - Siluet gunung Mount Rainier, garis horizon punggung bukit, serta formasi hutan di bagian tengah tetap terdefinisi dengan sangat jelas.
  - Lapisan semantik `conv4_2` berhasil menahan distorsi topologi besar sehingga objek utama tidak meleleh atau hancur menjadi corak acak.
  - Inisialisasi dari citra konten (`target_image = content.clone()`) memberikan prioritas spasial yang kuat dibandingkan inisialisasi white noise.

### 2. Pengaruh Gaya Lukisan (Style Influence & Texture Transfer):
- **Skor Kualitatif:** **Moderat / Subtil**
- **Observasi:**
  - **Palet Warna:** Warna langit biru foto lanskap bertransformasi mengadopsi rona biru toska (*cyan-blue*) dan sapuan kuning pucat khas langit berawan Van Gogh. Area vegetasi hijau menyerap nuansa hijau zaitun (*olive green*) dan ochre.
  - **Tekstur Impasto:** Pada rasio $\beta/\alpha = 10^5$, sapuan kuas tebal terlihat mulai muncul pada bidang langit dan lereng salju, namun karakternya belum se-ekstrem lukisan aslinya.
  - **Dinamika Konvergensi:** Setelah step 150, penurunan style loss melambat secara signifikan (mendatar dari $8.74 \times 10^{-6}$ ke $8.36 \times 10^{-6}$), menandakan learning rate Adam $0.02$ mulai mencapai *plateau* lokal pada ruang piksel.

### 3. Keberadaan Artefak Visual:
- **Skor Kualitatif:** **Terdapat Grain / Bintik Halus (Diharapkan karena $\gamma = 0$)**
- **Observasi:**
  - Karena konfigurasi awal menetapkan $\text{TV weight} = 0$, tidak ada penalti spasial terhadap disparitas piksel tetangga ($L_{\text{tv}}$).
  - Akibatnya, pada area gradien halus seperti langit, muncul bintik-bintik noise halus (*high-frequency grain*).
  - Penambahan regularisasi Total Variation ($\gamma = 10^{-4}$ atau $10^{-3}$) pada iterasi pengujian berikutnya diprediksi akan menghaluskan noise ini menjadi sapuan kuas yang lebih koheren.

---

## 5. Lokasi File Artefak

Seluruh file disimpan di folder terisolasi:
`outputs/experiments/exp01_mountain_van_gogh_adam/`

- Citra Awal (Step 0): `step_0000_initial.png`
- Progres Berkala: `progress/step_0050.png`, `step_0100.png`, `step_0150.png`, `step_0200.png`, `step_0250.png`, `step_0300.png`
- Hasil Akhir: `final_stylized.png`
- Grid Komparasi (Content - Style - Result): `comparison_content_style_result.png`
- Grafik Riwayat Loss (Mentah & Berbobot): `loss_history.png`
- Konfigurasi & Metrik Lengkap (JSON): `run_config_and_metrics.json`

