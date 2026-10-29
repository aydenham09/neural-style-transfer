# Laporan Eksperimen: Perbandingan Bobot Style (Style Weight Comparison: 1e4 vs 1e5 vs 1e6)

Dokumen ini menyajikan analisis komprehensif dari eksperimen perbandingan bobot gaya (*style weight* $\beta \in \{10^4, 10^5, 10^6\}$) pada pasangan Mount Rainier (`landscape_mountain.jpg`) dan Vincent van Gogh (*Wheat Field with Cypresses*, `style_van_gogh.jpg`).

---

## 1. Metadata & Tujuan Eksperimen

- **ID Eksperimen Terkait:**
  - `EXP-002`: $\beta = 10^4$ (Rasio $\beta/\alpha = 10^4$)
  - `EXP-001`: $\beta = 10^5$ (Baseline yang digunakan kembali secara identik)
  - `EXP-003`: $\beta = 10^6$ (Rasio $\beta/\alpha = 10^6$)
- **Tanggal Eksekusi:** 2026-10-29
- **Tujuan Eksperimen:**
  1. Mengamati pengaruh perubahan bobot gaya secara logaritmik (1 dekade ke bawah dan 1 dekade ke atas dari baseline) terhadap dinamika optimasi dan visual hasil transfer.
  2. Mengevaluasi trade-off antara keterjagaan bentuk gunung (content fidelity), penyerapan tekstur impasto lukisan Van Gogh, serta timbulnya distorsi/artefak.
  3. Memvalidasi hipotesis asal-muasal noise: menguji apakah grain noise semata-mata diakibatkan oleh $\text{TV weight} = 0$ atau berkorelasi langsung dengan magnitudo gradien style loss.
- **Perangkat Komputasi:** NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.1 / PyTorch 2.5.1)

---

## 2. Kontrol Eksperimen & Integritas Input

Untuk menjamin komparasi yang valid (*fair comparison*), seluruh parameter non-gaya dikunci secara ketat dan identitas tensor diverifikasi menggunakan kriptografi SHA-256:

### Hiperparameter Tetap:
- **Optimizer:** Adam (`lr = 0.02`, `steps = 300`)
- **Inisialisasi Target:** Salinan independen dari citra konten (`target = content.clone().detach().requires_grad_(True)`)
- **Bobot Konten ($\alpha$):** $1.0$
- **Bobot Total Variation ($\gamma$):** $0.0$ (dinonaktifkan pada seluruh percobaan)
- **Ukuran Maksimum Citra:** $256$ piksel (resolusi konten: $256 \times 144$, resolusi gaya: $256 \times 204$)
- **Layer Fitur VGG-19:**
  - Konten: `conv4_2` (post-ReLU)
  - Gaya: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` (masing-masing berbobot $0.2$)
- **Kontrak Normalisasi:** Input RGB float $[0, 1]$, normalisasi ImageNet internal satu kali di dalam `forward()`.
- **Clamping:** $[0.0, 1.0]$ in-place di bawah `torch.no_grad()` setelah update step optimizer.

### Verifikasi Hash Input & Bobot Model:
| Komponen | SHA-256 Checksum | Status Verifikasi |
|---|---|---|
| **Content Image File** | `cccecd28e78b7889a65d24dfed35cb3d280a6884ac6419165b80c92157aa5452` | Identik 100% |
| **Style Image File** | `b46d02a42fd3bb13b022cefc5beaa997b1d7a28cd7b53e4413d940aa0dc4014d` | Identik 100% |
| **Content Tensor (`1x3x144x256`)** | `d6afc384aeb173c5f6f4eddddf692dfe3c85d8db51f75f8ffc515419d948776b` | Identik 100% |
| **Style Tensor (`1x3x204x256`)** | `10fff03cf8dcf55c08fb6c5a0820904ab9d0afa24f3efe7402ccdb5269f91a94` | Identik 100% |
| **VGG-19 Weights Pre/Post** | `4ab1c9682b08a6c52c0c043020a710debec2c6257641b3db95081c365897341e` | Zero Drift (100% Terkunci) |

---

## 3. Hasil Kuantitatif & Dinamika Loss (Post-Update Clamping)

> **Catatan Metrik:** Seluruh metrik akhir dihitung langsung dari tensor citra setelah update step ke-300 dan operasi clamping final, sehingga nilai numerik konsisten 100% dengan citra PNG yang tersimpan di disk.

| Metrik Evaluasi | EXP-002 ($\beta = 10^4$) | EXP-001 Baseline ($\beta = 10^5$) | EXP-003 ($\beta = 10^6$) |
|---|---|---|---|
| **Style Weight ($\beta$)** | $10\,000.0$ ($10^4$) | $100\,000.0$ ($10^5$) | $1\,000\,000.0$ ($10^6$) |
| **Rasio $\beta / \alpha$** | $10^4$ | $10^5$ | $10^6$ |
| **Durasi Komputasi (CUDA)** | **3.938 s** | **3.712 s** | **6.545 s** |
| **Rata-rata Waktu per Step** | 13.13 ms | 12.37 ms | 21.82 ms |
| **Total Loss Awal** | 0.3103 | 3.1027 | 31.0270 |
| **Total Loss Akhir (Post-Update)** | **0.1706** | **1.0588** | **3.7537** |
| **Penurunan Total Loss (%)** | **-45.03%** | **-65.87%** | **-87.90%** |
| **Content Loss Awal** | 0.0000 | 0.0000 | 0.0000 |
| **Content Loss Akhir (Post-Update)** | **0.0281** | **0.2291** | **2.0386** |
| **Style Loss Mentah Awal** | $3.1027 \times 10^{-5}$ | $3.1027 \times 10^{-5}$ | $3.1027 \times 10^{-5}$ |
| **Style Loss Mentah Akhir** | **$1.4245 \times 10^{-5}$** | **$8.2971 \times 10^{-6}$** | **$1.7151 \times 10^{-6}$** |
| **Penurunan Style Loss (%)** | **-54.09%** | **-73.26%** | **-94.47%** |

---

## 4. Evaluasi Kualitatif Tripartit

Evaluasi visual dilakukan melalui observasi terhadap panel 5 grid komparatif:

```
[Content: Mt. Rainier] -> [Style: Van Gogh] -> [Hasil 1e4] -> [Hasil 1e5] -> [Hasil 1e6]
```

### 1. Keterjagaan Bentuk Gunung (Content Preservation)
- **$\beta = 10^4$:** Keterjagaan bentuk mendekati sempurna ($\mathcal{L}_c = 0.0281$). Garis puncak salju Mount Rainier, tekstur batuan lereng, dan siluet pohon pinus di garis horizon tidak mengalami distorsi struktural sedikit pun.
- **$\beta = 10^5$:** Keterjagaan bentuk sangat baik ($\mathcal{L}_c = 0.2291$). Bentuk topografi gunung tetap kokoh dan mudah dikenali, batas semantik antara langit dan daratan tetap jelas, sembari membuka ruang bagi adaptasi warna dan tekstur lukisan.
- **$\beta = 10^6$:** Keterjagaan bentuk mulai terdegradasi parah ($\mathcal{L}_c = 2.0386$, melonjak **8.9 kali lipat** dibanding baseline). Puncak gunung bersalju dan awan mulai melebur (*bleeding*) ke dalam sapuan abstrak meliuk. Garis pemisah antara pohon dan punggung bukit menjadi kabur.

### 2. Kekuatan Tekstur Lukisan (Style Texture Strength)
- **$\beta = 10^4$:** Transfer gaya sangat minim/lemah (*under-styled*). Penurunan style loss hanya mencapai 54.09%. Kanvas hanya mengalami sedikit pergeseran rona warna (*color tint*) kekuningan/kehijauan pucat pada langit. Sapuan kuas tebal khas Van Gogh (*impasto*) hampir tidak tampak.
- **$\beta = 10^5$:** Penyerapan gaya berada pada tingkat proporsional (*balanced*). Penurunan style loss mencapai 73.26%. Palet warna Van Gogh (biru toska, gandum emas, hijau zaitun) terserap harmonis. Pola guratan kuas mulai terbentuk di langit dan lereng salju tanpa merusak objek semantik.
- **$\beta = 10^6$:** Penyerapan gaya sangat agresif dan dramatis (*over-styled*). Penurunan style loss mencapai 94.47%. Sapuan kuas melingkar dan tekstur impasto tebal menyelimuti hampir seluruh kanvas, menyerupai karakter lukisan minyak Van Gogh yang sangat ekspresif.

### 3. Analisis Artefak & Wawasan Noise (TV Loss Perspective)
- **Analisis Kritis Noise:**
  - Pada pengujian baseline ($\beta = 10^5$), teramati adanya *high-frequency grain* pada area langit.
  - Namun, pada pengujian $\beta = 10^4$ dengan kondisi yang **sama persis tanpa regularisasi TV ($\gamma = 0$)**, bintik noise frekuensi tinggi tersebut **hampir tidak muncul sama sekali**.
  - Sebaliknya, pada $\beta = 10^6$, selain grain halus, timbul artefak distorsi lokal (*high-frequency structural artifacts*) berupa bercak kontras tinggi yang terfragmentasi.
- **Kesimpulan Kritis:**
  - Bintik noise pada kanvas **bukan semata-mata disebabkan oleh $\text{TV weight} = 0$**, melainkan akibat **besarnya magnitudo gradien style loss yang menekan representasi spasial tanpa penyeimbang peredam variasi**.
  - Ketiadaan TV loss ($\gamma = 0$) memperbolehkan tetangga piksel bergerak bebas, tetapi gaya pemicunya adalah tekanan gradien gaya yang tinggi. Semakin besar $\beta$, semakin agresif optimasi memaksakan kecocokan matriks Gram hingga menciptakan fluktuasi piksel frekuensi tinggi.

---

## 5. Rekomendasi Konfigurasi Sementara

1. **Preset Rekomendasi:** **$\beta = 10^5$ (Rasio $\beta/\alpha = 10^5$)**
   - Merupakan titik keseimbangan terbaik (*sweet spot*) antara integritas semantik foto lanskap dan ekspresi artistik lukisan maestro.
   - $\beta = 10^4$ tidak disarankan karena menghasilkan gambar yang kurang berkarakter seni (*under-styled*).
   - $\beta = 10^6$ dapat dijadikan preset alternatif (*high-stylization preset*), namun bukan default untuk lanskap karena degradasi topografi gunung cukup signifikan.
2. **Langkah Berikutnya (Next Iteration):**
   - Melanjutkan ke penataan bobot TV loss $\gamma$ pada $\beta = 10^5$ untuk menghaluskan grain noise di langit tanpa merusak tekstur impasto.

---

## 6. Lokasi File & Artefak Terintegrasi

Semua artefak tersimpan terpisah tanpa saling menimpa:

- **Eksperimen EXP-002 ($\beta = 10^4$):**
  - Folder: `outputs/experiments/exp02_mountain_van_gogh_style1e4/`
  - Hasil Akhir: `final_stylized.png`
  - Grafik Loss: `loss_history.png`
  - Konfigurasi & Metrik: `run_config_and_metrics.json`
- **Eksperimen EXP-001 Baseline ($\beta = 10^5$):**
  - Folder: `outputs/experiments/exp01_mountain_van_gogh_adam/`
- **Eksperimen EXP-003 ($\beta = 10^6$):**
  - Folder: `outputs/experiments/exp03_mountain_van_gogh_style1e6/`
  - Hasil Akhir: `final_stylized.png`
  - Grafik Loss: `loss_history.png`
  - Konfigurasi & Metrik: `run_config_and_metrics.json`
- **Artefak Komparasi Terpadu:**
  - Folder: `outputs/experiments/style_weight_comparison/`
  - Grid 5 Panel: `grid_content_style_three_results.png`
  - Grafik Loss Komparatif 3 Panel: `comparative_loss_plots.png`
  - Ringkasan JSON: `comparison_summary.json`

