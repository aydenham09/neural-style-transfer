# Laporan Eksperimen: Evaluasi Matriks 3 Pemandangan × 3 Gaya (9 Pasangan NST)

Dokumen ini menyajikan analisis komprehensif dari evaluasi matriks lengkap 3 foto lanskap alam dikombinasikan dengan 3 lukisan maestro klasik (total 9 pasangan optimasi Neural Style Transfer) pada resolusi maksimum 256 piksel.

---

## 1. Metadata Eksperimen

- **ID Eksperimen Terkait:**
  - `EXP-001`: Mount Rainier × Vincent van Gogh (*Wheat Field with Cypresses*) — *Baseline digunakan kembali*
  - `EXP-004`: Mount Rainier × Katsushika Hokusai (*The Great Wave off Kanagawa*)
  - `EXP-005`: Mount Rainier × Wassily Kandinsky (*Composition 8*)
  - `EXP-006`: Crater Lake × Vincent van Gogh (*Wheat Field with Cypresses*)
  - `EXP-007`: Crater Lake × Katsushika Hokusai (*The Great Wave off Kanagawa*)
  - `EXP-008`: Crater Lake × Wassily Kandinsky (*Composition 8*)
  - `EXP-009`: Redwood Forest × Vincent van Gogh (*Wheat Field with Cypresses*)
  - `EXP-010`: Redwood Forest × Katsushika Hokusai (*The Great Wave off Kanagawa*)
  - `EXP-011`: Redwood Forest × Wassily Kandinsky (*Composition 8*)
- **Tanggal Eksekusi:** 2026-10-29
- **Perangkat Komputasi:** NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.1 / PyTorch 2.5.1+cu121)
- **Status Eksekusi:** **9/9 Pasangan Selesai Sukses (100% Konvergen, Zero Drift pada VGG-19)**

---

## 2. Konfigurasi Hiperparameter Tetap

Seluruh 9 percobaan menggunakan konfigurasi seragam sesuai rekomendasi sementara:
- **Bobot Konten ($\alpha$):** $1.0$
- **Bobot Gaya ($\beta$):** $100\,000.0$ ($10^5$)
- **Bobot Total Variation ($\gamma$):** $0.0$
- **Optimizer:** Adam (`lr = 0.02`, `steps = 300`)
- **Resolusi Maksimum:** $256$ piksel (aspek rasio proporsional dipertahankan)
- **Inisialisasi Target:** Salinan independen dari citra konten masing-masing (`target = content.clone().detach().requires_grad_(True)`)
- **Representasi Layer VGG-19:**
  - Konten: `conv4_2` (post-ReLU)
  - Gaya: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` (bobot seimbang $0.2$)
- **Kontrak Normalisasi:** Raw RGB float $[0, 1] \rightarrow$ normalisasi ImageNet internal tepat 1 kali di `VGG19FeatureExtractor.forward()`.
- **Pixel Clamping:** Clamping $[0.0, 1.0]$ in-place di bawah `torch.no_grad()` setelah setiap step optimizer.

---

## 3. Verifikasi Integritas Input (SHA-256 Hashes)

Untuk memastikan reproducibilitas mutlak, seluruh hash citra sumber dan tensor input diverifikasi:

### Citra Konten:
| Citra Konten | Resolusi NST | File SHA-256 | Tensor SHA-256 |
|---|---|---|---|
| **Mount Rainier** (`landscape_mountain.jpg`) | $256 \times 144$ | `cccecd28e78b7889a65d24dfed35cb3d280a6884ac6419165b80c92157aa5452` | `d6afc384aeb173c5f6f4eddddf692dfe3c85d8db51f75f8ffc515419d948776b` |
| **Crater Lake** (`landscape_lake.jpg`) | $256 \times 169$ | `22902eab83971c76d58d597560452c091fffb70741dbec6699214596aad9181e` | `d29188cae03949a43dcd81b15802ce2c62accc42b2ab33de02cfb885ae5482e2` |
| **Redwood Forest** (`landscape_forest.jpg`) | $256 \times 174$ | `71eec48714c8607ec2374b5479a6f7c37eb59290f6dea95ca76f8c116086e868` | `4aa98ec0b795657ae1fc721f62c0f7dda44715e47cb363b370ac0f76427a9a80` |

### Citra Gaya:
| Citra Gaya | Resolusi NST | File SHA-256 | Tensor SHA-256 |
|---|---|---|---|
| **Van Gogh (*Wheat Field*)** (`style_van_gogh.jpg`) | $256 \times 204$ | `b46d02a42fd3bb13b022cefc5beaa997b1d7a28cd7b53e4413d940aa0dc4014d` | `10fff03cf8dcf55c08fb6c5a0820904ab9d0afa24f3efe7402ccdb5269f91a94` |
| **Hokusai (*Great Wave*)** (`style_hokusai.jpg`) | $256 \times 172$ | `30c170260de393f51bd592adacc6f0714b60f157c09744e9c5b6d46e6a613718` | `bc204fc2ecdf86dafd2fd82cb9d015324022d539bd50f67e088532c896109837` |
| **Kandinsky (*Comp 8*)** (`style_kandinsky.jpg`) | $256 \times 186$ | `4d96c756f2db349ea1f778b7ddb2bd2a0fd2d27149d08adb1c1dd8bbaf1a0dc7` | `5e1d72e4e85bad611ade0b9936b5d208829ff84418ac9a03ff50b07170f3e75c` |

---

## 4. Tabel Hasil Kuantitatif Lengkap (9 Pasangan)

Metrik akhir dihitung **setelah update step ke-300 dan operasi clamping final**. Tabel di bawah menyajikan komparasi antara nilai loss pada **tensor float32** dan nilai loss saat citra dibaca kembali dari **file PNG hasil kuantisasi uint8 $[0, 255]$** di disk.

| Exp ID | Pasangan (Konten × Gaya) | Durasi | Loss Total Awal | $\mathcal{L}_c$ Akhir (Float) | $\mathcal{L}_s$ Mentah Akhir | Reduksi Gaya | Loss Total Akhir (Float) | Loss Total Akhir (PNG uint8) | Kuantisasi $\Delta \mathcal{L}_{\text{total}}$ |
|---|---|---|---|---|---|---|---|---|---|
| **EXP-001** | Gunung × Van Gogh *(Baseline)* | 3.71 s | 3.1027 | **0.2291** | $8.297 \times 10^{-6}$ | **-73.26%** | **1.0588** | 1.0588 | $0.00000$ |
| **EXP-004** | Gunung × Hokusai | 6.44 s | 11.9293 | **0.4662** | $8.224 \times 10^{-6}$ | **-93.11%** | **1.2886** | 1.2896 | $+0.00100$ |
| **EXP-005** | Gunung × Kandinsky | 5.88 s | 21.1660 | **1.6210** | $1.819 \times 10^{-5}$ | **-91.41%** | **3.4401** | 3.4421 | $+0.00197$ |
| **EXP-006** | Danau × Van Gogh | 5.78 s | 3.1913 | **0.2372** | $8.969 \times 10^{-6}$ | **-71.89%** | **1.1341** | 1.1350 | $+0.00088$ |
| **EXP-007** | Danau × Hokusai | 7.25 s | 14.1471 | **0.5562** | $7.335 \times 10^{-6}$ | **-94.82%** | **1.2897** | 1.2905 | $+0.00078$ |
| **EXP-008** | Danau × Kandinsky | 7.15 s | 22.5154 | **1.5713** | $1.692 \times 10^{-5}$ | **-92.49%** | **3.2628** | 3.2640 | $+0.00115$ |
| **EXP-009** | Hutan × Van Gogh | 5.53 s | 3.1293 | **0.2438** | $9.051 \times 10^{-6}$ | **-71.08%** | **1.1488** | 1.1497 | $+0.00082$ |
| **EXP-010** | Hutan × Hokusai | 6.70 s | 12.8595 | **0.4992** | $7.971 \times 10^{-6}$ | **-93.80%** | **1.2964** | 1.2976 | $+0.00123$ |
| **EXP-011** | Hutan × Kandinsky | 8.92 s | 19.9618 | **1.6609** | $1.582 \times 10^{-5}$ | **-92.07%** | **3.2433** | 3.2438 | $+0.00056$ |

---

## 5. Analisis Efek Kuantisasi (Tensor Float32 vs Gambar PNG uint8)

Sesuai metodologi pengujian yang ketat:
1. Saat citra target disimpan ke disk sebagai format PNG, nilai intensitas float kontinu $[0.0, 1.0]$ dikonversi menjadi integer diskrit 8-bit uint8 $[0, 255]$ melalui pembulatan ($\text{round}(x \times 255)$).
2. Ketika citra PNG tersebut dimuat kembali ke dalam ruang tensor float, terdapat galat diskretisasi kuantisasi numerik maksimum teoretis sebesar $\le \frac{0.5}{255} \approx 0.00196$ per piksel.
3. **Temuan Kuantitatif Aktual:**
   - Selisih total loss akibat kuantisasi ($\Delta \mathcal{L}_{\text{total}}$) di seluruh 9 pasangan berkisar antara **$+0.00000$ hingga $+0.00197$**.
   - Deviasi relatif rata-rata terhadap total loss adalah **kurang dari $0.09\%$**.
   - Kuantisasi uint8 sedikit meningkatkan total loss karena membulatkan transisi gradien halus, namun magnitudonya sangat kecil sehingga integritas visual hasil optimasi di disk terbukti identik secara perseptual dengan kondisi tensor float di memori GPU.

---

## 6. Evaluasi Kualitatif Tripartit (Perilaku Gaya & Konten)

> **Peringatan Metodologis:** Konfigurasi $\beta = 10^5$ **tidak dapat diklaim sebagai konfigurasi optimal universal untuk semua pasangan gaya**. Hasil di bawah memperlihatkan interaksi visual spesifik antar-pasangan. Selain itu, **penyebab timbulnya noise tetap berstatus hipotesis ilmiah** hingga dibuktikan secara eksperimental melalui pengujian variasi bobot TV loss secara bertahap.

### 1. Keterjagaan Bentuk Pemandangan (Content Preservation)
- **Gaya Van Gogh (EXP-001, EXP-006, EXP-009):**
  - Keterjagaan bentuk sangat konsisten dan tinggi di semua pemandangan ($\mathcal{L}_c \in [0.229, 0.244]$).
  - Siluet gunung Mount Rainier, garis kawah Crater Lake, dan batang pohon Redwood tetap tegak dan dapat diidentifikasi secara jelas.
- **Gaya Hokusai (EXP-004, EXP-007, EXP-010):**
  - Keterjagaan bentuk moderat-tinggi ($\mathcal{L}_c \in [0.466, 0.556]$).
  - Batas topografi utama tetap kokoh, namun transisi gradien lereng dan awan digantikan oleh kontur garis tajam khas cetak balok kayu Jepang.
- **Gaya Kandinsky (EXP-005, EXP-008, EXP-011):**
  - Keterjagaan bentuk terdegradasi secara substansial ($\mathcal{L}_c \in [1.571, 1.661]$).
  - Kandinsky *Composition 8* memuat elemen geometris kaku sintetis (lingkaran konsentris, bujur sangkar, garis silang) yang secara fundamental bertentangan dengan struktur kurva alami lanskap. Akibatnya, lereng gunung dan permukaan air terfragmentasi menjadi bidang geometris terpotong-potong.

### 2. Pengaruh Warna dan Tekstur Lukisan (Color & Texture Influence)
- **Gaya Van Gogh:**
  - Palet warna biru toska, kuning gandum, dan hijau zaitun terserap harmonis.
  - Tekstur sapuan kuas meliuk (*impasto*) membentuk aliran dinamis pada langit dan permukaan air danau kawah.
- **Gaya Hokusai:**
  - Penurunan style loss sangat dominan ($93.1\% - 94.8\%$).
  - Warna biru nila (*Prussian blue*) dan krem/putih busa ombak terserap kuat ke dalam air Crater Lake dan puncak es Mount Rainier, menciptakan efek visual yang sangat estetik.
- **Gaya Kandinsky:**
  - Penurunan style loss mencapai $>91\%$.
  - Warna-warna primer cerah (merah kirmizi, kuning cerah, hitam kontras) disuntikkan secara tegas ke seluruh kanvas, menggantikan rona alami lanskap dengan komposisi abstrak Bauhaus modern.

### 3. Analisis Artefak Visual & Hipotesis Noise
- **Grain Halus di Langit (Van Gogh):**
  - Muncul bintik-bintik noise halus (*high-frequency grain*) pada area langit terbuka di EXP-001 dan EXP-006.
  - **Hipotesis Terbuka:** Fenomena ini dihipotesiskan timbul akibat tingginya gradien style loss pada frekuensi spasial tanpa kehadiran penyeimbang peredam variasi ($\text{TV weight} = 0$). Hipotesis ini memerlukan konfirmasi empiris pada eksperimen tuning TV loss mendatang.
- **Edge Banding (Hokusai):**
  - Gaya Hokusai tidak memunculkan bintik noise acak, melainkan artefak pita warna bertingkat (*color banding*) di sepanjang garis tepi tebing dan ombak akibat karakteristik bidang warna datar pada lukisan ukiyo-e.
- **Fragmentasi Struktural (Kandinsky):**
  - Pada bobot gaya $10^5$, gaya Kandinsky memicu artefak struktural berat berupa bercak warna kontras tinggi dan pecahan kurva geometris yang memecah kontinuitas permukaan alam.

---

## 7. Rekomendasi Pasangan Terbaik untuk Demo Interaktif

Berdasarkan evaluasi estetika, ketegasan karakteristik gaya, dan keterjagaan semantik lanskap, berikut adalah pasangan yang paling direkomendasikan untuk demo pengguna (Streamlit):

1. **Pilihan Utama (Top Showcase): Mount Rainier × Hokusai (*Great Wave*) (EXP-004)**
   - *Alasan:* Keselarasan tema budaya dan visual yang luar biasa. Gunung Mount Rainier bertransformasi dengan estetika yang sangat mirip Gunung Fuji pada lukisan cetak kayu Jepang klasik, dihiasi warna Prussian blue dan tekstur salju seperti buih ombak laut. Bentuk gunung tetap sangat jelas ($\mathcal{L}_c = 0.466$).
2. **Pilihan Kedua (Water Dynamics Showcase): Crater Lake × Van Gogh (*Wheat Field*) (EXP-006)**
   - *Alasan:* Permukaan air danau kawah yang dalam menyatu secara spektakuler dengan pusaran kuas impasto Van Gogh. Bentuk pulau vulkanik Wizard Island dan tebing kawah tetap presisi ($\mathcal{L}_c = 0.237$).
3. **Pilihan Ketiga (Classic Baseline): Mount Rainier × Van Gogh (EXP-001)**
   - *Alasan:* Pasangan representatif yang menunjukkan pergeseran palet warna toska-emas dan tekstur impasto pada lanskap pegunungan dengan fidelitas semantik tinggi.
4. **Pilihan Khusus (Abstract Art Experiment): Crater Lake × Kandinsky (EXP-008)**
   - *Alasan:* Sangat menarik untuk mendemonstrasikan batas ekstrem transfer gaya abstrak modern kepada pengguna, sekaligus mengilustrasikan trade-off hilangnya bentuk semantik demi ekspresi geometris.

---

## 8. Lokasi File & Artefak Terintegrasi

- **Matriks Komparasi 4×4 Terpadu:**
  - Visual Grid: [`outputs/experiments/evaluation_3x3_matrix/matrix_3x3_grid.png`](file:///d:/neural-style-transfer/outputs/experiments/evaluation_3x3_matrix/matrix_3x3_grid.png)
  - Master JSON Summary: [`outputs/experiments/evaluation_3x3_matrix/matrix_3x3_summary.json`](file:///d:/neural-style-transfer/outputs/experiments/evaluation_3x3_matrix/matrix_3x3_summary.json)
- **Skrip Eksekusi Otomatis:**
  - [`scripts/run_3x3_matrix_evaluation.py`](file:///d:/neural-style-transfer/scripts/run_3x3_matrix_evaluation.py)
- **Folder Hasil Individual Tiap Pasangan:**
  - EXP-001 (Gunung × Van Gogh): [`outputs/experiments/exp01_mountain_van_gogh_adam/`](file:///d:/neural-style-transfer/outputs/experiments/exp01_mountain_van_gogh_adam)
  - EXP-004 (Gunung × Hokusai): [`outputs/experiments/exp04_mountain_hokusai/`](file:///d:/neural-style-transfer/outputs/experiments/exp04_mountain_hokusai)
  - EXP-005 (Gunung × Kandinsky): [`outputs/experiments/exp05_mountain_kandinsky/`](file:///d:/neural-style-transfer/outputs/experiments/exp05_mountain_kandinsky)
  - EXP-006 (Danau × Van Gogh): [`outputs/experiments/exp06_lake_van_gogh/`](file:///d:/neural-style-transfer/outputs/experiments/exp06_lake_van_gogh)
  - EXP-007 (Danau × Hokusai): [`outputs/experiments/exp07_lake_hokusai/`](file:///d:/neural-style-transfer/outputs/experiments/exp07_lake_hokusai)
  - EXP-008 (Danau × Kandinsky): [`outputs/experiments/exp08_lake_kandinsky/`](file:///d:/neural-style-transfer/outputs/experiments/exp08_lake_kandinsky)
  - EXP-009 (Hutan × Van Gogh): [`outputs/experiments/exp09_forest_van_gogh/`](file:///d:/neural-style-transfer/outputs/experiments/exp09_forest_van_gogh)
  - EXP-010 (Hutan × Hokusai): [`outputs/experiments/exp10_forest_hokusai/`](file:///d:/neural-style-transfer/outputs/experiments/exp10_forest_hokusai)
  - EXP-011 (Hutan × Kandinsky): [`outputs/experiments/exp11_forest_kandinsky/`](file:///d:/neural-style-transfer/outputs/experiments/exp11_forest_kandinsky)

