# Log Interaksi & Pengembangan NST — Sesi 04

Dokumen ini mencatat kelanjutan pekerjaan Neural Style Transfer (NST) pada tahap implementasi fungsi loss kustom, perhitungan matematis Gram Matrix skala-invarian, Content Loss, Multi-layer Style Loss, Total Variation (TV) Loss, serta modul composite loss manager.

---

## Ringkasan Sesi

- **ID Sesi / Tanggal:** `2026-10-28 - Sesi 04`
- **Fokus / Tujuan:** Membangun seluruh fungsi loss kustom di `src/loss.py` dari dasar (tanpa pustaka NST luar):
  1. Implementasi matematis Gram Matrix ($G = F F^T / (C \cdot H \cdot W)$) dengan sifat simetris dan invariansi skala terhadap resolusi berbeda.
  2. Implementasi `ContentLoss` berbasis Mean Squared Error berfaktor $0.5$ pada layer `conv4_2`.
  3. Implementasi `SingleLayerStyleLoss` dan `StyleLoss` multi-layer (pada 5 layer VGG-19) dengan bobot konfigurabel.
  4. Implementasi `TotalVariationLoss` (TV Loss) untuk meredam noise bintik dan menjaga kehalusan goresan kuas lukisan.
  5. Implementasi composite loss manager `NSTLoss` yang mengintegrasikan $\alpha$, $\beta$, dan $\gamma$.
  6. Penulisan test suite komprehensif (`tests/test_loss.py`) dan skrip verifikasi mandiri (`scripts/verify_loss.py`).
- **Status:** [x] Selesai

---

## 1. Landasan Matematis & Keputusan Desain

### A. Gram Matrix ($G_{ij} = \sum_k F_{ik} F_{jk}$)
- **Konsep:** Fitur konvolusional menangkap keberadaan motif visual. Perkalian titik antar saluran (*inner product between channels*) mengukur korelasi fitur tanpa memedulikan posisi spasial $(x, y)$. Ini menangkap esensi "gaya" (tekstur, pola sapuan kuas, harmoni warna) terpisah dari "isi" (posisi objek).
- **Invariansi Skala:** Dengan membagi hasil perkalian terhadap $(C \cdot H \cdot W)$, nilai elemen matriks Gram menjadi tidak terpengaruh oleh perbedaan dimensi spasial antara foto pemandangan ($144 \times 256$) dan lukisan referensi ($204 \times 256$).
- **Simetri:** $G = G^T$ diuji dan terbukti memiliki selisih numerik mutlak $0.0$.

### B. Content Loss ($\mathcal{L}_{\text{content}}$)
- Mengukur perbedaan semantik global antara gambar target yang dioptimasi dan foto pemandangan pada layer `conv4_2`:
  $$\mathcal{L}_{\text{content}} = \frac{1}{2} \text{MSE}(F_{\text{gen}}, P_{\text{content}})$$
- Target $P_{\text{content}}$ dibekukan (`detach().clone()`) sehingga tidak menerima gradien.

### C. Multi-Layer Style Loss ($\mathcal{L}_{\text{style}}$)
- Mengukur perbedaan tekstur lukisan di berbagai kedalaman representasi:
  $$\mathcal{L}_{\text{style}} = \sum_{l \in \{\text{conv1\_1, 2\_1, 3\_1, 4\_1, 5\_1}\}} w_l \cdot \text{MSE}(G_{\text{gen}}^l, A_{\text{style}}^l)$$
- Bobot default $w_l = 0.2$ untuk setiap layer. Dilengkapi breakdown per layer untuk monitoring tuning gaya Van Gogh, Hokusai, dan Kandinsky.

### D. Total Variation (TV) Regularization ($\mathcal{L}_{\text{tv}}$)
- Regularisasi kontinuitas spasial untuk menekan artefak bintik kasar (*high-frequency checkerboard artifacts*):
  $$\mathcal{L}_{\text{tv}} = \frac{1}{B} \sum_{b=1}^B \left( \frac{1}{C(H-1)W} \sum (x_{i+1, j} - x_{i, j})^2 + \frac{1}{CH(W-1)} \sum (x_{i, j+1} - x_{i, j})^2 \right)$$
- Nilai bernilai $0.0$ pada citra rata (*flat*) dan memberikan penalti tinggi pada bintik noise acak.

---

## 2. Hasil Pengujian Unit (`tests/test_loss.py`) & Skrip Verifikasi

### Eksekusi Unit Test Komprehensif
```text
Ran 24 tests in 2.672s
OK
```
Rincian 24 tests:
- 9 tests di `tests/test_preprocessing.py` (semua lulus)
- 8 tests di `tests/test_model.py` (semua lulus)
- 7 tests di `tests/test_loss.py` (semua lulus)

### Hasil Skrip Verifikasi (`python scripts/verify_loss.py`)
- **Device:** Apple Silicon MPS (`mps`)
- **Gram Matrix Symmetry:** Max error $= 0.00000000e+00$ ($G == G^T$)
- **TV Loss Flat vs Noise:** Flat $= 0.000000$, Noise $= 0.331312$
- **End-to-End NST Loss pada Citra Nyata:**
  - Content Image: `landscape_mountain.jpg` $[1, 3, 144, 256]$
  - Style Image: `style_van_gogh.jpg` $[1, 3, 204, 256]$
  - Content Loss: $0.000000$ (karena diinisialisasi dari foto konten)
  - Style Loss: $3.103038 \times 10^{-5}$ (Weighted $\beta = 10^6$: $31.0304$)
  - TV Loss: $0.009559$ (Weighted $\gamma = 10^{-3}$: $0.000010$)
  - Input Gradient Norm: $6.840586$ (mengalir sempurna ke piksel gambar)
  - VGG Parameters Frozen: `True` (`param.grad is None`)

