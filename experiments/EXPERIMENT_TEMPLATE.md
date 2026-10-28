# Template Catatan Eksperimen Neural Style Transfer (NST)

Gunakan template ini setiap kali menjalankan variasi eksperimen baru untuk merekam parameter, dinamika loss, dan evaluasi visual hasil transfer gaya lukisan.

---

## Metadata Eksperimen

- **ID Eksperimen:** `EXP-001`
- **Tanggal & Waktu:** `YYYY-MM-DD HH:MM`
- **Tujuan / Hipotesis:** (Contoh: *Menguji transfer gaya Van Gogh dengan inisialisasi gambar konten vs white noise pada resolusi 512px*)
- **Penanggung Jawab / Vibe Coder:** `irham`
- **Perangkat Komputasi:** `Apple Silicon MPS (Metal Performance Shaders)` / `CUDA` / `CPU`

---

## 1. Input Data & Referensi Gaya

| Komponen | Nama File / Sumber | Resolusi Awal | Catatan Karakteristik Visual |
|---|---|---|---|
| **Content Image** | `data/content/landscape_01.jpg` | 1920x1080 | Pemandangan pegunungan & langit cerah |
| **Style Image** | `data/style/van_gogh_wheat_field.jpg` | 1200x950 | Tekstur kuas meliuk tebal (impasto), warna emas & biru langit |

Pilihan Karya Referensi:
- [ ] Vincent van Gogh – *Wheat Field with Cypresses*
- [ ] Katsushika Hokusai – *The Great Wave off Kanagawa*
- [ ] Wassily Kandinsky – *Composition 8*
- [ ] Referensi Lain: `...`

---

## 2. Konfigurasi Hiperparameter

```yaml
experiment_id: EXP-001
image:
  target_size: 512 # Panjang sisi terpanjang / resize
  init_method: "content" # Pilihan: "content", "noise", "style"

weights:
  content_weight: 1.0 # alpha
  style_weight: 1000000.0 # beta (1e6)
  tv_weight: 1e-4 # gamma (Total Variation loss untuk kehalusan)

layers:
  content_layers:
    - "conv_4_2"
  style_layers:
    conv_1_1: 0.2
    conv_2_1: 0.2
    conv_3_1: 0.2
    conv_4_1: 0.2
    conv_5_1: 0.2

optimizer:
  type: "LBFGS" # LBFGS atau Adam
  learning_rate: 1.0
  steps: 300
  history_size: 100
```

---

## 3. Hasil & Metrik Pelatihan

- **Durasi Eksekusi Total:** `... menit / detik`
- **Kecepatan:** `... iterasi/detik`
- **Memory Peak VRAM / Unified Memory:** `... MB`

| Step | Content Loss | Style Loss | TV Loss (opsional) | Total Loss |
|---|---|---|---|---|
| 50 | | | | |
| 100 | | | | |
| 200 | | | | |
| 300 (Final) | | | | |

---

## 4. Evaluasi Visual & Hasil Gambar

- **Path Output Gambar:** `outputs/EXP-001_result.png`
- **Path Grafik Loss:** `outputs/EXP-001_loss_curve.png`

### Kriteria Kualitatif:
1. **Preservasi Konten Semantik:**
   - *Apakah bentuk fisik pemandangan (gunung, pohon, cakrawala) masih teridentifikasi jelas?*
2. **Transfer Gaya & Tekstur Lukisan:**
   - *Apakah sapuan kuas, ornamen ombak, atau geometri abstrak terserap dengan baik?*
3. **Keseimbangan Warna:**
   - *Apakah palet warna lukisan berhasil merekonstruksi warna pemandangan secara natural?*
4. **Artefak & Noise:**
   - *Apakah terdapat artefak bintik kasar (high frequency noise) atau distorsi berlebih?*

---

## 5. Kesimpulan & Penyesuaian Berikutnya
- **Apa yang berhasil:**
- **Kekurangan yang ditemukan:**
- **Tindakan selanjutnya (Next Trial):** (Contoh: *Naikkan rasio style/content weight, uji inisialisasi noise, atau ubah layer konten ke conv3_2*)

