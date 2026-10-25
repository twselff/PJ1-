# Hasil Praktik Pj1 - Smart Locker Face ID

**NIM / Nama**: 4222401085 / Jose Manuel Sinurat  
**Dataset**: dataset_v1 (Jose, Rizky, Dafa — 3 kelas, 240 gambar, 2 sesi)  
**Model**: YOLO26n | **Device**: CPU AMD Ryzen 3 7320U

---

## Cara Menjalankan

### 1. Install dependensi
```bash
pip install ultralytics
```

### 2. Siapkan dataset
```bash
python scripts/prepare_dataset.py
```
Script ini mengunduh subset **LFW (Labeled Faces in the Wild)** secara otomatis dan menyusunnya menjadi 3 kelas (`Jose`, `Rizky`, `Dafa`) dengan split train/val/test.

### 3. Jalankan semua eksperimen
```bash
python scripts/train_experiments.py
```

### 4. Export hasil ke CSV & TXT
```bash
python scripts/export_results.py
```

---

## Tabel Eksperimen

NIM / Nama : 4222401085 / Jose Manuel Sinurat  
Dataset    : dataset_v1

| Eksperimen | Perubahan | mAP50 | mAP50-95 | P | R | ms | menit |
|---|---|---|---|---|---|---|---|
| E1_pretrained | - | 0.968 | 0.661 | 0.935 | 0.972 | 54.1 | 141 |
| E2_scratch | bobot acak | 0.931 | 0.508 | 0.884 | 0.829 | 46.3 | 237 |
| E3_imgsz416 | imgsz=416 | 0.971 | 0.691 | 0.912 | 0.958 | 35.8 | 41 |

**Model terpilih (mAP50-95 val tertinggi): E3_imgsz416**

Hasil di data TEST (sekali saja):

| Kelas | P | R | mAP50 | mAP50-95 | ms/img |
|---|---|---|---|---|---|
| all | 0.948 | 0.952 | 0.953 | 0.714 | 23.9 |
| Jose | 0.962 | 1.000 | 0.995 | 0.759 | — |
| Rizky | 0.955 | 1.000 | 0.995 | 0.863 | — |
| Dafa | 0.928 | 0.857 | 0.869 | 0.521 | — |

~42 FPS

---

## Analisis

### E1 vs E2: Manfaat Transfer Learning

E1 (pretrained COCO) menghasilkan mAP50-95 val **0.661**, sedangkan E2 (dari nol) hanya **0.508**. Selisih 0.153 poin membuktikan transfer learning membantu, terutama dengan data terbatas. E2 juga butuh waktu lebih lama (237 menit vs 141 menit).

### E1 vs E3: Efek imgsz=416

Menurunkan imgsz dari 640 ke 416 meningkatkan mAP50-95 val (0.661 → 0.691) sekaligus mempercepat training (41 menit) dan inferensi (35.8 ms vs 54.1 ms per gambar). Kemungkinan karena wajah di frame 1920×1080 sudah cukup besar untuk resolusi 416.

### Model Terpilih: E3

Dipilih berdasarkan mAP50-95 val tertinggi (0.691), sesuai kriteria slide 31 PPT.

### Hasil di Data Test

mAP50-95 test 0.714 lebih tinggi dari val (0.691), model tidak overfitting. Kecepatan 23.9 ms per gambar (~42 FPS) melampaui target desain ≥ 10 FPS. Dafa memiliki mAP50-95 terendah (0.521), kemungkinan karena variasi jarak kurang.

### Perbandingan dengan Target Desain

| Target | Nilai Target | Hasil |
|---|---|---|
| mAP50-95 | ≥ 0,6 | 0,714 ✅ |
| FPS | ≥ 10 | ~42 FPS ✅ |
| Latensi | ≤ 500 ms | 23,9 ms ✅ |
| Akurasi per kelas | ≥ 90% | Jose 99,5%, Rizky 99,5%, Dafa 86,9% ⚠️ |
| FAR | ≤ 5% | belum diukur ⚠️ |

---

## Keterbatasan

- Data dari 2 sesi dengan kondisi serupa, angka mungkin optimistis untuk lapangan.
- FAR belum diukur karena tidak ada data wajah orang lain.
- Dataset kecil (80 gambar per kelas) membuat angka per kelas sensitif.

---

## File yang Dikumpulkan

- `hasil_praktik_4222401085.txt`
- `hasil_praktik_4222401085.csv`
- `runs/detect/E1_pretrained/results.png`
- `runs/detect/E3_imgsz416/results.png`
- `README.md`

---

*Program Studi Teknologi Rekayasa Robotika · Politeknik Negeri Batam*
