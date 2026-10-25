# Praktik Terpadu YOLO: Jose Manuel Sinurat (4222401085)

Mata kuliah RET503, Praktik Terpadu Minggu 4-7. Transfer learning YOLO untuk mendeteksi tiga jenis buah.

## Dataset

- Sumber: Roboflow Universe, `apple-banana-orange` v3 (workspace `li-yu`), lisensi CC BY 4.0. https://universe.roboflow.com/li-yu/apple-banana-orange-zja84
- 3 kelas: `apple`, `banana`, `orange`. Total 300 gambar 640x640 tanpa augmentasi, 581 kotak anotasi (apple 191, banana 202, orange 188), dan 4 gambar tanpa objek.
- Dua baris label aslinya berupa poligon. Keduanya diubah menjadi kotak (nilai minimum dan maksimum titik) agar sesuai format deteksi YOLO.
- Cek anotasi (`01_cek_anotasi.py`): 0 masalah ditemukan. Pembagian data dengan `05_split_dataset_yolo.py` (sidik jari label `931918507653`).
- Model dasar: `yolo26n`, 30 epoch, batch 16, imgsz 416, seed 0. Perubahan E3: `imgsz=640`.

## Hasil eksperimen (data VAL)

| eksperimen | perubahan | pretrained | mAP50 | mAP50-95 | precision | recall | inferensi_ms | waktu_latih_menit | imgsz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_pretrained | - | True | 0.9023 | 0.6391 | 0.85 | 0.8787 | 5.16 | 2.02 | 416 |
| E2_scratch | - | False | 0.4723 | 0.2262 | 0.456 | 0.4766 | 2.9 | 1.69 | 416 |
| E3_tuning | imgsz=640 | True | 0.8976 | 0.6287 | 0.8415 | 0.8836 | 6.29 | 2.12 | 640 |

## Hasil uji (data TEST, sekali saja)

- Model terpilih (mAP50-95 val tertinggi): E1_pretrained
- Hasil di data TEST (sekali saja): mAP50 0.946 | mAP50-95 0.694

Rekap lengkap ada di `hasil_praktik_4222401085.txt` dan `hasil_praktik_4222401085.csv`.

## Analisis

### 1. Ringkasan

Tiga eksperimen YOLO (`yolo26n`) dibandingkan untuk mendeteksi apple, banana, dan orange. Model pretrained (E1) terbaik dengan mAP50-95 0,639 pada data validasi dan 0,694 pada data test. Model yang dilatih dari nol (E2) hanya mencapai 0,226. Memperbesar ukuran gambar menjadi 640 (E3) tidak memberi peningkatan (0,629) dan membuat inferensi sekitar 22% lebih lambat (5,16 menjadi 6,29 ms).

### 2. Dataset dan pengaturan

Dataset dibagi dengan skrip `05_split_dataset_yolo.py` menjadi tiga bagian:

| Bagian | Gambar | apple | banana | orange | Total kotak |
| --- | --- | --- | --- | --- | --- |
| train | 210 | 127 | 148 | 141 | 416 |
| val | 60 | 47 | 41 | 32 | 120 |
| test | 30 | 17 | 13 | 15 | 45 |

Dua gambar di train dan dua di val tidak memiliki objek (gambar latar). Semua eksperimen memakai pengaturan yang sama (30 epoch, batch 16, imgsz 416, seed 0), kecuali yang sengaja diubah: E2 tanpa bobot pretrained dan E3 dengan imgsz 640. Gambar asli dataset sudah berukuran 640x640, jadi E1 dan E2 memperkecil gambar saat pelatihan, sedangkan E3 memakai ukuran aslinya.

### 3. Cara membaca metrik

mAP50 menghitung ketepatan deteksi bila kotak prediksi tumpang tindih minimal 50% dengan kotak sebenarnya. mAP50-95 merata-ratakan ketepatan pada ambang 50% sampai 95%, sehingga menuntut kotak yang lebih presisi. Pada dataset ini mAP50 E1 dan E3 sudah sekitar 0,90 dan hampir sama, sehingga mAP50-95 dipakai sebagai penentu pemilihan model, sedangkan precision dan recall menjadi pelengkap.

### 4. E1 (pretrained) vs E2 (scratch)

Pada data validasi, E1 mencapai mAP50-95 0,639 dan E2 hanya 0,226, selisih sekitar 0,413. Perbedaan terlihat jelas pada kecepatan belajar. E1 sudah 0,533 pada epoch 1, melewati 0,6 pada epoch 16, dan puncaknya 0,637 pada epoch 19 (0,636 pada epoch 30), sehingga kurvanya sudah mendatar. E2 hampir nol pada epoch 1 (0,0001), baru 0,029 pada epoch 10, 0,147 pada epoch 20, dan 0,227 pada epoch 30 yang sekaligus puncaknya. Artinya E2 belum konvergen dalam 30 epoch. Loss pada epoch 30 juga mendukung hal ini: loss latih E1 (box 0,804; cls 0,475) jauh lebih rendah dari E2 (box 2,045; cls 2,823).

Penyebab yang paling mungkin adalah transfer learning. Bobot pretrained sudah memuat fitur visual dasar seperti tepi, bentuk, dan warna, sehingga E1 tinggal menyesuaikan diri ke tiga kelas baru. E2 harus mempelajarinya dari awal hanya dari 210 gambar latih. Dugaan saya, selisih kedua model dapat mengecil bila E2 dilatih lebih lama, tetapi hal ini belum diuji. Waktu latih keduanya hampir sama (2,02 vs 1,69 menit) karena arsitekturnya sama.

![Grafik pelatihan E1](E1/results.png)

![Grafik pelatihan E2](E2/results.png)

### 5. E3 (tuning imgsz=640)

Satu-satunya perubahan pada E3 adalah ukuran gambar masukan dari 416 menjadi 640. Hasilnya mAP50-95 0,629, sedikit di bawah E1 (0,639), dengan inferensi naik dari 5,16 ke 6,29 ms dan waktu latih dari 2,02 ke 2,12 menit. Komputasi per gambar naik sekitar 2,4 kali (rasio luas 640² terhadap 416²), tetapi tidak ada informasi baru karena gambar asli sudah 640x640 dan yang berubah hanyalah E1 memakai versi yang diperkecil.

Kurva E3 berbeda dari E1: mAP50-95 sempat turun dari 0,502 (epoch 1) ke 0,472 (epoch 5), baru melewati 0,6 pada epoch 23, dan puncaknya 0,629 justru di epoch terakhir (epoch 30). Jadi E3 masih naik saat pelatihan berhenti dan kemungkinan masih bisa membaik dengan epoch lebih banyak, namun ini belum diuji. Selisih akhir 0,010 sangat kecil dan berasal dari satu kali percobaan dengan data validasi hanya 60 gambar, sehingga belum cukup untuk menyimpulkan E3 pasti lebih buruk. Yang dapat disimpulkan, pada pengaturan 30 epoch ini memperbesar imgsz tidak memberi keuntungan yang sepadan dengan biayanya.

![Grafik pelatihan E3](E3/results.png)

### 6. Pemilihan model, hasil uji, dan kelas yang sulit

Model dipilih berdasarkan mAP50-95 validasi tertinggi, yaitu E1 (0,639). Model ini diuji sekali pada data test dan memperoleh mAP50 0,946 serta mAP50-95 0,694. Nilai test lebih tinggi dari validasi, tetapi data test hanya 30 gambar dengan 45 objek, sehingga selisih itu lebih mungkin mencerminkan sampel yang kecil daripada kemampuan generalisasi yang lebih baik. Angka test sebaiknya dibaca sebagai gambaran kasar.

**Kelas yang sulit.** Confusion matrix E1 pada data validasi (120 objek: apple 47, banana 41, orange 32) menunjukkan hasil berikut. Kolom adalah kelas sebenarnya dan baris adalah prediksi.

![Confusion matrix E1](E1/confusion_matrix.png)

- banana paling baik: 37 dari 41 objek (90,2%) terklasifikasi benar.
- apple: 27 dari 47 (57,4%) benar, dan 17 apple (36,2%) terbaca sebagai orange.
- orange: 18 dari 32 (56,3%) benar, 7 terbaca sebagai apple dan 7 sebagai banana.
- Kesalahan antar kelas yang lain kecil (3 apple terbaca banana, 3 banana terbaca apple, 1 banana terbaca orange).
- Kotak prediksi yang tidak sesuai objek apa pun (kolom `background`) berjumlah 587: 340 berlabel banana, 155 apple, dan 92 orange. Baris `background` kosong, artinya tidak ada objek yang tercatat terlewat. Dugaan saya, matriks ini dihitung pada ambang confidence yang rendah sehingga hampir semua objek tertangkap oleh suatu kotak dan kotak tambahan ikut terhitung banyak; hal ini belum diverifikasi, dan angkanya tidak langsung sebanding dengan precision dan recall di tabel.

**Pengamatan pada gambar prediksi E1** (`E1/val_batch0_pred.jpg` dan `E1/val_batch1_pred.jpg`, hanya 2 batch sehingga hanya contoh):
- Buah tunggal di latar polos hampir selalu terdeteksi dengan kotak yang pas dan confidence tinggi (sekitar 0,8 sampai 0,9).
- Gambar apel kartun terbaca orange (confidence 0,7), dan pada foto campuran ada jeruk yang terbaca apple dengan confidence rendah (0,3).
- Apel hijau cenderung mendapat confidence rendah (0,3) atau tidak mendapat kotak sendiri saat berdampingan dengan apel merah.
- Pada buah yang bertumpuk atau bergerombol (pisang, apel di pohon), muncul beberapa kotak yang saling menimpa pada objek yang sama.

![Contoh prediksi E1 batch 0](E1/val_batch0_pred.jpg)

![Contoh prediksi E1 batch 1](E1/val_batch1_pred.jpg)

### 7. Keterbatasan dan saran

- **Dataset kecil.** Hanya 300 gambar (210 latih), dan data test hanya 30 gambar dengan 45 objek, sehingga angka test tidak stabil.
- **Satu kali percobaan per eksperimen.** Selisih kecil seperti E1 vs E3 (0,010) belum bisa dipastikan nyata. Sebaiknya diulang dengan beberapa seed dan dilaporkan rata-rata serta simpangannya.
- **E2 dan E3 masih naik di epoch 30.** Perbandingan yang lebih adil memerlukan epoch lebih banyak (misalnya 60), belum diuji.
- **Gambar beragam dan campuran.** Dataset memuat gambar web dengan gaya berbeda (foto, kartun, sulaman) dan foto berisi beberapa jenis buah, yang menyulitkan model. Penambahan data dan augmentasi kemungkinan membantu, terutama untuk apel hijau dan buah yang bergerombol.
- **Kotak berlebih.** Confusion matrix menunjukkan banyak kotak tanpa objek yang sesuai. Penyebabnya perlu diselidiki dengan memeriksa gambar prediksi pada beberapa ambang confidence.
- **Konversi label.** Dua baris poligon diubah menjadi kotak sehingga kotaknya dapat sedikit lebih longgar dari bentuk objek.

### 8. Kesimpulan

Transfer learning terbukti mempercepat konvergensi dan memberi mAP50-95 jauh lebih tinggi dibanding melatih dari nol (0,639 vs 0,226). Memperbesar imgsz menjadi 640 tidak memberi keuntungan pada pengaturan ini dan hanya menambah waktu inferensi. E1 dipilih dan memperoleh hasil test mAP50 0,946 dan mAP50-95 0,694 dengan inferensi sekitar 5,2 ms per gambar. Kelas banana paling mudah dikenali, sedangkan apple dan orange sering tertukar, dan model cenderung menghasilkan kotak tambahan pada buah yang bergerombol.

## Isi repositori

```
.
├── README.md
├── hasil_praktik_4222401077.txt / .csv   # rekap hasil (E1, E2, E3 dan data test)
├── E1/  E2/  E3/                         # results.png/csv, confusion matrix, gambar prediksi
├── dataset_v1/data.yaml                  # konfigurasi pembagian data
└── data/
    └── dataset_yolo.zip                  # gambar, label YOLO, classes.txt
```

Catatan: `path:` di `dataset_v1/data.yaml` dan kolom `bobot` di CSV masih memakai path Colab (`/content/YOLO/YOLO/...`).
