"""
export_results.py
Membaca hasil training dari runs/detect/ dan mengekspor ke:
    - hasil_praktik_4222401085.csv
    - hasil_praktik_4222401085.txt

NIM / Nama : 4222401085 / Jose Manuel Sinurat
"""

import csv
import os
from pathlib import Path

NIM  = "4222401085"
NAMA = "Jose Manuel Sinurat"

# ==============================================================
# Isi hasil — di-update otomatis oleh train_experiments.py
# atau isi manual setelah training selesai
# ==============================================================

EXPERIMENTS = [
    # (nama, perubahan, mAP50_val, mAP50-95_val, P_val, R_val, ms_val, menit_train)
    ("E1_pretrained", "-",          0.968, 0.661, 0.935, 0.972, 54.1, 141),
    ("E2_scratch",    "bobot acak", 0.931, 0.508, 0.884, 0.829, 46.3, 237),
    ("E3_imgsz416",   "imgsz=416",  0.971, 0.691, 0.912, 0.958, 35.8,  41),
]

BEST_EXP = "E3_imgsz416"   # mAP50-95 val tertinggi

# Hasil di data TEST (sekali saja, model terpilih)
TEST_ALL   = ("all",   0.948, 0.952, 0.953, 0.714, 23.9)
TEST_KELAS = [
    ("Jose",  0.962, 1.000, 0.995, 0.759),
    ("Rizky", 0.955, 1.000, 0.995, 0.863),
    ("Dafa",  0.928, 0.857, 0.869, 0.521),
]

# ==============================================================

header = ["Eksperimen", "Perubahan", "mAP50", "mAP50-95", "P", "R", "ms", "menit"]

csv_path = Path(f"hasil_praktik_{NIM}.csv")
txt_path = Path(f"hasil_praktik_{NIM}.txt")

# ---------- CSV ----------
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["NIM / Nama", f"{NIM} / {NAMA}"])
    writer.writerow(["Dataset", "dataset_v1"])
    writer.writerow([])
    writer.writerow(header)
    for row in EXPERIMENTS:
        writer.writerow(row)
    writer.writerow([])
    writer.writerow(["Model terpilih (mAP50-95 val tertinggi):", BEST_EXP])
    writer.writerow([])
    writer.writerow(["=== HASIL DATA TEST ==="])
    writer.writerow(["Kelas", "P", "R", "mAP50", "mAP50-95", "ms/img"])
    n, p, r, m50, m5095, ms = TEST_ALL
    writer.writerow([n, p, r, m50, m5095, ms])
    for kelas in TEST_KELAS:
        writer.writerow(kelas)

print(f"CSV  ditulis: {csv_path}")

# ---------- TXT ----------
col_w = [16, 12, 8, 10, 8, 8, 8, 7]

def row_fmt(cells):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

lines = [
    "=" * 72,
    f"Hasil Praktik Pj1 — Smart Locker Face ID",
    "=" * 72,
    f"NIM / Nama : {NIM} / {NAMA}",
    f"Dataset    : dataset_v1 (Jose, Rizky, Dafa — 3 kelas, 240 gambar, 2 sesi)",
    f"Model      : YOLO26n | Device: CPU AMD Ryzen 3 7320U",
    "",
    "--- Tabel Eksperimen ---",
    "",
    row_fmt(header),
    "-" * 72,
]
for row in EXPERIMENTS:
    lines.append(row_fmt(row))

lines += [
    "",
    f"Model terpilih (mAP50-95 val tertinggi): {BEST_EXP}",
    "",
    "--- Hasil di Data TEST (sekali saja) ---",
    "",
    f"  all   P={TEST_ALL[1]}  R={TEST_ALL[2]}  mAP50={TEST_ALL[3]}  "
    f"mAP50-95={TEST_ALL[4]}  {TEST_ALL[5]}ms/img  ~{round(1000/TEST_ALL[5])} FPS",
]
for k in TEST_KELAS:
    lines.append(f"  {k[0]:<6} P={k[1]}  R={k[2]}  mAP50={k[3]}  mAP50-95={k[4]}")

lines += [
    "",
    "--- Analisis Singkat ---",
    "",
    "E1 vs E2 (Transfer Learning):",
    "  E1 pretrained mAP50-95=0.661, E2 scratch mAP50-95=0.508.",
    "  Selisih 0.153 poin — transfer learning membantu dengan data terbatas.",
    "  E2 juga butuh waktu 237 menit vs 141 menit.",
    "",
    "E1 vs E3 (imgsz 640→416):",
    "  mAP50-95 meningkat 0.661→0.691, training 141→41 menit,",
    "  inferensi 54.1→35.8 ms. Wajah di frame 1080p cukup besar untuk 416px.",
    "",
    "Model Terpilih: E3_imgsz416",
    "  Dipilih berdasarkan mAP50-95 val tertinggi (0.691).",
    "",
    "Hasil di Data Test:",
    "  mAP50-95 test 0.714 > val 0.691 — tidak overfitting.",
    "  ~42 FPS melampaui target desain ≥10 FPS.",
    "  Dafa mAP50-95 terendah (0.521) — variasi jarak kurang.",
    "",
    "--- Perbandingan dengan Target Desain ---",
    "",
    "  mAP50-95    : ≥0.6     → 0.714  ✓",
    "  FPS         : ≥10      → ~42    ✓",
    "  Latensi     : ≤500 ms  → 23.9ms ✓",
    "  Akurasi/kls : ≥90%     → Jose 99.5%, Rizky 99.5%, Dafa 86.9%  ⚠",
    "  FAR         : ≤5%      → belum diukur  ⚠",
    "",
    "--- Keterbatasan ---",
    "",
    "  - Data dari 2 sesi dengan kondisi serupa, angka mungkin optimistis.",
    "  - FAR belum diukur karena tidak ada data wajah orang lain.",
    "  - Dataset kecil (80 gambar/kelas) membuat angka per kelas sensitif.",
    "",
    "=" * 72,
    "Program Studi Teknologi Rekayasa Robotika · Politeknik Negeri Batam",
    "=" * 72,
]

with open(txt_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"TXT  ditulis: {txt_path}")
print("Selesai.")
