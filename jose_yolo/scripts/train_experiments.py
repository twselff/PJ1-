"""
train_experiments.py
Menjalankan 3 eksperimen training YOLO sesuai modul praktikum RET503.

NIM / Nama : 4222401085 / Jose Manuel Sinurat
Dataset    : dataset_v1 (Jose, Rizky, Dafa — 3 kelas, 240 gambar, 2 sesi)
Model      : YOLO26n | Device: CPU AMD Ryzen 3 7320U

Eksperimen:
    E1_pretrained  — bobot COCO pretrained, imgsz=640
    E2_scratch     — bobot acak (dari nol), imgsz=640
    E3_imgsz416    — bobot COCO pretrained, imgsz=416

Jalankan:
    pip install ultralytics
    python scripts/train_experiments.py
"""

from ultralytics import YOLO
import time

DATA     = "dataset_v1/data.yaml"
EPOCHS   = 100
BATCH    = 16
PROJECT  = "runs/detect"
DEVICE   = "cpu"

EXPERIMENTS = [
    {
        "name":     "E1_pretrained",
        "weights":  "yolo11n.pt",   # pretrained COCO
        "imgsz":    640,
        "pretrained": True,
    },
    {
        "name":     "E2_scratch",
        "weights":  "yolo11n.yaml", # dari nol (random weights)
        "imgsz":    640,
        "pretrained": False,
    },
    {
        "name":     "E3_imgsz416",
        "weights":  "yolo11n.pt",   # pretrained COCO
        "imgsz":    416,
        "pretrained": True,
    },
]

results_log = []

for exp in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"Menjalankan: {exp['name']}")
    print(f"  weights : {exp['weights']}")
    print(f"  imgsz   : {exp['imgsz']}")
    print(f"{'='*60}")

    model = YOLO(exp["weights"])

    t0 = time.time()
    results = model.train(
        data       = DATA,
        epochs     = EPOCHS,
        batch      = BATCH,
        imgsz      = exp["imgsz"],
        name       = exp["name"],
        project    = PROJECT,
        device     = DEVICE,
        exist_ok   = True,
        verbose    = False,
    )
    elapsed_min = (time.time() - t0) / 60

    # Ambil metrik dari results
    metrics = results.results_dict
    map50    = metrics.get("metrics/mAP50(B)",    0)
    map5095  = metrics.get("metrics/mAP50-95(B)", 0)
    prec     = metrics.get("metrics/precision(B)",0)
    rec      = metrics.get("metrics/recall(B)",   0)

    # Ukur inferensi speed
    val_res = model.val(data=DATA, split="val", verbose=False)
    speed_ms = val_res.speed.get("inference", 0)

    entry = {
        "Eksperimen": exp["name"],
        "Perubahan":  "bobot acak" if not exp["pretrained"] else
                      (f"imgsz={exp['imgsz']}" if exp["imgsz"] != 640 else "-"),
        "mAP50":      round(map50,   3),
        "mAP50-95":   round(map5095, 3),
        "P":          round(prec,    3),
        "R":          round(rec,     3),
        "ms":         round(speed_ms,1),
        "menit":      round(elapsed_min),
    }
    results_log.append(entry)
    print(f"\nHasil {exp['name']}: mAP50={map50:.3f} | mAP50-95={map5095:.3f} | {elapsed_min:.0f} menit")

# Pilih model terbaik (mAP50-95 val tertinggi)
best = max(results_log, key=lambda x: x["mAP50-95"])
print(f"\nModel terpilih: {best['Eksperimen']} (mAP50-95={best['mAP50-95']})")

# ==============================================================
# Evaluasi final di data TEST (sekali saja, model terpilih)
# ==============================================================
best_weights = f"runs/detect/{best['Eksperimen']}/weights/best.pt"
final_model  = YOLO(best_weights)
test_res     = final_model.val(data=DATA, split="test", verbose=True)

print("\nHasil TEST selesai — cek runs/detect/ untuk detail.")
print("Jalankan scripts/export_results.py untuk membuat CSV & TXT.")
