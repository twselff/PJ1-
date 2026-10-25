"""
prepare_dataset.py
Menyiapkan dataset wajah dari dataset publik (LFW subset)
untuk 3 kelas: Jose, Rizky, Dafa
NIM / Nama : 4222401085 / Jose Manuel Sinurat
"""

import os
import random
import shutil
import urllib.request
import tarfile
from pathlib import Path

# ==============================================================
# KONFIGURASI
# ==============================================================
DATASET_DIR = Path("dataset_v1")
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
CLASSES = ["Jose", "Rizky", "Dafa"]
SEED = 42

# LFW public subset (digunakan sebagai proxy wajah per kelas)
# Setiap kelas dipetakan ke 1 identitas dari LFW
LFW_IDENTITIES = {
    "Jose":  "George_W_Bush",    # 530 gambar — kelas utama
    "Rizky": "Colin_Powell",     # 236 gambar
    "Dafa":  "Tony_Blair",       # 144 gambar
}
LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"

random.seed(SEED)

def download_lfw(dest: Path):
    tgz = dest / "lfw-deepfunneled.tgz"
    if not tgz.exists():
        print(f"Downloading LFW deep-funneled subset...")
        urllib.request.urlretrieve(LFW_URL, tgz)
        print("Download selesai.")
    else:
        print("File LFW sudah ada, skip download.")
    lfw_dir = dest / "lfw_deepfunneled"
    if not lfw_dir.exists():
        print("Extracting...")
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(dest)
        print("Extract selesai.")
    return lfw_dir / "lfw_deepfunneled"

def collect_images(lfw_dir: Path, identity: str, n: int = 80):
    """Ambil n gambar dari folder identitas LFW."""
    folder = lfw_dir / identity
    imgs = sorted(folder.glob("*.jpg"))
    if len(imgs) < n:
        # Ulangi jika kurang
        imgs = (imgs * ((n // len(imgs)) + 1))[:n]
    else:
        imgs = random.sample(imgs, n)
    return imgs

def make_yolo_label(img_path: Path, class_id: int, out_dir: Path):
    """Buat dummy YOLO label (wajah di tengah, bbox ~0.4x0.5)."""
    label_name = img_path.stem + ".txt"
    label_path = out_dir / label_name
    with open(label_path, "w") as f:
        f.write(f"{class_id} 0.500 0.450 0.400 0.500\n")

def split_and_copy(images, class_id, class_name, base_dir):
    random.shuffle(images)
    n = len(images)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    splits_map = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split, imgs in splits_map.items():
        img_out = base_dir / split / "images"
        lbl_out = base_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(imgs):
            dst_name = f"{class_name}_{i:03d}.jpg"
            dst_img  = img_out / dst_name
            shutil.copy(src, dst_img)
            make_yolo_label(dst_img, class_id, lbl_out)

    print(f"  {class_name}: train={len(splits_map['train'])}, "
          f"val={len(splits_map['val'])}, test={len(splits_map['test'])}")

def main():
    tmp = Path("_lfw_tmp")
    tmp.mkdir(exist_ok=True)

    lfw_dir = download_lfw(tmp)

    print("\nMenyusun dataset...")
    for class_id, class_name in enumerate(CLASSES):
        identity = LFW_IDENTITIES[class_name]
        print(f"  Mengambil gambar untuk kelas '{class_name}' dari {identity}...")
        imgs = collect_images(lfw_dir, identity, n=80)
        split_and_copy(imgs, class_id, class_name, DATASET_DIR)

    print(f"\nDataset siap di '{DATASET_DIR}/'")
    print("Total kelas :", len(CLASSES))
    print("Nama kelas  :", CLASSES)

if __name__ == "__main__":
    main()
