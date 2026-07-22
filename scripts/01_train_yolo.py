#!/usr/bin/env python3
"""
Langkah 1 -- Pelatihan model YOLO instance segmentation.

Melatih satu atau seluruh varian YOLO pada dataset yang ditentukan, sesuai
konfigurasi pada Tabel 3.4 buku Tugas Akhir. Parameter yang tidak disebutkan
secara eksplisit mengikuti nilai bawaan Ultralytics.

Contoh penggunaan
-----------------
    # Melatih satu varian
    python scripts/01_train_yolo.py \
        --data datasets/waste/data.yaml \
        --model yolov8s-seg.pt \
        --epochs 100 \
        --name waste-yolov8s

    # Melatih keempat varian sekaligus untuk keperluan perbandingan
    python scripts/01_train_yolo.py \
        --data datasets/waste/data.yaml \
        --all-variants \
        --name-prefix waste
"""

from __future__ import annotations

# Memastikan direktori akar repositori berada pada sys.path, sehingga script
# dapat dijalankan langsung tanpa perlu menyetel PYTHONPATH secara manual.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
from pathlib import Path

from ultralytics import YOLO

VARIANTS = ["yolov8s-seg.pt", "yolo11s-seg.pt", "yolo12s-seg.pt", "yolo26s-seg.pt"]


def train_one(
    data: str,
    model_name: str,
    epochs: int,
    imgsz: int,
    project: str,
    run_name: str,
    device: str,
) -> None:
    print(f"\n{'=' * 70}")
    print(f"Melatih {model_name} -> {run_name}")
    print(f"{'=' * 70}")

    model = YOLO(model_name)
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=run_name,
        device=device,
        exist_ok=True,
    )

    metrics = model.val()
    print(f"\nHasil validasi {model_name}:")
    print(f"  mask mAP@50    : {metrics.seg.map50:.4f}")
    print(f"  mask mAP@50-95 : {metrics.seg.map:.4f}")
    print(f"  box  mAP@50-95 : {metrics.box.map:.4f}")
    print(f"  Bobot tersimpan: {Path(results.save_dir) / 'weights' / 'best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path ke data.yaml dataset")
    parser.add_argument("--model", default="yolov8s-seg.pt", help="Bobot awal varian")
    parser.add_argument("--all-variants", action="store_true",
                        help="Melatih keempat varian untuk perbandingan")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="train")
    parser.add_argument("--name-prefix", default="run")
    parser.add_argument("--device", default="0", help="'0' untuk GPU, 'cpu' untuk CPU")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"Berkas data.yaml tidak ditemukan: {args.data}")

    if args.all_variants:
        for variant in VARIANTS:
            run_name = f"{args.name_prefix}-{variant.replace('.pt', '')}"
            train_one(args.data, variant, args.epochs, args.imgsz,
                      args.project, run_name, args.device)
        print("\nSeluruh varian selesai dilatih. Bandingkan nilai mask mAP@50-95 "
              "untuk menentukan model terbaik.")
    else:
        train_one(args.data, args.model, args.epochs, args.imgsz,
                  args.project, args.name, args.device)


if __name__ == "__main__":
    main()
