#!/usr/bin/env python3
"""
Langkah 5 -- Perbandingan keenam model MDE.

Menjalankan pipeline berulang kali dengan seluruh model MDE yang terdaftar,
lalu menyusun tabel peringkat berdasarkan MAE dan MAPE. Script ini
mereproduksi Tabel 4.4 (sampah terapung) dan Tabel 4.7 (rumput laut) pada
buku Tugas Akhir.

Contoh penggunaan
-----------------
    python scripts/05_compare_mde_models.py \
        --case waste \
        --weights weights/yolov8s-seg-waste.pt \
        --video data/videos/uji.mp4 \
        --ground-truth data/ground_truth_waste.csv \
        --output results/perbandingan_mde_sampah.csv
"""

from __future__ import annotations

# Memastikan direktori akar repositori berada pada sys.path, sehingga script
# dapat dijalankan langsung tanpa perlu menyetel PYTHONPATH secara manual.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import csv
import time
from pathlib import Path

from src.common.depth_models import MDE_REGISTRY, list_models
from src.common.metrics import evaluate_volume
from src.seaweed.pipeline import SeaweedVolumePipeline, load_samples_from_csv
from src.waste.pipeline import WasteVolumePipeline


def load_ground_truth(path: Path) -> dict[str, float]:
    mapping: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[row["class_name"].strip().lower()] = float(row["volume_gt_cm3"])
    return mapping


def evaluate_waste(args, model_key: str) -> tuple[list[float], list[float]]:
    pipeline = WasteVolumePipeline(
        yolo_weights=args.weights, mde_model=model_key, device=args.device
    )
    result = pipeline.process_video(
        args.video, frame_stride=args.frame_stride,
        max_frames=args.max_frames, verbose=False
    )
    gt_map = load_ground_truth(Path(args.ground_truth))

    predicted, ground_truth = [], []
    for info in result["tracks"].values():   # type: ignore[union-attr]
        key = str(info["class_name"]).strip().lower()
        if key in gt_map:
            predicted.append(float(info["volume_cm3"]))
            ground_truth.append(gt_map[key])
    return predicted, ground_truth


def evaluate_seaweed(args, model_key: str) -> tuple[list[float], list[float]]:
    samples = load_samples_from_csv(args.samples)
    pipeline = SeaweedVolumePipeline(
        yolo_weights=args.weights, mde_model=model_key, device=args.device
    )
    pipeline.calibrate(samples, verbose=False)
    results = pipeline.estimate(samples, verbose=False)

    predicted = [float(r["volume_pred_cm3"]) for r in results]
    ground_truth = [float(r["volume_gt_cm3"]) for r in results]
    return predicted, ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["waste", "seaweed"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--video", help="Wajib untuk --case waste")
    parser.add_argument("--ground-truth", help="Wajib untuk --case waste")
    parser.add_argument("--samples", help="Wajib untuk --case seaweed")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset model MDE. Kosong berarti seluruh model.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", default="results/perbandingan_mde.csv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.case == "waste" and not (args.video and args.ground_truth):
        raise SystemExit("--case waste memerlukan --video dan --ground-truth.")
    if args.case == "seaweed" and not args.samples:
        raise SystemExit("--case seaweed memerlukan --samples.")

    model_keys = args.models or list_models()
    rows: list[dict[str, object]] = []

    for model_key in model_keys:
        if model_key not in MDE_REGISTRY:
            print(f"Melewati model tidak dikenal: {model_key}")
            continue

        spec = MDE_REGISTRY[model_key]
        print(f"\n{'=' * 70}")
        print(f"Mengevaluasi: {spec.display_name} ({spec.depth_type})")
        print(f"{'=' * 70}")

        start = time.time()
        try:
            if args.case == "waste":
                predicted, ground_truth = evaluate_waste(args, model_key)
            else:
                predicted, ground_truth = evaluate_seaweed(args, model_key)
        except Exception as error:   # noqa: BLE001
            print(f"  Gagal: {error}")
            continue

        elapsed = time.time() - start

        if not predicted:
            print("  Tidak ada pasangan prediksi dan ground truth yang cocok.")
            continue

        metrics = evaluate_volume(predicted, ground_truth)
        print(metrics)
        print(f"Waktu proses      : {elapsed:.1f} detik")

        rows.append({
            "model": spec.display_name,
            "model_key": model_key,
            "depth_type": spec.depth_type,
            "n": metrics.n,
            "mean_gt_cm3": round(metrics.mean_gt, 2),
            "mean_pred_cm3": round(metrics.mean_pred, 2),
            "mae_cm3": round(metrics.mae, 2),
            "mae_std_cm3": round(metrics.mae_std, 2),
            "mape_pct": round(metrics.mape, 2),
            "mape_std_pct": round(metrics.mape_std, 2),
            "accuracy_pct": round(metrics.accuracy, 2),
            "elapsed_s": round(elapsed, 1),
        })

    if not rows:
        print("\nTidak ada model yang berhasil dievaluasi.")
        return

    rows.sort(key=lambda r: r["mape_pct"])   # type: ignore[arg-type,return-value]

    print(f"\n\n{'=' * 78}")
    print("PERINGKAT MODEL MDE (diurutkan berdasarkan MAPE)")
    print(f"{'=' * 78}")
    print(f"{'Model':<28}{'MAE (cm3)':>12}{'MAPE (%)':>11}{'Akurasi (%)':>13}")
    print("-" * 78)
    for row in rows:
        print(f"{row['model']:<28}{row['mae_cm3']:>12.2f}"
              f"{row['mape_pct']:>11.2f}{row['accuracy_pct']:>13.2f}")
    print("-" * 78)
    print(f"Model terbaik: {rows[0]['model']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nHasil tersimpan: {output_path}")


if __name__ == "__main__":
    main()
