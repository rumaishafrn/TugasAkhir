#!/usr/bin/env python3
"""
Langkah 4 -- Menjalankan pipeline estimasi volume rumput laut (metode PLT).

Menjalankan kalibrasi dua tahap terlebih dahulu, kemudian mengestimasi volume
seluruh foto sampel dan mengevaluasinya terhadap ground truth.

Format CSV masukan memerlukan kolom: photo_id, image_path, zone,
photo_number, gt_length_cm, gt_width_cm, gt_thickness_cm.

Contoh penggunaan
-----------------
    python scripts/04_run_seaweed_pipeline.py \
        --weights weights/yolo11s-seg-seaweed.pt \
        --samples data/seaweed_samples.csv \
        --mde dav2-large \
        --output results/seaweed_dav2_large.csv
"""

from __future__ import annotations

# Memastikan direktori akar repositori berada pada sys.path, sehingga script
# dapat dijalankan langsung tanpa perlu menyetel PYTHONPATH secara manual.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import csv
from collections import defaultdict
from pathlib import Path

from src.common.metrics import evaluate_volume
from src.seaweed.pipeline import SeaweedVolumePipeline, load_samples_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--samples", required=True, help="CSV daftar sampel")
    parser.add_argument("--mde", default="dav2-large")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", default="results/seaweed_estimation.csv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    samples = load_samples_from_csv(args.samples)
    print(f"Memuat {len(samples)} foto sampel dari {args.samples}")

    pipeline = SeaweedVolumePipeline(
        yolo_weights=args.weights,
        mde_model=args.mde,
        conf=args.conf,
        device=args.device,
    )

    print("\n[Tahap 1] Kalibrasi panjang fokus per zona dan skala ketebalan per grup")
    pipeline.calibrate(samples)

    print("\n[Tahap 2] Estimasi volume per foto")
    results = pipeline.estimate(samples)

    if not results:
        print("Tidak ada sampel yang berhasil diestimasi.")
        return

    predicted = [float(r["volume_pred_cm3"]) for r in results]
    ground_truth = [float(r["volume_gt_cm3"]) for r in results]
    metrics = evaluate_volume(predicted, ground_truth)

    print("\n" + "=" * 60)
    print(f"EVALUASI KESELURUHAN -- {args.mde}")
    print("=" * 60)
    print(metrics)

    # Rincian per zona, mereproduksi Tabel 4.8 pada buku Tugas Akhir.
    per_zone: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for result in results:
        per_zone[str(result["zone"])].append(
            (float(result["volume_pred_cm3"]), float(result["volume_gt_cm3"]))
        )

    print("\nRincian per zona:")
    for zone, pairs in sorted(per_zone.items()):
        zone_metrics = evaluate_volume([p for p, _ in pairs], [g for _, g in pairs])
        print(f"\n  Zona {zone} (n = {zone_metrics.n})")
        print(f"    MAE     : {zone_metrics.mae:.2f} cm3")
        print(f"    MAPE    : {zone_metrics.mape:.2f} %")
        print(f"    Akurasi : {zone_metrics.accuracy:.2f} %")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nHasil tersimpan: {output_path}")


if __name__ == "__main__":
    main()
