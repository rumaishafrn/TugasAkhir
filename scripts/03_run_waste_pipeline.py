#!/usr/bin/env python3
"""
Langkah 3 -- Menjalankan pipeline estimasi volume sampah terapung.

Memproses video atau citra masukan, lalu mengeluarkan volume akhir tiap objek
beserta metrik evaluasi terhadap ground truth apabila tersedia.

Contoh penggunaan
-----------------
    # Estimasi dari video menggunakan model MDE terbaik
    python scripts/03_run_waste_pipeline.py \
        --weights weights/yolov8s-seg-waste.pt \
        --video data/videos/uji.mp4 \
        --mde zoedepth-kitti \
        --output results/waste_zoedepth_kitti.csv

    # Evaluasi terhadap ground truth
    python scripts/03_run_waste_pipeline.py \
        --weights weights/yolov8s-seg-waste.pt \
        --video data/videos/uji.mp4 \
        --ground-truth data/ground_truth_waste.csv \
        --output results/waste_eval.csv
"""

from __future__ import annotations

# Memastikan direktori akar repositori berada pada sys.path, sehingga script
# dapat dijalankan langsung tanpa perlu menyetel PYTHONPATH secara manual.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import csv
from pathlib import Path

from src.common.metrics import evaluate_volume
from src.waste.pipeline import WasteVolumePipeline


def load_ground_truth(path: Path) -> dict[str, float]:
    """Memuat volume ground truth per kelas dari CSV."""
    mapping: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[row["class_name"].strip().lower()] = float(row["volume_gt_cm3"])
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Bobot YOLO hasil pelatihan")
    parser.add_argument("--video", help="Path video masukan")
    parser.add_argument("--image", help="Path citra masukan")
    parser.add_argument("--mde", default="zoedepth-kitti",
                        help="Kunci model MDE, lihat src/common/depth_models.py")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--output", default="results/waste_estimation.csv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.video and not args.image:
        raise SystemExit("Tentukan salah satu dari --video atau --image.")

    print(f"Memuat pipeline: YOLO={args.weights}, MDE={args.mde}")
    pipeline = WasteVolumePipeline(
        yolo_weights=args.weights,
        mde_model=args.mde,
        conf=args.conf,
        device=args.device,
    )

    rows: list[dict[str, object]] = []

    if args.video:
        print(f"\nMemproses video: {args.video}")
        result = pipeline.process_video(
            args.video, frame_stride=args.frame_stride, max_frames=args.max_frames
        )
        print(f"\nTotal frame diproses: {result['frames_processed']}")
        print(f"Jumlah track terbentuk: {len(result['tracks'])}\n")

        for track_id, info in result["tracks"].items():   # type: ignore[union-attr]
            rows.append({
                "track_id": track_id,
                "class_name": info["class_name"],
                "volume_pred_cm3": round(float(info["volume_cm3"]), 2),
                "n_frames": info["n_frames"],
            })
    else:
        print(f"\nMemproses citra: {args.image}")
        for index, obs in enumerate(pipeline.process_image(args.image)):
            rows.append({
                "track_id": index,
                "class_name": obs.class_name,
                "volume_pred_cm3": round(obs.volume_cm3, 2),
                "n_frames": 1,
            })

    if not rows:
        print("Tidak ada objek terdeteksi.")
        return

    # Evaluasi terhadap ground truth bila tersedia.
    if args.ground_truth:
        gt_map = load_ground_truth(Path(args.ground_truth))
        predicted, ground_truth = [], []

        for row in rows:
            key = str(row["class_name"]).strip().lower()
            if key in gt_map:
                row["volume_gt_cm3"] = gt_map[key]
                predicted.append(float(row["volume_pred_cm3"]))
                ground_truth.append(gt_map[key])

        if predicted:
            metrics = evaluate_volume(predicted, ground_truth)
            print("=" * 60)
            print(f"EVALUASI ESTIMASI VOLUME -- {args.mde}")
            print("=" * 60)
            print(metrics)
            print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Hasil tersimpan: {output_path}")


if __name__ == "__main__":
    main()
