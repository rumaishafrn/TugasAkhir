#!/usr/bin/env python3
"""
Langkah 2 -- Kalibrasi jarak kamera-objek (studi kasus sampah terapung).

Mencocokkan model D = a/s + b terhadap data kalibrasi menggunakan regresi
non-linear kuadrat terkecil. Jalankan ulang langkah ini apabila penelitian
direplikasi dengan kamera selain Logitech C310, karena konstanta kalibrasi
bersifat spesifik terhadap perangkat.

Format CSV masukan memerlukan kolom: w_px, h_px, distance_cm.

Contoh penggunaan
-----------------
    python scripts/02_calibrate_distance.py \
        --csv data/calibration_distance.csv \
        --plot assets/kurva_kalibrasi.png
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

import numpy as np

from src.calibration.distance import fit_distance_model, inverse_distance_model


def read_calibration_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    s_values, distances = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            s_values.append(float(row["w_px"]) + float(row["h_px"]))
            distances.append(float(row["distance_cm"]))
    return np.array(s_values), np.array(distances)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/calibration_distance.csv")
    parser.add_argument("--plot", default=None, help="Path keluaran grafik kalibrasi")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Berkas kalibrasi tidak ditemukan: {csv_path}")

    s_values, distances = read_calibration_csv(csv_path)
    a, b = fit_distance_model(s_values, distances)

    predicted = inverse_distance_model(s_values, a, b)
    residuals = distances - predicted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((distances - distances.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("=" * 60)
    print("HASIL KALIBRASI JARAK KAMERA DAN OBJEK")
    print("=" * 60)
    print(f"Jumlah titik data : {s_values.size}")
    print(f"Konstanta a       : {a:.2f}")
    print(f"Konstanta b       : {b:.5f}")
    print(f"Persamaan akhir   : D = {a:.2f} / s + {b:.5f}")
    print(f"RMSE              : {rmse:.3f} cm")
    print(f"R kuadrat         : {r_squared:.5f}")
    print("\nPerbarui nilai DIST_A dan DIST_B pada src/common/config.py "
          "bila hendak memakai hasil kalibrasi ini.")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib belum terpasang, grafik dilewati.")
            return

        s_grid = np.linspace(s_values.min() * 0.95, s_values.max() * 1.05, 300)
        plt.figure(figsize=(8, 5))
        plt.scatter(s_values, distances, label="Data kalibrasi", zorder=3)
        plt.plot(s_grid, inverse_distance_model(s_grid, a, b),
                 label=f"D = {a:.0f} / s + {b:.3f}", zorder=2)
        plt.xlabel("s = w + h (piksel)")
        plt.ylabel("D (cm)")
        plt.title("Kurva Kalibrasi Jarak Kamera dan Objek")
        plt.legend()
        plt.grid(alpha=0.3)
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"\nGrafik tersimpan: {args.plot}")


if __name__ == "__main__":
    main()
