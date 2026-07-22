"""
Pipeline estimasi volume studi kasus sampel rumput laut (metode PLT).

Implementasi ini merupakan realisasi Kode Semu 3.2 pada buku Tugas Akhir.
Berbeda dengan jalur sampah terapung yang langsung memakai matriks kamera,
panjang fokus pada jalur ini tidak diketahui karena akuisisi menggunakan dua
kamera ponsel yang berbeda. Prosedur karenanya dijalankan dalam dua tahap
atau two-pass.

Tahap 1 -- Kalibrasi
    a. Panjang fokus efektif per zona (Persamaan 3.12 dan 3.13):
           f_px_i    = ukuran_px_i * jarak_i / ukuran_cm_GT_i
           f_px_zona = median{ f_px_i : i dalam zona }
       Median dipilih alih-alih rata-rata agar tahan terhadap pencilan yang
       dapat muncul dari kesalahan segmentasi pada sebagian foto.

    b. Faktor skala ketebalan per grup (Persamaan 3.14 dan 3.15):
           s    = P95(d_rel) - P5(d_rel)
           sz_g = median{ t_GT / s : foto dalam grup g }
       Persentil ke-95 dan ke-5 dipakai, bukan nilai maksimum dan minimum,
       untuk mengabaikan pencilan kedalaman ekstrem di tepi objek.

Tahap 2 -- Estimasi volume per foto
       Pjg = P_px * jarak / f_px_zona
       Lbr = L_px * jarak / f_px_zona
       Tgi = s * sz_g
       Vol = Pjg * Lbr * Tgi                          (Persamaan 3.17)

Satu grup didefinisikan sebagai satu sampel yang terdiri atas enam foto,
dengan group_key = (nomor_foto - 1) // 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.common.config import (
    CONF_THRESHOLD_SEAWEED,
    DEPTH_PERCENTILE_HIGH,
    DEPTH_PERCENTILE_LOW,
    PHOTOS_PER_GROUP,
    PHOTO_DISTANCE_MAP,
    YOLO_IMGSZ,
)
from src.common.depth_models import DepthEstimator
from src.common.segmentation import InstanceSegmenter


@dataclass
class SeaweedSample:
    """Satu foto sampel rumput laut beserta ground truth-nya."""

    photo_id: str
    image_path: str
    zone: str
    photo_number: int             # nomor urut foto, dimulai dari 1
    gt_length_cm: float
    gt_width_cm: float
    gt_thickness_cm: float

    @property
    def group_key(self) -> int:
        """Kunci grup sampel: satu grup terdiri atas enam foto."""
        return (self.photo_number - 1) // PHOTOS_PER_GROUP

    @property
    def distance_cm(self) -> int:
        """Jarak akuisisi berdasarkan posisi foto dalam grup (Tabel 3.2)."""
        position = (self.photo_number - 1) % PHOTOS_PER_GROUP + 1
        return PHOTO_DISTANCE_MAP[position]

    @property
    def gt_volume_cm3(self) -> float:
        return self.gt_length_cm * self.gt_width_cm * self.gt_thickness_cm


def depth_span(depth_map: np.ndarray, mask: np.ndarray) -> float:
    """Menghitung rentang kedalaman objek -- Persamaan (3.14).

        s = P95(d_rel) - P5(d_rel)
    """
    values = depth_map[mask]
    if values.size == 0:
        raise ValueError("Mask tidak memuat piksel apa pun.")

    high = float(np.percentile(values, DEPTH_PERCENTILE_HIGH))
    low = float(np.percentile(values, DEPTH_PERCENTILE_LOW))
    return abs(high - low)


class SeaweedVolumePipeline:
    """Pipeline estimasi volume rumput laut dengan kalibrasi dua tahap.

    Contoh
    ------
    >>> pipeline = SeaweedVolumePipeline(
    ...     yolo_weights="weights/yolo11s-seg-seaweed.pt",
    ...     mde_model="dav2-large",
    ... )
    >>> pipeline.calibrate(samples)
    >>> hasil = pipeline.estimate(samples)
    """

    def __init__(
        self,
        yolo_weights: str,
        mde_model: str = "dav2-large",
        conf: float = CONF_THRESHOLD_SEAWEED,
        device: str | None = None,
    ) -> None:
        self.segmenter = InstanceSegmenter(
            weights_path=yolo_weights, conf=conf, imgsz=YOLO_IMGSZ, device=device
        )
        self.depth_estimator = DepthEstimator(mde_model, device=device)
        self.mde_model = mde_model

        self.focal_per_zone: dict[str, float] = {}
        self.thickness_scale_per_group: dict[int, float] = {}
        self._measurement_cache: dict[str, dict[str, float]] = {}

    # -- pengukuran mentah -----------------------------------------------------

    def _measure(self, sample: SeaweedSample) -> dict[str, float] | None:
        """Mengukur dimensi piksel dan rentang kedalaman satu foto.

        Hasil disimpan dalam cache agar tahap kalibrasi dan tahap estimasi
        tidak menjalankan inferensi dua kali untuk foto yang sama.
        """
        if sample.photo_id in self._measurement_cache:
            return self._measurement_cache[sample.photo_id]

        image = cv2.imread(sample.image_path)
        if image is None:
            return None

        detections = self.segmenter.predict(image)
        if not detections:
            return None

        # Satu foto memuat satu sampel, sehingga deteksi dengan mask terluas
        # diambil sebagai objek utama.
        detection = max(detections, key=lambda d: int(d.mask.sum()))
        length_px, width_px = detection.obb_length_width()
        if length_px <= 0 or width_px <= 0:
            return None

        depth_map = self.depth_estimator.predict(image)
        span = depth_span(depth_map, detection.mask)

        measurement = {
            "length_px": length_px,
            "width_px": width_px,
            "depth_span": span,
        }
        self._measurement_cache[sample.photo_id] = measurement
        return measurement

    # -- tahap 1: kalibrasi ----------------------------------------------------

    def calibrate(self, samples: list[SeaweedSample], verbose: bool = True) -> None:
        """Menjalankan kalibrasi panjang fokus per zona dan skala ketebalan per grup."""
        focal_candidates: dict[str, list[float]] = {}
        scale_candidates: dict[int, list[float]] = {}

        for index, sample in enumerate(samples, start=1):
            measurement = self._measure(sample)
            if measurement is None:
                continue

            # 1a. Panjang fokus dari dimensi terpanjang -- Persamaan (3.12).
            if sample.gt_length_cm > 0:
                focal = (
                    measurement["length_px"] * sample.distance_cm / sample.gt_length_cm
                )
                focal_candidates.setdefault(sample.zone, []).append(focal)

            # 1b. Faktor skala ketebalan -- Persamaan (3.15).
            if measurement["depth_span"] > 1e-9:
                scale = sample.gt_thickness_cm / measurement["depth_span"]
                scale_candidates.setdefault(sample.group_key, []).append(scale)

            if verbose and index % 50 == 0:
                print(f"  kalibrasi: {index}/{len(samples)} foto diproses")

        self.focal_per_zone = {
            zone: float(np.median(values)) for zone, values in focal_candidates.items()
        }
        self.thickness_scale_per_group = {
            group: float(np.median(values))
            for group, values in scale_candidates.items()
        }

        if verbose:
            print("\nPanjang fokus hasil kalibrasi (piksel):")
            for zone, focal in sorted(self.focal_per_zone.items()):
                print(f"  {zone}: {focal:.2f}")
            print(f"Jumlah grup terkalibrasi: {len(self.thickness_scale_per_group)}")

    # -- tahap 2: estimasi volume ---------------------------------------------

    def estimate_one(self, sample: SeaweedSample) -> dict[str, float | str] | None:
        """Mengestimasi volume satu foto sampel."""
        if not self.focal_per_zone:
            raise RuntimeError(
                "Kalibrasi belum dijalankan. Panggil `calibrate` terlebih dahulu."
            )

        focal = self.focal_per_zone.get(sample.zone)
        scale = self.thickness_scale_per_group.get(sample.group_key)
        if focal is None or scale is None:
            return None

        measurement = self._measure(sample)
        if measurement is None:
            return None

        # Konversi piksel ke sentimeter -- Persamaan (3.11).
        length_cm = measurement["length_px"] * sample.distance_cm / focal
        width_cm = measurement["width_px"] * sample.distance_cm / focal
        thickness_cm = measurement["depth_span"] * scale

        volume = length_cm * width_cm * thickness_cm   # Persamaan (3.17)

        return {
            "photo_id": sample.photo_id,
            "zone": sample.zone,
            "distance_cm": float(sample.distance_cm),
            "length_cm": length_cm,
            "width_cm": width_cm,
            "thickness_cm": thickness_cm,
            "volume_pred_cm3": volume,
            "volume_gt_cm3": sample.gt_volume_cm3,
        }

    def estimate(
        self, samples: list[SeaweedSample], verbose: bool = True
    ) -> list[dict[str, float | str]]:
        """Mengestimasi volume seluruh foto sampel."""
        results = []
        for index, sample in enumerate(samples, start=1):
            result = self.estimate_one(sample)
            if result is not None:
                results.append(result)
            if verbose and index % 50 == 0:
                print(f"  estimasi: {index}/{len(samples)} foto diproses")
        return results


def load_samples_from_csv(csv_path: str | Path) -> list[SeaweedSample]:
    """Memuat daftar sampel dari berkas CSV.

    Kolom yang diperlukan: photo_id, image_path, zone, photo_number,
    gt_length_cm, gt_width_cm, gt_thickness_cm.
    """
    import csv

    samples: list[SeaweedSample] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            samples.append(
                SeaweedSample(
                    photo_id=row["photo_id"],
                    image_path=row["image_path"],
                    zone=row["zone"],
                    photo_number=int(row["photo_number"]),
                    gt_length_cm=float(row["gt_length_cm"]),
                    gt_width_cm=float(row["gt_width_cm"]),
                    gt_thickness_cm=float(row["gt_thickness_cm"]),
                )
            )
    return samples
