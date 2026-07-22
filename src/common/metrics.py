"""
Metrik evaluasi estimasi volume -- Persamaan (3.20) sampai (3.22).

    MAE     = (1/n) * SUM |V_pred_i - V_gt_i|
    MAPE    = (100%/n) * SUM |(V_pred_i - V_gt_i) / V_gt_i|
    Akurasi = 100% - MAPE

Simpangan baku MAE dan MAPE turut dilaporkan untuk menggambarkan konsistensi
estimasi antar objek. Perlu dicatat bahwa pada studi kasus sampah terapung
nilai simpangan baku MAE dapat melampaui nilai rata-ratanya. Hal tersebut
merupakan konsekuensi wajar dari distribusi galat yang menjulur ke kanan
akibat rentang volume ground truth yang sangat lebar, yaitu dari 10,60 cm3
(tutup botol) hingga 4200,00 cm3 (wadah plastik).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class VolumeMetrics:
    """Ringkasan metrik evaluasi estimasi volume."""

    n: int
    mean_gt: float
    mean_pred: float
    mae: float
    mae_std: float
    mape: float
    mape_std: float
    accuracy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"n = {self.n}\n"
            f"Rata-rata GT      : {self.mean_gt:.2f} cm3\n"
            f"Rata-rata prediksi: {self.mean_pred:.2f} cm3\n"
            f"MAE               : {self.mae:.2f} +/- {self.mae_std:.2f} cm3\n"
            f"MAPE              : {self.mape:.2f} +/- {self.mape_std:.2f} %\n"
            f"Akurasi           : {self.accuracy:.2f} %"
        )


def evaluate_volume(
    predicted: np.ndarray | list[float],
    ground_truth: np.ndarray | list[float],
) -> VolumeMetrics:
    """Menghitung seluruh metrik evaluasi volume.

    Parameters
    ----------
    predicted
        Volume hasil estimasi pipeline, satuan cm3.
    ground_truth
        Volume acuan hasil pengukuran fisik, satuan cm3.

    Raises
    ------
    ValueError
        Bila panjang kedua larik tidak sama, larik kosong, atau terdapat
        nilai ground truth bernilai nol (MAPE tidak terdefinisi).
    """
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    gt = np.asarray(ground_truth, dtype=np.float64).ravel()

    if pred.size != gt.size:
        raise ValueError(
            f"Jumlah data tidak sama: prediksi {pred.size}, ground truth {gt.size}"
        )
    if pred.size == 0:
        raise ValueError("Larik masukan kosong.")
    if np.any(gt == 0):
        raise ValueError("Terdapat ground truth bernilai nol, MAPE tidak terdefinisi.")

    absolute_error = np.abs(pred - gt)
    percentage_error = np.abs((pred - gt) / gt) * 100.0

    mape = float(percentage_error.mean())

    return VolumeMetrics(
        n=int(pred.size),
        mean_gt=float(gt.mean()),
        mean_pred=float(pred.mean()),
        mae=float(absolute_error.mean()),
        mae_std=float(absolute_error.std(ddof=1)) if pred.size > 1 else 0.0,
        mape=mape,
        mape_std=float(percentage_error.std(ddof=1)) if pred.size > 1 else 0.0,
        accuracy=100.0 - mape,
    )


def cylinder_volume(diameter_cm: float, height_cm: float) -> float:
    """Volume model geometri silinder -- Persamaan (3.18)."""
    return float(np.pi * (diameter_cm / 2.0) ** 2 * height_cm)


def box_volume(length_cm: float, width_cm: float, height_cm: float) -> float:
    """Volume model geometri balok -- Persamaan (3.19)."""
    return float(length_cm * width_cm * height_cm)
