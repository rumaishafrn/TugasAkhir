"""
Kalibrasi jarak kamera-objek untuk studi kasus sampah terapung.

Pada model kamera pinhole, ukuran proyeksi objek pada bidang citra berbanding
terbalik terhadap jaraknya. Jarak karenanya dimodelkan sebagai fungsi invers
terhadap ukuran tampak objek (Subbab 3.6.1 dan Lampiran 3):

    D = a / s + b        dengan  s = w_obb + h_obb   (Persamaan 3.2)

Suku a/s menangkap hubungan proyektif terbalik tersebut, sedangkan konstanta
aditif b mengoreksi penyimpangan pada jarak dekat yang muncul akibat selisih
antara ukuran oriented bounding box dan ukuran proyeksi ideal.

Konstanta hasil kalibrasi pada penelitian ini adalah a = 47410 dan b = 16,159.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from src.common.config import DIST_A, DIST_B


def inverse_distance_model(s: np.ndarray, a: float, b: float) -> np.ndarray:
    """Model D = a/s + b."""
    return a / s + b


def estimate_distance(obb_size_sum: float) -> float:
    """Mengestimasi jarak kamera-objek dari ukuran tampak objek.

    Parameters
    ----------
    obb_size_sum
        Nilai s = w_obb + h_obb dalam satuan piksel.

    Returns
    -------
    float
        Jarak kamera terhadap objek dalam satuan sentimeter.
    """
    if obb_size_sum <= 0:
        raise ValueError("Ukuran OBB harus bernilai positif.")
    return DIST_A / obb_size_sum + DIST_B


def fit_distance_model(
    s_values: np.ndarray | list[float],
    distances_cm: np.ndarray | list[float],
) -> tuple[float, float]:
    """Mencocokkan konstanta a dan b terhadap data kalibrasi.

    Digunakan bila penelitian direplikasi dengan kamera lain. Pencocokan
    dilakukan dengan regresi non-linear kuadrat terkecil melalui `curve_fit`,
    yaitu meminimalkan SUM (D_i - (a/s_i + b))^2.

    Returns
    -------
    tuple[float, float]
        Pasangan konstanta (a, b) hasil regresi.
    """
    s = np.asarray(s_values, dtype=np.float64).ravel()
    d = np.asarray(distances_cm, dtype=np.float64).ravel()

    if s.size != d.size:
        raise ValueError("Jumlah nilai s dan D harus sama.")
    if s.size < 2:
        raise ValueError("Dibutuhkan minimal dua titik data untuk regresi.")

    popt, _ = curve_fit(
        inverse_distance_model, s, d, p0=[DIST_A, DIST_B], maxfev=10000
    )
    return float(popt[0]), float(popt[1])


def pixel_to_cm(pixel_size: float, distance_cm: float, focal_px: float) -> float:
    """Konversi dimensi piksel ke sentimeter -- Persamaan (3.3) dan (3.11).

        ukuran_cm = ukuran_px * jarak_cm / f_px
    """
    if focal_px <= 0:
        raise ValueError("Panjang fokus harus bernilai positif.")
    return pixel_size * distance_cm / focal_px
