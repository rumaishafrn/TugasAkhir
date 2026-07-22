"""
Uji unit modul inti pipeline.

Menjalankan uji:
    pytest tests/ -v

Uji pada berkas ini sengaja dibatasi pada komponen yang tidak memerlukan
bobot model maupun GPU, sehingga dapat dijalankan pada mesin mana pun.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.distance import (
    estimate_distance,
    fit_distance_model,
    pixel_to_cm,
)
from src.common.config import CORRECTION_FACTOR, DIST_A, DIST_B
from src.common.metrics import box_volume, cylinder_volume, evaluate_volume
from src.waste.convex_hull import convex_hull_volume, corrected_volume
from src.waste.pointcloud import unproject_mask_to_points
from src.waste.tracking import HungarianIoUTracker, compute_iou, trimmed_mean


# --------------------------------------------------------------------------
# Kalibrasi jarak
# --------------------------------------------------------------------------

def test_konstanta_kalibrasi_sesuai_buku():
    assert DIST_A == pytest.approx(47410.0)
    assert DIST_B == pytest.approx(16.159)


def test_jarak_menurun_saat_objek_tampak_lebih_besar():
    """Objek yang tampak lebih besar berarti lebih dekat ke kamera."""
    assert estimate_distance(2000) < estimate_distance(1400)


def test_regresi_mereproduksi_konstanta_buku():
    """Data kalibrasi Lampiran 3 harus menghasilkan a = 47410 dan b = 16,159."""
    s = np.array([1415, 1615, 1782, 1462, 1527, 1587, 1559, 1643,
                  1740, 1787, 1842, 1972, 1968, 2033, 1977, 1462], dtype=float)
    d = np.array([50, 45, 40, 49, 48, 47, 46, 44,
                  43, 42, 41, 38, 49, 38, 39, 49], dtype=float)

    a, b = fit_distance_model(s, d)
    assert a == pytest.approx(47410.0, rel=1e-3)
    assert b == pytest.approx(16.159, rel=1e-3)


def test_konversi_piksel_ke_cm():
    assert pixel_to_cm(1272.0, 100.0, 1272.0) == pytest.approx(100.0)


def test_jarak_menolak_masukan_tidak_valid():
    with pytest.raises(ValueError):
        estimate_distance(0)


# --------------------------------------------------------------------------
# Convex hull
# --------------------------------------------------------------------------

def test_volume_hull_kubus():
    """Kubus bersisi 10 cm harus menghasilkan volume hull tepat 1000 cm3."""
    cube = np.array(
        [[x, y, z] for x in (0, 10) for y in (0, 10) for z in (0, 10)], dtype=float
    )
    assert convex_hull_volume(cube) == pytest.approx(1000.0)


def test_faktor_koreksi_diterapkan_sebagai_pembagi():
    cube = np.array(
        [[x, y, z] for x in (0, 10) for y in (0, 10) for z in (0, 10)], dtype=float
    )
    assert corrected_volume(cube) == pytest.approx(1000.0 / CORRECTION_FACTOR)


def test_hull_titik_terlalu_sedikit():
    assert convex_hull_volume(np.zeros((3, 3))) == 0.0


def test_hull_titik_koplanar_tidak_menimbulkan_galat():
    """Titik koplanar tidak memiliki volume dan harus ditangani dengan aman."""
    planar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    assert convex_hull_volume(planar) == 0.0


# --------------------------------------------------------------------------
# Proyeksi balik
# --------------------------------------------------------------------------

def test_proyeksi_balik_titik_pusat():
    """Piksel pada titik utama harus terproyeksi ke sumbu optik (X = Y = 0)."""
    depth = np.full((720, 1280), 50.0, dtype=np.float32)
    mask = np.zeros((720, 1280), dtype=bool)
    mask[360, 640] = True

    points = unproject_mask_to_points(depth, mask)
    assert points.shape == (1, 3)
    assert points[0, 0] == pytest.approx(0.0)
    assert points[0, 1] == pytest.approx(0.0)
    assert points[0, 2] == pytest.approx(50.0)


def test_proyeksi_balik_mask_kosong():
    depth = np.ones((100, 100), dtype=np.float32)
    mask = np.zeros((100, 100), dtype=bool)
    assert unproject_mask_to_points(depth, mask).shape == (0, 3)


# --------------------------------------------------------------------------
# Pelacakan
# --------------------------------------------------------------------------

def test_iou_kotak_identik():
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_kotak_terpisah():
    assert compute_iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_iou_tumpang_tindih_separuh():
    # Irisan 5x10 = 50, gabungan 100 + 100 - 50 = 150
    assert compute_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_trimmed_mean_meredam_pencilan():
    nilai = [100.0, 101.0, 99.0, 100.0, 5000.0]
    assert trimmed_mean(nilai) < 200.0


def test_tracker_mempertahankan_identitas_objek():
    """Objek yang bergeser sedikit antar frame harus tetap satu track."""
    tracker = HungarianIoUTracker()
    tracker.update([((0, 0, 100, 100), "can", 300.0)])
    tracker.update([((5, 5, 105, 105), "can", 305.0)])
    tracker.update([((10, 10, 110, 110), "can", 298.0)])

    assert len(tracker.tracks) == 1
    assert len(tracker.tracks[0].volumes) == 3


def test_tracker_memisahkan_objek_berjauhan():
    tracker = HungarianIoUTracker()
    tracker.update([((0, 0, 50, 50), "can", 300.0)])
    tracker.update([((500, 500, 550, 550), "can", 310.0)])
    assert len(tracker.tracks) == 2


# --------------------------------------------------------------------------
# Metrik evaluasi
# --------------------------------------------------------------------------

def test_estimasi_sempurna_menghasilkan_akurasi_penuh():
    metrics = evaluate_volume([100.0, 200.0], [100.0, 200.0])
    assert metrics.mape == pytest.approx(0.0)
    assert metrics.accuracy == pytest.approx(100.0)


def test_mape_dihitung_dengan_benar():
    # Galat 10% pada kedua data
    metrics = evaluate_volume([110.0, 90.0], [100.0, 100.0])
    assert metrics.mape == pytest.approx(10.0)
    assert metrics.mae == pytest.approx(10.0)


def test_metrik_menolak_panjang_larik_berbeda():
    with pytest.raises(ValueError):
        evaluate_volume([1.0, 2.0], [1.0])


def test_metrik_menolak_ground_truth_nol():
    with pytest.raises(ValueError):
        evaluate_volume([1.0], [0.0])


# --------------------------------------------------------------------------
# Volume ground truth geometri
# --------------------------------------------------------------------------

def test_volume_silinder_tutup_botol():
    """Tutup Botol Aqua: d = 3,0 cm; t = 1,5 cm -> 10,60 cm3 (Tabel 3.1)."""
    assert cylinder_volume(3.0, 1.5) == pytest.approx(10.60, abs=0.01)


def test_volume_balok_chitato():
    """Chitato Lite Hangout Pack: 30 x 20 x 7 -> 4200,00 cm3 (Tabel 3.1)."""
    assert box_volume(30.0, 20.0, 7.0) == pytest.approx(4200.0)
