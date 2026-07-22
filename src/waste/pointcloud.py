"""
Rekonstruksi dan pembersihan point cloud tiga dimensi.

Tahap ini mencakup dua langkah pada jalur sampah terapung:

    1. Penskalaan peta kedalaman relatif menjadi kedalaman metrik dengan
       jarak hasil kalibrasi sebagai jangkar (Persamaan 3.4 sampai 3.6).
    2. Proyeksi balik piksel mask ke ruang tiga dimensi menggunakan model
       kamera pinhole (Persamaan 3.7), diikuti pembersihan pencilan dengan
       Statistical Outlier Removal (Persamaan 3.8).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.common.config import (
    CX,
    CY,
    FX,
    FY,
    SOR_NB_NEIGHBORS,
    SOR_STD_RATIO,
)

# Open3D dipakai bila tersedia agar identik dengan implementasi pada
# penelitian. Bila pustaka tersebut tidak terpasang, digunakan implementasi
# setara berbasis scipy.spatial.cKDTree yang menghasilkan nilai sama.
try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:  # pragma: no cover
    _HAS_OPEN3D = False


def scale_depth_to_metric(
    relative_depth: np.ndarray,
    mask: np.ndarray,
    anchor_distance_cm: float,
) -> np.ndarray:
    """Mengubah peta kedalaman relatif menjadi kedalaman metrik.

    Rata-rata kedalaman relatif pada area mask dipetakan tepat ke jarak
    jangkar hasil Persamaan (3.2), sedangkan variasi kedalaman antar-piksel
    di dalam objek tetap dipertahankan secara proporsional.

        d_rel_bar = (1/N) * SUM d_rel_i                (Persamaan 3.4)
        k         = D_nyata / d_rel_bar                (Persamaan 3.5)
        D_skala   = d_rel(x, y) * k                    (Persamaan 3.6)

    Parameters
    ----------
    relative_depth
        Peta kedalaman keluaran model MDE, bentuk (H, W).
    mask
        Mask biner objek, bentuk (H, W).
    anchor_distance_cm
        Jarak kamera-objek hasil kalibrasi, satuan sentimeter.

    Returns
    -------
    np.ndarray
        Peta kedalaman metrik dalam satuan sentimeter, bentuk (H, W).
    """
    masked_values = relative_depth[mask]
    if masked_values.size == 0:
        raise ValueError("Mask tidak memuat piksel apa pun.")

    mean_relative = float(masked_values.mean())
    if abs(mean_relative) < 1e-9:
        raise ValueError("Rata-rata kedalaman relatif mendekati nol.")

    scale_constant = anchor_distance_cm / mean_relative
    return relative_depth * scale_constant


def unproject_mask_to_points(
    metric_depth: np.ndarray,
    mask: np.ndarray,
    fx: float = FX,
    fy: float = FY,
    cx: float = CX,
    cy: float = CY,
) -> np.ndarray:
    """Memproyeksikan balik piksel mask ke ruang tiga dimensi.

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy          (Persamaan 3.7)
        Z = kedalaman metrik piksel

    Returns
    -------
    np.ndarray
        Larik titik berbentuk (N, 3) dengan satuan sentimeter.
    """
    v_indices, u_indices = np.nonzero(mask)
    if v_indices.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = metric_depth[v_indices, u_indices].astype(np.float64)

    # Buang piksel dengan kedalaman tidak valid.
    valid = np.isfinite(z) & (z > 0)
    u_indices, v_indices, z = u_indices[valid], v_indices[valid], z[valid]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    x = (u_indices - cx) * z / fx
    y = (v_indices - cy) * z / fy

    return np.stack([x, y, z], axis=1)


def remove_statistical_outliers(
    points: np.ndarray,
    nb_neighbors: int = SOR_NB_NEIGHBORS,
    std_ratio: float = SOR_STD_RATIO,
) -> np.ndarray:
    """Membersihkan pencilan point cloud dengan Statistical Outlier Removal.

    Untuk setiap titik dihitung jarak rata-rata terhadap k tetangga
    terdekatnya. Titik yang memenuhi d_i > mu + alpha * sigma dianggap
    pencilan dan dibuang (Persamaan 3.8), dengan k = 20 dan alpha = 1,5.

    Returns
    -------
    np.ndarray
        Point cloud yang telah dibersihkan, bentuk (M, 3) dengan M <= N.
    """
    if points.shape[0] <= nb_neighbors:
        # Titik terlalu sedikit untuk analisis ketetanggaan yang bermakna.
        return points

    if _HAS_OPEN3D:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        filtered, _ = cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return np.asarray(filtered.points)

    return _sor_scipy(points, nb_neighbors, std_ratio)


def _sor_scipy(
    points: np.ndarray, nb_neighbors: int, std_ratio: float
) -> np.ndarray:
    """Implementasi SOR berbasis KD-Tree sebagai pengganti Open3D.

    Menghitung jarak rata-rata tiap titik terhadap k tetangga terdekatnya,
    lalu membuang titik yang memenuhi d_i > mu + alpha * sigma. Titik itu
    sendiri dikecualikan dari perhitungan jarak, mengikuti konvensi Open3D.
    """
    tree = cKDTree(points)
    # k+1 karena tetangga terdekat pertama adalah titik itu sendiri.
    distances, _ = tree.query(points, k=nb_neighbors + 1)
    mean_distances = distances[:, 1:].mean(axis=1)

    mu = float(mean_distances.mean())
    sigma = float(mean_distances.std())
    threshold = mu + std_ratio * sigma

    return points[mean_distances <= threshold]


def build_clean_pointcloud(
    relative_depth: np.ndarray,
    mask: np.ndarray,
    anchor_distance_cm: float,
) -> np.ndarray:
    """Menjalankan seluruh tahap rekonstruksi dalam satu pemanggilan.

    Urutan: penskalaan metrik -> proyeksi balik -> pembersihan pencilan.
    """
    metric_depth = scale_depth_to_metric(relative_depth, mask, anchor_distance_cm)
    points = unproject_mask_to_points(metric_depth, mask)
    return remove_statistical_outliers(points)
