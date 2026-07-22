"""
Pelacakan objek antar-frame dan penstabilan estimasi volume.

Masukan studi kasus sampah terapung berupa video, sehingga objek yang sama
harus dikenali sebagai satu entitas di sepanjang frame. Asosiasi dilakukan
dengan algoritma penugasan Hungarian berbasis kemiripan Intersection over
Union (IoU). Volume tiap track kemudian distabilkan menggunakan rata-rata
terpangkas atau trimmed mean pada jendela lima frame untuk meredam fluktuasi
sesaat akibat oklusi maupun kesalahan deteksi (Subbab 3.6.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.config import (
    IOU_MATCH_THRESHOLD,
    MAX_FRAMES_MISSING,
    TRIMMED_MEAN_WINDOW,
    TRIMMED_PROPORTION,
)

BBox = tuple[int, int, int, int]


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Menghitung Intersection over Union dua kotak pembatas aksis-sejajar."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def trimmed_mean(
    values: list[float] | np.ndarray,
    proportion: float = TRIMMED_PROPORTION,
) -> float:
    """Menghitung rata-rata terpangkas dengan membuang nilai ekstrem.

    Sebagian nilai terkecil dan terbesar dibuang sebelum rata-rata dihitung,
    sehingga hasilnya lebih tahan terhadap frame yang mengalami oklusi atau
    kesalahan segmentasi.
    """
    array = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if array.size == 0:
        return 0.0
    if array.size <= 2:
        return float(array.mean())

    cut = int(np.floor(array.size * proportion))
    trimmed = array[cut : array.size - cut] if cut > 0 else array
    if trimmed.size == 0:
        trimmed = array

    return float(trimmed.mean())


@dataclass
class Track:
    """Riwayat satu objek di sepanjang frame video."""

    track_id: int
    class_name: str
    bbox: BBox
    volumes: list[float] = field(default_factory=list)
    frames_missing: int = 0

    def update(self, bbox: BBox, volume: float) -> None:
        self.bbox = bbox
        self.volumes.append(volume)
        self.frames_missing = 0

    def smoothed_volume(self, window: int = TRIMMED_MEAN_WINDOW) -> float:
        """Volume akhir track sebagai rata-rata terpangkas jendela terakhir."""
        if not self.volumes:
            return 0.0
        return trimmed_mean(self.volumes[-window:])

    def final_volume(self) -> float:
        """Volume akhir track dari seluruh riwayat estimasi."""
        return trimmed_mean(self.volumes) if self.volumes else 0.0


class HungarianIoUTracker:
    """Pelacak multi-objek berbasis penugasan Hungarian dan kemiripan IoU."""

    def __init__(
        self,
        iou_threshold: float = IOU_MATCH_THRESHOLD,
        max_missing: int = MAX_FRAMES_MISSING,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.tracks: list[Track] = []
        self._next_id = 0

    def update(
        self,
        detections: list[tuple[BBox, str, float]],
    ) -> None:
        """Memperbarui seluruh track dengan deteksi pada satu frame.

        Parameters
        ----------
        detections
            Daftar tuple (bbox, class_name, volume) untuk frame saat ini.
        """
        if not self.tracks:
            for bbox, class_name, volume in detections:
                self._create_track(bbox, class_name, volume)
            return

        if not detections:
            self._age_tracks(matched_indices=set())
            return

        # Matriks biaya = 1 - IoU, sehingga meminimalkan biaya sama dengan
        # memaksimalkan kemiripan IoU.
        cost = np.ones((len(self.tracks), len(detections)), dtype=np.float64)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, (bbox, _, _) in enumerate(detections):
                cost[t_idx, d_idx] = 1.0 - compute_iou(track.bbox, bbox)

        track_indices, detection_indices = linear_sum_assignment(cost)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        for t_idx, d_idx in zip(track_indices, detection_indices):
            iou = 1.0 - cost[t_idx, d_idx]
            if iou < self.iou_threshold:
                continue
            bbox, class_name, volume = detections[d_idx]
            if self.tracks[t_idx].class_name != class_name:
                continue
            self.tracks[t_idx].update(bbox, volume)
            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)

        for d_idx, (bbox, class_name, volume) in enumerate(detections):
            if d_idx not in matched_detections:
                self._create_track(bbox, class_name, volume)

        self._age_tracks(matched_tracks)

    def _create_track(self, bbox: BBox, class_name: str, volume: float) -> None:
        track = Track(track_id=self._next_id, class_name=class_name, bbox=bbox)
        track.volumes.append(volume)
        self.tracks.append(track)
        self._next_id += 1

    def _age_tracks(self, matched_indices: set[int]) -> None:
        for idx, track in enumerate(self.tracks):
            if idx not in matched_indices:
                track.frames_missing += 1
        self.tracks = [t for t in self.tracks if t.frames_missing <= self.max_missing]

    def results(self) -> dict[int, dict[str, float | str | int]]:
        """Mengembalikan volume akhir tiap track setelah seluruh frame diproses."""
        return {
            track.track_id: {
                "class_name": track.class_name,
                "volume_cm3": track.final_volume(),
                "n_frames": len(track.volumes),
            }
            for track in self.tracks
        }
