"""
Pipeline estimasi volume studi kasus sampah terapung.

Implementasi ini merupakan realisasi Kode Semu 3.1 pada buku Tugas Akhir.
Alur pemrosesan per frame adalah sebagai berikut.

    1. Segmentasi objek dengan YOLO instance segmentation.
    2. Estimasi peta kedalaman dengan model MDE.
    3. Untuk tiap objek:
       a. Hitung OBB melalui minAreaRect, ambil s = w + h.
       b. Estimasi jarak D = 47410 / s + 16,159.
       c. Skalakan peta kedalaman menjadi metrik dengan jangkar D.
       d. Proyeksikan balik piksel mask menjadi point cloud tiga dimensi.
       e. Bersihkan pencilan dengan SOR (k = 20, alpha = 1,5).
       f. Hitung volume convex hull, lalu bagi dengan faktor koreksi 1,58.
    4. Asosiasikan objek ke track dengan Hungarian dan IoU.

Setelah seluruh frame diproses, volume akhir tiap track dihitung sebagai
rata-rata terpangkas pada jendela lima frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.calibration.distance import estimate_distance
from src.common.config import CONF_THRESHOLD_WASTE, YOLO_IMGSZ
from src.common.depth_models import DepthEstimator
from src.common.segmentation import InstanceSegmenter
from src.waste.convex_hull import corrected_volume
from src.waste.pointcloud import build_clean_pointcloud
from src.waste.tracking import HungarianIoUTracker


@dataclass
class FrameObservation:
    """Hasil estimasi satu objek pada satu frame."""

    frame_index: int
    class_name: str
    distance_cm: float
    n_points: int
    volume_cm3: float


class WasteVolumePipeline:
    """Pipeline estimasi volume sampah terapung dari video.

    Contoh
    ------
    >>> pipeline = WasteVolumePipeline(
    ...     yolo_weights="weights/yolov8s-seg-waste.pt",
    ...     mde_model="zoedepth-kitti",
    ... )
    >>> hasil = pipeline.process_video("data/videos/uji.mp4")
    """

    def __init__(
        self,
        yolo_weights: str,
        mde_model: str = "zoedepth-kitti",
        conf: float = CONF_THRESHOLD_WASTE,
        device: str | None = None,
    ) -> None:
        self.segmenter = InstanceSegmenter(
            weights_path=yolo_weights, conf=conf, imgsz=YOLO_IMGSZ, device=device
        )
        self.depth_estimator = DepthEstimator(mde_model, device=device)
        self.mde_model = mde_model

    # -- pemrosesan satu frame -------------------------------------------------

    def process_frame(
        self, frame_bgr: np.ndarray, frame_index: int = 0
    ) -> list[tuple[tuple[int, int, int, int], str, float, FrameObservation]]:
        """Mengestimasi volume seluruh objek pada satu frame."""
        detections = self.segmenter.predict(frame_bgr)
        if not detections:
            return []

        # Peta kedalaman dihitung satu kali per frame, lalu dipakai ulang
        # untuk seluruh objek pada frame tersebut.
        depth_map = self.depth_estimator.predict(frame_bgr)

        outputs = []
        for detection in detections:
            size_sum = detection.obb_size_sum()
            if size_sum <= 0:
                continue

            distance_cm = estimate_distance(size_sum)

            try:
                points = build_clean_pointcloud(
                    relative_depth=depth_map,
                    mask=detection.mask,
                    anchor_distance_cm=distance_cm,
                )
            except ValueError:
                continue

            volume = corrected_volume(points)
            if volume <= 0:
                continue

            observation = FrameObservation(
                frame_index=frame_index,
                class_name=detection.class_name,
                distance_cm=distance_cm,
                n_points=int(points.shape[0]),
                volume_cm3=volume,
            )
            outputs.append((detection.bbox, detection.class_name, volume, observation))

        return outputs

    # -- pemrosesan video ------------------------------------------------------

    def process_video(
        self,
        video_path: str | Path,
        frame_stride: int = 1,
        max_frames: int | None = None,
        verbose: bool = True,
    ) -> dict[str, object]:
        """Memproses seluruh video dan mengembalikan volume akhir tiap track.

        Parameters
        ----------
        video_path
            Lokasi berkas video masukan.
        frame_stride
            Memproses satu dari setiap N frame. Bernilai 1 berarti seluruh
            frame diproses.
        max_frames
            Batas jumlah frame yang diproses. Berguna untuk pengujian cepat.

        Returns
        -------
        dict
            Berisi kunci `tracks` (volume akhir tiap objek) dan
            `observations` (riwayat estimasi per frame).
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Video gagal dibuka: {video_path}")

        tracker = HungarianIoUTracker()
        observations: list[FrameObservation] = []
        frame_index = 0
        processed = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % frame_stride == 0:
                    results = self.process_frame(frame, frame_index)
                    tracker.update([(b, c, v) for b, c, v, _ in results])
                    observations.extend(obs for _, _, _, obs in results)
                    processed += 1

                    if verbose and processed % 25 == 0:
                        print(f"  frame {frame_index} diproses ({processed} total)")

                    if max_frames is not None and processed >= max_frames:
                        break

                frame_index += 1
        finally:
            capture.release()

        return {
            "mde_model": self.mde_model,
            "frames_processed": processed,
            "tracks": tracker.results(),
            "observations": observations,
        }

    # -- pemrosesan citra tunggal ---------------------------------------------

    def process_image(self, image_path: str | Path) -> list[FrameObservation]:
        """Mengestimasi volume objek pada satu citra diam."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Citra tidak terbaca: {image_path}")
        return [obs for _, _, _, obs in self.process_frame(image, frame_index=0)]
