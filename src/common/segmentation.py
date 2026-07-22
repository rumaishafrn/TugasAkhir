"""
Pembungkus model YOLO instance segmentation (Ultralytics).

Modul ini menyediakan dua hal yang dibutuhkan seluruh jalur pipeline:

    1. Ekstraksi mask biner pada resolusi citra asli.
    2. Ekstraksi oriented bounding box (OBB) melalui `cv2.minAreaRect`
       terhadap kontur terbesar dari mask (Subbab 3.6.1 dan 3.7.2).

Empat varian YOLO dibandingkan pada penelitian ini, yaitu YOLOv8s-seg,
YOLOv11s-seg, YOLOv12s-seg, dan YOLO26s-seg. Varian terbaik ditentukan
berdasarkan nilai mask mAP@50-95.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Satu objek hasil segmentasi pada sebuah citra."""

    mask: np.ndarray          # bool ndarray (H, W)
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 (untuk IoU tracking)

    def oriented_bbox(self) -> tuple[float, float, float]:
        """Mengembalikan (w_obb, h_obb, sudut) dari kontur terbesar mask.

        Menggunakan `cv2.minAreaRect` sesuai Subbab 3.6.1. Ukuran dikembalikan
        dalam satuan piksel.
        """
        mask_u8 = self.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0, 0.0, 0.0

        largest = max(contours, key=cv2.contourArea)
        (_, _), (w, h), angle = cv2.minAreaRect(largest)
        return float(w), float(h), float(angle)

    def obb_size_sum(self) -> float:
        """Menghitung s = w_obb + h_obb (piksel), masukan Persamaan (3.2)."""
        w, h, _ = self.oriented_bbox()
        return w + h

    def obb_length_width(self) -> tuple[float, float]:
        """Mengembalikan (panjang, lebar) piksel: sisi terpanjang dan terpendek.

        Konvensi ini digunakan pada studi kasus rumput laut (Subbab 3.7.2).
        """
        w, h, _ = self.oriented_bbox()
        return max(w, h), min(w, h)


class InstanceSegmenter:
    """Pembungkus model YOLO instance segmentation.

    Contoh
    ------
    >>> segmenter = InstanceSegmenter("weights/yolov8s-seg-waste.pt", conf=0.5)
    >>> detections = segmenter.predict(frame_bgr)
    """

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.5,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.model = YOLO(weights_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.class_names: dict[int, str] = self.model.names

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        """Menjalankan segmentasi pada satu citra dan mengembalikan deteksi."""
        height, width = image_bgr.shape[:2]

        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]

        if result.masks is None:
            return []

        detections: list[Detection] = []
        mask_data = result.masks.data.cpu().numpy()          # (N, h, w)
        boxes = result.boxes.xyxy.cpu().numpy()              # (N, 4)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for raw_mask, box, class_id, confidence in zip(
            mask_data, boxes, class_ids, confidences
        ):
            # Mask keluaran YOLO berada pada resolusi internal, sehingga perlu
            # dikembalikan ke resolusi citra asli sebelum diproyeksikan.
            resized = cv2.resize(
                raw_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
            detections.append(
                Detection(
                    mask=resized.astype(bool),
                    class_id=int(class_id),
                    class_name=self.class_names.get(int(class_id), str(class_id)),
                    confidence=float(confidence),
                    bbox=tuple(int(v) for v in box),  # type: ignore[arg-type]
                )
            )

        return detections
