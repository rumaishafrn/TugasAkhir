"""
Pemuat terpadu untuk enam model Monocular Depth Estimation (MDE).

Keenam model dievaluasi dalam penelitian ini menggunakan bobot pre-trained
tanpa fine-tuning (Tabel 3.5). Modul ini menyediakan antarmuka tunggal
`DepthEstimator` sehingga keenam model dapat dipertukarkan tanpa mengubah
kode pipeline.

Perbedaan penting antara kedua jenis keluaran:

    - Model METRIK  (ZoeDepth) mengeluarkan nilai kedalaman dalam satuan
      meter secara langsung.
    - Model RELATIF (Depth-Anything V2, MiDaS, DPT-Large) mengeluarkan
      inverse relative depth yang tidak memiliki satuan, sehingga perlu
      dikalibrasi terlebih dahulu (Persamaan 3.5 dan 3.6).

Pipeline pada penelitian ini melakukan penskalaan berbasis jangkar jarak
untuk KEDUA jenis keluaran, sehingga keduanya diperlakukan seragam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DepthType = Literal["relative", "metric"]


@dataclass(frozen=True)
class MDEModelSpec:
    """Spesifikasi satu model MDE."""

    key: str
    hf_id: str
    depth_type: DepthType
    display_name: str


# Tabel 3.5 -- Enam model MDE yang dievaluasi
MDE_REGISTRY: dict[str, MDEModelSpec] = {
    "dav2-base": MDEModelSpec(
        key="dav2-base",
        hf_id="depth-anything/Depth-Anything-V2-Base-hf",
        depth_type="relative",
        display_name="Depth-Anything V2 Base",
    ),
    "dav2-large": MDEModelSpec(
        key="dav2-large",
        hf_id="depth-anything/Depth-Anything-V2-Large-hf",
        depth_type="relative",
        display_name="Depth-Anything V2 Large",
    ),
    "midas-beit-l512": MDEModelSpec(
        key="midas-beit-l512",
        hf_id="Intel/dpt-beit-large-512",
        depth_type="relative",
        display_name="MiDaS v3.1 BEiT-L-512",
    ),
    "zoedepth-kitti": MDEModelSpec(
        key="zoedepth-kitti",
        hf_id="Intel/zoedepth-kitti",
        depth_type="metric",
        display_name="ZoeDepth (KITTI)",
    ),
    "zoedepth-nyu": MDEModelSpec(
        key="zoedepth-nyu",
        hf_id="Intel/zoedepth-nyu",
        depth_type="metric",
        display_name="ZoeDepth (NYUv2)",
    ),
    "dpt-large": MDEModelSpec(
        key="dpt-large",
        hf_id="Intel/dpt-large",
        depth_type="relative",
        display_name="DPT-Large (Intel)",
    ),
}


class DepthEstimator:
    """Pembungkus model MDE HuggingFace dengan antarmuka seragam.

    Contoh
    ------
    >>> estimator = DepthEstimator("zoedepth-kitti")
    >>> depth = estimator.predict(frame_bgr)   # ndarray float32, HxW
    """

    def __init__(self, model_key: str, device: str | None = None) -> None:
        if model_key not in MDE_REGISTRY:
            raise KeyError(
                f"Model '{model_key}' tidak dikenal. "
                f"Pilihan yang tersedia: {sorted(MDE_REGISTRY)}"
            )

        self.spec = MDE_REGISTRY[model_key]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(self.spec.hf_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.spec.hf_id)
        self.model.to(self.device).eval()

    @property
    def depth_type(self) -> DepthType:
        return self.spec.depth_type

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Menghasilkan peta kedalaman untuk satu citra.

        Parameters
        ----------
        image_bgr
            Citra BGR (konvensi OpenCV) dengan bentuk (H, W, 3), dtype uint8.

        Returns
        -------
        np.ndarray
            Peta kedalaman float32 berbentuk (H, W) dengan resolusi identik
            terhadap citra masukan.
        """
        image_rgb = image_bgr[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)
        original_size = (pil_image.height, pil_image.width)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        predicted = outputs.predicted_depth

        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)

        resized = torch.nn.functional.interpolate(
            predicted,
            size=original_size,
            mode="bicubic",
            align_corners=False,
        )
        return resized.squeeze().cpu().numpy().astype(np.float32)


def list_models() -> list[str]:
    """Mengembalikan seluruh kunci model MDE yang terdaftar."""
    return sorted(MDE_REGISTRY)
