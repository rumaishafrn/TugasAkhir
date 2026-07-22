"""
Konstanta global pipeline estimasi volumetrik objek dari kamera RGB.

Seluruh nilai di berkas ini bersumber langsung dari Bab III Buku Tugas Akhir
"Estimasi Volumetrik Objek dari Kamera RGB Menggunakan YOLO Instance
Segmentation dan Monocular Depth Estimation" (Rumaisha Afrina, 5025221146,
Teknik Informatika ITS, 2026).

JANGAN mengubah nilai-nilai di bawah tanpa melakukan kalibrasi ulang.
Konstanta kalibrasi bersifat spesifik terhadap perangkat akuisisi yang
digunakan pada penelitian ini.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. PARAMETER INTRINSIK KAMERA  --  Persamaan (3.1)
# ---------------------------------------------------------------------------
# Diturunkan dari FOV diagonal 60 derajat milik webcam Logitech C310 pada
# resolusi 1280 x 720:
#     d_diag = sqrt(1280^2 + 720^2) ~= 1468,6 piksel
#     f      = (d_diag / 2) / tan(60 / 2) ~= 1271,8 -> dibulatkan 1272 piksel
# Titik utama diasumsikan berada tepat di pusat citra.
FX: float = 1272.0
FY: float = 1272.0
CX: float = 640.0
CY: float = 360.0

IMAGE_WIDTH: int = 1280
IMAGE_HEIGHT: int = 720

# ---------------------------------------------------------------------------
# 2. KALIBRASI JARAK KAMERA-OBJEK  --  Persamaan (3.2)
# ---------------------------------------------------------------------------
# Model:  D = DIST_A / s + DIST_B ,  dengan  s = w_obb + h_obb  (piksel)
# Hasil regresi non-linear least squares (scipy.optimize.curve_fit) terhadap
# 16 pasangan data kalibrasi menggunakan botol plastik referensi 600 ml
# (lebar nyata 6,4 cm; tinggi nyata 23,3 cm). Lihat data/calibration_distance.csv
DIST_A: float = 47410.0
DIST_B: float = 16.159

# ---------------------------------------------------------------------------
# 3. STATISTICAL OUTLIER REMOVAL  --  Persamaan (3.8)
# ---------------------------------------------------------------------------
SOR_NB_NEIGHBORS: int = 20     # k = jumlah tetangga terdekat yang ditinjau
SOR_STD_RATIO: float = 1.5     # alpha = faktor pengali simpangan baku

# ---------------------------------------------------------------------------
# 4. FAKTOR KOREKSI CONVEX HULL  --  Persamaan (3.10)
# ---------------------------------------------------------------------------
# Convex hull cenderung melebih-estimasi volume karena membungkus seluruh
# titik terluar secara konveks. Faktor 1,58 merupakan hasil regresi empiris
# terhadap volume ground truth dan diterapkan sebagai PEMBAGI.
#     V_terkoreksi = V_hull / 1,58
CORRECTION_FACTOR: float = 1.58

# ---------------------------------------------------------------------------
# 5. PELACAKAN ANTAR-FRAME (studi kasus sampah terapung)
# ---------------------------------------------------------------------------
IOU_MATCH_THRESHOLD: float = 0.3   # ambang minimum IoU untuk asosiasi Hungarian
MAX_FRAMES_MISSING: int = 10       # track dihapus bila hilang lebih dari N frame
TRIMMED_MEAN_WINDOW: int = 5       # jendela rata-rata terpangkas
TRIMMED_PROPORTION: float = 0.2    # proporsi dipangkas di tiap ujung

# ---------------------------------------------------------------------------
# 6. AMBANG KEPERCAYAAN INFERENSI YOLO  --  Tabel 3.4
# ---------------------------------------------------------------------------
CONF_THRESHOLD_WASTE: float = 0.50
CONF_THRESHOLD_SEAWEED: float = 0.25
YOLO_IMGSZ: int = 640

# ---------------------------------------------------------------------------
# 7. STUDI KASUS RUMPUT LAUT  --  Subbab 3.7
# ---------------------------------------------------------------------------
# Persentil untuk perhitungan rentang kedalaman -- Persamaan (3.14)
#     s = P95(d_rel) - P5(d_rel)
DEPTH_PERCENTILE_HIGH: int = 95
DEPTH_PERCENTILE_LOW: int = 5

# Satu grup = satu sampel = enam foto. group_key = (nomor_foto - 1) // 6
PHOTOS_PER_GROUP: int = 6

# Pemetaan posisi foto dalam grup terhadap jarak akuisisi -- Tabel 3.2
PHOTO_DISTANCE_MAP: dict[int, int] = {
    1: 30, 2: 30,
    3: 40, 4: 40,
    5: 50, 6: 50,
}

# ---------------------------------------------------------------------------
# 8. MODEL TERPILIH  --  Bab IV
# ---------------------------------------------------------------------------
BEST_YOLO_WASTE: str = "yolov8s-seg"        # mask mAP@50-95 = 0,926
BEST_YOLO_SEAWEED: str = "yolo11s-seg"      # mask mAP@50-95 = 0,8786
BEST_MDE_WASTE: str = "zoedepth-kitti"      # akurasi 97,08% (MAPE 2,92%)
BEST_MDE_SEAWEED: str = "dav2-large"        # akurasi 96,95% (MAPE 3,05%)

# ---------------------------------------------------------------------------
# 9. KELAS OBJEK
# ---------------------------------------------------------------------------
WASTE_CLASSES: tuple[str, ...] = (
    "bottle cap",
    "can",
    "cardboard",
    "plastic bottle",
    "plastic container",
    "plastic lid",
)
SEAWEED_CLASSES: tuple[str, ...] = ("seaweed",)
