# Panduan Penggunaan

Dokumen ini menjabarkan kelima langkah pipeline secara berurutan, lengkap dengan seluruh argumen dan contoh keluaran.

---

## Langkah 1 — Pelatihan Model YOLO Instance Segmentation

### Menyiapkan dataset

Dataset dianotasi dan diekspor melalui Roboflow dalam format **YOLOv8 Segmentation**. Struktur yang dihasilkan:

```
datasets/waste/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Isi `data.yaml` untuk dataset sampah:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 6
names: ['bottle cap', 'can', 'cardboard', 'plastic bottle', 'plastic container', 'plastic lid']
```

### Menjalankan pelatihan

```bash
python scripts/01_train_yolo.py \
    --data datasets/waste/data.yaml \
    --model yolov8s-seg.pt \
    --epochs 100 \
    --imgsz 640 \
    --name waste-yolov8s \
    --device 0
```

| Argumen | Bawaan | Keterangan |
|---|---|---|
| `--data` | wajib | Path ke `data.yaml` |
| `--model` | `yolov8s-seg.pt` | Bobot awal varian |
| `--all-variants` | mati | Melatih keempat varian berurutan |
| `--epochs` | 100 | Jumlah epoch |
| `--imgsz` | 640 | Resolusi masukan pelatihan |
| `--project` | `runs/segment` | Direktori keluaran |
| `--name` | `train` | Nama run |
| `--device` | `0` | `0` untuk GPU, `cpu` untuk CPU |

Untuk membandingkan keempat varian sekaligus:

```bash
python scripts/01_train_yolo.py \
    --data datasets/waste/data.yaml \
    --all-variants \
    --name-prefix waste
```

Contoh keluaran:

```
======================================================================
Melatih yolov8s-seg.pt -> waste-yolov8s-seg
======================================================================
...
Hasil validasi yolov8s-seg.pt:
  mask mAP@50    : 0.9910
  mask mAP@50-95 : 0.9260
  box  mAP@50-95 : 0.9410
  Bobot tersimpan: runs/segment/waste-yolov8s-seg/weights/best.pt
```

Salin bobot terbaik ke folder `weights/`:

```bash
mkdir -p weights
cp runs/segment/waste-yolov8s-seg/weights/best.pt weights/yolov8s-seg-waste.pt
```

**Pemilihan model** ditentukan oleh nilai **mask mAP@50-95**, bukan mAP@50 maupun metrik box. Alasannya, metrik tersebut menilai ketepatan batas piksel mask pada rentang ambang IoU yang ketat, dan ketepatan batas itulah yang menentukan kualitas titik-titik yang diproyeksikan ke ruang tiga dimensi.

---

## Langkah 2 — Kalibrasi Jarak Kamera dan Objek

Langkah ini **hanya perlu dijalankan bila menggunakan kamera selain Logitech C310**. Konstanta hasil kalibrasi penelitian ini sudah tertanam di `src/common/config.py`.

### Mengumpulkan data kalibrasi

1. Siapkan objek referensi berdimensi diketahui. Penelitian ini memakai botol plastik 600 ml berukuran 6,4 × 23,3 cm.
2. Rekam objek pada beberapa jarak terukur, misalnya 38 cm sampai 50 cm.
3. Ekstraksi oriented bounding box tiap frame melalui segmentasi, lalu catat lebar `w` dan tinggi `h` dalam piksel.
4. Susun sebagai CSV dengan kolom `w_px`, `h_px`, `distance_cm`.

### Menjalankan kalibrasi

```bash
python scripts/02_calibrate_distance.py \
    --csv data/calibration_distance.csv \
    --plot assets/kurva_kalibrasi.png
```

Keluaran:

```
============================================================
HASIL KALIBRASI JARAK KAMERA DAN OBJEK
============================================================
Jumlah titik data : 16
Konstanta a       : 47410.67
Konstanta b       : 16.15837
Persamaan akhir   : D = 47410.67 / s + 16.15837
RMSE              : 2.470 cm
R kuadrat         : 0.63437
```

Buku Tugas Akhir membulatkan kedua konstanta tersebut menjadi **a = 47410** dan **b = 16,159**, dan nilai bulat itulah yang tertanam pada `src/common/config.py`.

Perbarui `DIST_A` dan `DIST_B` bila hendak memakai hasil kalibrasi baru.

---

## Langkah 3 — Estimasi Volume Sampah Terapung

```bash
python scripts/03_run_waste_pipeline.py \
    --weights weights/yolov8s-seg-waste.pt \
    --video data/videos/uji.mp4 \
    --mde zoedepth-kitti \
    --conf 0.5 \
    --ground-truth data/ground_truth_waste.csv \
    --output results/waste_zoedepth_kitti.csv
```

| Argumen | Bawaan | Keterangan |
|---|---|---|
| `--weights` | wajib | Bobot YOLO hasil Langkah 1 |
| `--video` | — | Path video masukan |
| `--image` | — | Path citra masukan (alternatif `--video`) |
| `--mde` | `zoedepth-kitti` | Kunci model MDE |
| `--conf` | 0.5 | Ambang confidence YOLO |
| `--frame-stride` | 1 | Memproses satu dari setiap N frame |
| `--max-frames` | tanpa batas | Batas jumlah frame, berguna untuk uji cepat |
| `--ground-truth` | — | CSV ground truth untuk evaluasi |
| `--output` | `results/waste_estimation.csv` | Path CSV keluaran |
| `--device` | otomatis | `cuda`, `cpu`, atau kosong |

Untuk uji cepat sebelum pemrosesan penuh:

```bash
python scripts/03_run_waste_pipeline.py \
    --weights weights/yolov8s-seg-waste.pt \
    --video data/videos/uji.mp4 \
    --frame-stride 10 \
    --max-frames 20
```

Contoh keluaran:

```
Memuat pipeline: YOLO=weights/yolov8s-seg-waste.pt, MDE=zoedepth-kitti

Memproses video: data/videos/uji.mp4
  frame 24 diproses (25 total)
  frame 49 diproses (50 total)

Total frame diproses: 120
Jumlah track terbentuk: 4

============================================================
EVALUASI ESTIMASI VOLUME -- zoedepth-kitti
============================================================
n = 4
Rata-rata GT      : 941.82 cm3
Rata-rata prediksi: 945.68 cm3
MAE               : 26.42 +/- 34.18 cm3
MAPE              : 2.92 +/- 2.21 %
Akurasi           : 97.08 %

Hasil tersimpan: results/waste_zoedepth_kitti.csv
```

Simpangan baku MAE yang melampaui rata-ratanya merupakan hal wajar pada dataset ini. Distribusi galat menjulur ke kanan karena rentang volume ground truth sangat lebar, yaitu dari 10,60 cm³ untuk tutup botol hingga 4200,00 cm³ untuk wadah plastik.

---

## Langkah 4 — Estimasi Volume Rumput Laut

### Menyiapkan CSV sampel

Kolom yang diperlukan:

| Kolom | Tipe | Keterangan |
|---|---|---|
| `photo_id` | teks | Pengenal unik foto |
| `image_path` | teks | Path ke berkas citra |
| `zone` | teks | `ZoneA` atau `ZoneB` |
| `photo_number` | bilangan | Nomor urut foto, dimulai dari 1 |
| `gt_length_cm` | desimal | Panjang ground truth |
| `gt_width_cm` | desimal | Lebar ground truth |
| `gt_thickness_cm` | desimal | Ketebalan ground truth |

Nilai `photo_number` **wajib berurutan** karena menentukan dua hal sekaligus:

- **Kunci grup**: `group_key = (photo_number - 1) // 6`, karena satu sampel terdiri atas enam foto.
- **Jarak akuisisi**: foto ke-1 dan ke-2 pada jarak 30 cm, ke-3 dan ke-4 pada 40 cm, ke-5 dan ke-6 pada 50 cm.

Lihat `data/seaweed_samples_example.csv` sebagai contoh.

### Menjalankan pipeline

```bash
python scripts/04_run_seaweed_pipeline.py \
    --weights weights/yolo11s-seg-seaweed.pt \
    --samples data/seaweed_samples.csv \
    --mde dav2-large \
    --conf 0.25 \
    --output results/seaweed_dav2_large.csv
```

Contoh keluaran:

```
Memuat 366 foto sampel dari data/seaweed_samples.csv

[Tahap 1] Kalibrasi panjang fokus per zona dan skala ketebalan per grup
  kalibrasi: 50/366 foto diproses
  ...
Panjang fokus hasil kalibrasi (piksel):
  ZoneA: 2841.36
  ZoneB: 3105.72
Jumlah grup terkalibrasi: 61

[Tahap 2] Estimasi volume per foto
  ...

============================================================
EVALUASI KESELURUHAN -- dav2-large
============================================================
n = 366
MAPE              : 3.05 %
Akurasi           : 96.95 %

Rincian per zona:

  Zona ZoneA (n = 186)
    MAPE    : 2.98 %
    Akurasi : 97.02 %

  Zona ZoneB (n = 180)
    MAPE    : 3.12 %
    Akurasi : 96.88 %
```

---

## Langkah 5 — Perbandingan Keenam Model MDE

Mereproduksi Tabel 4.4 dan Tabel 4.7 pada buku Tugas Akhir.

```bash
# Studi kasus sampah terapung
python scripts/05_compare_mde_models.py \
    --case waste \
    --weights weights/yolov8s-seg-waste.pt \
    --video data/videos/uji.mp4 \
    --ground-truth data/ground_truth_waste.csv \
    --output results/perbandingan_mde_sampah.csv

# Studi kasus rumput laut
python scripts/05_compare_mde_models.py \
    --case seaweed \
    --weights weights/yolo11s-seg-seaweed.pt \
    --samples data/seaweed_samples.csv \
    --output results/perbandingan_mde_rumput_laut.csv
```

Untuk mengevaluasi sebagian model saja:

```bash
python scripts/05_compare_mde_models.py \
    --case waste \
    --weights weights/yolov8s-seg-waste.pt \
    --video data/videos/uji.mp4 \
    --ground-truth data/ground_truth_waste.csv \
    --models zoedepth-kitti dav2-large
```

Contoh keluaran akhir:

```
==============================================================================
PERINGKAT MODEL MDE (diurutkan berdasarkan MAPE)
==============================================================================
Model                          MAE (cm3)   MAPE (%)  Akurasi (%)
------------------------------------------------------------------------------
ZoeDepth (KITTI)                   26.42       2.92        97.08
ZoeDepth (NYUv2)                   27.94       2.93        97.07
Depth-Anything V2 Base             27.97       3.08        96.92
DPT-Large (Intel)                  28.66       3.13        96.87
MiDaS v3.1 BEiT-L-512              30.50       3.24        96.76
Depth-Anything V2 Large            31.65       3.45        96.55
------------------------------------------------------------------------------
Model terbaik: ZoeDepth (KITTI)
```

Perhatikan pembalikan peringkat antara kedua studi kasus. Depth-Anything V2 Large menempati posisi **terakhir** pada studi kasus sampah, namun menjadi yang **terbaik** pada studi kasus rumput laut. Inilah bukti empiris bahwa pemilihan model MDE bergantung pada metode volume yang digunakan, bukan pada kualitas model secara umum.

---

## Menjalankan Uji Unit

```bash
pytest tests/ -v                       # seluruh uji
pytest tests/ -v -k "convex_hull"      # subset tertentu
pytest tests/ --tb=short -q            # keluaran ringkas
```

Uji ini tidak memerlukan bobot model maupun GPU, sehingga dapat dijalankan kapan saja untuk memastikan tidak ada yang rusak setelah modifikasi.
