# Panduan Instalasi

## Kebutuhan Sistem

| Komponen | Minimum | Direkomendasikan |
|---|---|---|
| Python | 3.10 | 3.11 atau 3.12 |
| RAM | 8 GB | 16 GB |
| GPU | Tidak wajib | NVIDIA dengan VRAM 6 GB atau lebih |
| Ruang disk | 10 GB | 20 GB (bobot model MDE cukup besar) |

Penelitian ini dijalankan pada lingkungan **GPU Kaggle (Tesla T4)**. Pipeline tetap dapat berjalan pada CPU, hanya saja inferensi MDE menjadi jauh lebih lambat, khususnya untuk masukan berupa video.

---

## 1. Instalasi Dasar

```bash
git clone https://github.com/USERNAME/volumetric-estimation-rgb.git
cd volumetric-estimation-rgb

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verifikasi:

```bash
pytest tests/ -v
```

Seluruh 23 uji harus lolos. Uji ini tidak memerlukan bobot model maupun GPU.

---

## 2. Instalasi dengan Dukungan GPU

Perintah `pip install torch` bawaan seringkali memasang versi CPU. Untuk CUDA, pasang PyTorch terlebih dahulu dari indeks resminya:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Baru pasang sisanya
pip install -r requirements.txt
```

Verifikasi GPU terdeteksi:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU saja')"
```

---

## 3. Dependensi Opsional

Open3D dipakai pada penelitian asli untuk Statistical Outlier Removal. Repositori ini sudah menyertakan implementasi pengganti berbasis `scipy.spatial.cKDTree`, sehingga **Open3D tidak wajib dipasang**.

```bash
pip install -r requirements-optional.txt
```

Pasang hanya bila hendak melakukan visualisasi point cloud secara interaktif atau memakai pustaka yang identik dengan penelitian asli.

---

## 4. Menjalankan di Kaggle atau Google Colab

```python
# Sel 1: klon dan pasang dependensi
!git clone https://github.com/USERNAME/volumetric-estimation-rgb.git
%cd volumetric-estimation-rgb
!pip install -q ultralytics transformers accelerate timm

# Sel 2: verifikasi
!python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Sel 3: jalankan pipeline
!python scripts/03_run_waste_pipeline.py \
    --weights weights/yolov8s-seg-waste.pt \
    --video data/videos/uji.mp4 \
    --mde zoedepth-kitti
```

Pada Kaggle, aktifkan akselerator GPU melalui menu **Settings → Accelerator → GPU T4 x2**.

---

## 5. Menyiapkan Bobot Model

Bobot YOLO hasil pelatihan tidak disertakan dalam repositori karena ukurannya besar. Terdapat dua pilihan.

**Melatih sendiri** (lihat [PENGGUNAAN.md](PENGGUNAAN.md) Langkah 1):

```bash
python scripts/01_train_yolo.py --data datasets/waste/data.yaml --model yolov8s-seg.pt
cp runs/segment/train/weights/best.pt weights/yolov8s-seg-waste.pt
```

**Mengunduh bobot terlatih** dari halaman Releases repositori, lalu letakkan pada struktur berikut:

```
weights/
├── yolov8s-seg-waste.pt
└── yolo11s-seg-seaweed.pt
```

Model MDE **tidak perlu diunduh manual**. Pustaka `transformers` mengunduhnya otomatis dari HuggingFace Hub saat pertama kali dipanggil, lalu menyimpannya di cache `~/.cache/huggingface`.

---

## 6. Penanganan Galat Umum

### `ModuleNotFoundError: No module named 'src'`

Script harus dijalankan dari direktori akar repositori, bukan dari dalam folder `scripts/`.

```bash
cd /path/ke/volumetric-estimation-rgb
python scripts/03_run_waste_pipeline.py ...
```

Alternatifnya, tambahkan direktori akar ke `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### `CUDA out of memory`

Model MDE besar seperti Depth-Anything V2 Large dan MiDaS BEiT-L-512 cukup boros memori. Beberapa langkah yang dapat ditempuh:

- Gunakan varian yang lebih ringan, misalnya `dav2-base` alih-alih `dav2-large`.
- Naikkan nilai `--frame-stride` agar frame yang diproses lebih sedikit.
- Kosongkan cache di antara pemanggilan model: `torch.cuda.empty_cache()`.
- Paksa berjalan pada CPU dengan `--device cpu`.

### `ImportError` terkait `timm` saat memuat model MDE

Beberapa model DPT dan BEiT bergantung pada `timm` dengan versi tertentu.

```bash
pip install --upgrade timm transformers
```

### Instalasi Open3D gagal

Open3D belum mendukung seluruh versi Python secara langsung. Karena Open3D bersifat opsional pada repositori ini, kegagalan tersebut dapat diabaikan begitu saja. Pipeline akan otomatis memakai implementasi SOR berbasis SciPy.

### `cv2.error` saat membaca video

Codec video tertentu tidak didukung OpenCV bawaan. Konversikan terlebih dahulu dengan ffmpeg:

```bash
ffmpeg -i input.mov -c:v libx264 -pix_fmt yuv420p output.mp4
```

### Unduhan model MDE sangat lambat atau terputus

Atur cache HuggingFace ke lokasi dengan ruang memadai, lalu ulangi. Unduhan bersifat resumable.

```bash
export HF_HOME=/path/dengan/ruang/lega
```
