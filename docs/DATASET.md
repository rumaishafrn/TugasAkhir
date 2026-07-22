# Dokumentasi Dataset

Kedua dataset dikumpulkan sendiri dalam penelitian ini, dengan karakteristik akuisisi yang berbeda sesuai kebutuhan masing-masing studi kasus.

---

## Ringkasan Statistik

| Aspek | Sampah Terapung | Rumput Laut |
|---|---|---|
| Jenis masukan | Video | Foto tunggal |
| Jumlah citra sebelum augmentasi | 135 | 366 |
| Jumlah citra setelah augmentasi | 325 | 959 |
| Pembagian latih / validasi / uji | 285 / 27 / 13 | 893 / 33 / 33 |
| Jumlah kelas | 6 | 1 |
| Resize (stretch) | 432 × 432 | 640 × 640 |
| Keluaran augmentasi per citra | 3 | 3 |

---

## Dataset 1: Sampah Terapung

### Akuisisi

Video direkam menggunakan webcam eksternal **Logitech C310** pada lingkungan terkontrol, yaitu objek yang mengapung di atas permukaan air dalam wadah, sehingga kondisi pencahayaan dan latar relatif stabil. Video kemudian dikonversi menjadi kumpulan frame citra.

Sebuah objek referensi berdimensi diketahui, yaitu botol plastik 600 ml, turut diletakkan pada area perekaman sebagai acuan kalibrasi skala.

### Spesimen dan volume ground truth

Sembilan spesimen mewakili enam kelas objek.

| Kelas | Spesimen | Model Geometri | Dimensi (cm) | Volume GT (cm³) |
|---|---|---|---|---|
| bottle cap | Tutup Botol Aqua 330 ml | Silinder | d = 3,0; t = 1,5 | 10,60 |
| can | Kaleng Nescafe Latte 240 ml | Silinder | d = 5,3; t = 13,5 | 297,55 |
| can | Kaleng Hemaviton C1000 330 ml | Silinder | d = 6,0; t = 12,0 | 339,12 |
| cardboard | Kardus Odol Enzim | Balok | 20,0 × 4,5 × 3,5 | 315,00 |
| cardboard | Kardus Nissin Longer Stick | Balok | 18,0 × 8,0 × 3,0 | 432,00 |
| plastic bottle | Botol Aqua 330 ml | Silinder | d = 6,0; t = 15,0 | 423,90 |
| plastic bottle | Botol Air Minum Indomaret 600 ml | Silinder | d = 7,0; t = 23,0 | 884,61 |
| plastic container | Chitato Lite Hangout Pack | Balok | 30,0 × 20,0 × 7,0 | 4200,00 |
| plastic lid | Tutup Kaleng Fox's Berries | Silinder | d = 10,0; t = 1,2 | 94,20 |

Data ini tersedia pada `data/ground_truth_waste.csv`.

Rentang volume membentang lebih dari **400 kali lipat**, dari 10,60 cm³ hingga 4200,00 cm³. Rentang yang lebar inilah yang menjadikan konsistensi MAPE di sekitar 3 persen sebagai temuan yang bermakna, sekaligus menjelaskan mengapa simpangan baku MAE dapat melampaui nilai rata-ratanya.

### Augmentasi

| Jenis | Parameter |
|---|---|
| Auto-orient | Diterapkan |
| Resize | 432 × 432 (stretch) |
| Flip | Horizontal |
| Rotasi | −15° sampai +15° |
| Brightness | −15% sampai +15% |
| Blur | Hingga 2,5 piksel |
| Noise | Hingga 0,1% piksel |

Augmentasi hanya diterapkan pada data latih untuk mencegah kebocoran data ke data validasi dan uji.

---

## Dataset 2: Sampel Rumput Laut

### Akuisisi

Dataset terdiri atas **61 sampel** yang dikumpulkan dari dua lokasi budidaya berbeda.

| Zona | Lokasi | Kamera | Jumlah Foto |
|---|---|---|---|
| Zone A | Takalar, Sulawesi Selatan | Poco M4 Pro 4G | 186 |
| Zone B | Mamuju, Sulawesi Barat | Samsung Galaxy A53 5G | 180 |

Setiap sampel difoto pada tiga ketinggian kamera dengan dua foto pada tiap ketinggian, sehingga diperoleh enam foto per sampel.

| Posisi Foto dalam Grup | Jarak Kamera |
|---|---|
| Foto ke-1 dan ke-2 | 30 cm |
| Foto ke-3 dan ke-4 | 40 cm |
| Foto ke-5 dan ke-6 | 50 cm |

Ukuran fisik objek pada satu sampel bernilai sama meskipun jarak kamera berubah. Nomor urut foto dalam satu grup karenanya dipetakan langsung ke jarak akuisisi yang bersesuaian.

Penggunaan dua kamera ponsel berbeda inilah yang mengharuskan kalibrasi panjang fokus dilakukan **per zona**, bukan secara global.

### Ground truth

Volume acuan diperoleh dari pengukuran fisik menggunakan penggaris, yaitu perkalian panjang, lebar, dan ketebalan sampel, sehingga konsisten dengan model PLT.

### Augmentasi

| Jenis | Parameter |
|---|---|
| Auto-orient | Diterapkan |
| Resize | 640 × 640 (stretch) |
| Flip | Horizontal dan vertikal |
| Blur | Hingga 2,5 piksel |

---

## Anotasi

Seluruh citra dianotasi pada tingkat piksel untuk tugas instance segmentation menggunakan platform **Roboflow**. Dataset diekspor dalam format **YOLOv8 Segmentation**.

Struktur ekspor:

```
datasets/<nama>/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

Berkas label berformat polygon YOLO, yaitu satu baris per objek dengan koordinat ternormalisasi:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

---

## Menyiapkan Dataset Sendiri

Bagi yang hendak mereplikasi penelitian dengan objek berbeda, berikut langkah-langkahnya.

**1. Kumpulkan citra.** Untuk masukan video, rekam pada lingkungan dengan pencahayaan stabil. Sertakan objek referensi berdimensi diketahui bila hendak melakukan kalibrasi jarak.

**2. Ukur ground truth.** Ukur dimensi fisik tiap spesimen, lalu hitung volumenya menggunakan model geometri yang sesuai. Fungsi pembantu tersedia:

```python
from src.common.metrics import cylinder_volume, box_volume

print(cylinder_volume(diameter_cm=7.0, height_cm=23.0))   # 885.14
print(box_volume(30.0, 20.0, 7.0))                        # 4200.00
```

**3. Anotasi.** Unggah ke Roboflow, anotasi pada tingkat piksel dengan alat polygon, lalu terapkan pra-pemrosesan dan augmentasi.

**4. Ekspor.** Pilih format YOLOv8 Segmentation, lalu unduh ke folder `datasets/`.

**5. Kalibrasi ulang.** Jalankan Langkah 2 pada [PENGGUNAAN.md](PENGGUNAAN.md) untuk memperoleh konstanta jarak yang sesuai dengan kamera yang digunakan. Konstanta bawaan repositori ini **tidak berlaku** untuk kamera lain.

---

## Ketersediaan Data

Citra mentah dan video tidak disertakan dalam repositori karena ukurannya besar. Yang disertakan hanyalah:

- `data/calibration_distance.csv` — 16 titik data kalibrasi jarak
- `data/ground_truth_waste.csv` — volume ground truth sembilan spesimen sampah
- `data/seaweed_samples_example.csv` — contoh format CSV sampel rumput laut

Permintaan akses dataset lengkap dapat diajukan kepada penulis atau dosen pembimbing.
