# Penjabaran Metodologi Pipeline

Dokumen ini memetakan tiap persamaan pada Bab III buku Tugas Akhir terhadap fungsi yang mengimplementasikannya, sehingga kode dapat ditelusuri balik ke rumusan metodologisnya.

Gambaran menyeluruh alur pipeline tersedia pada [diagram arsitektur](../assets/pipeline.svg).

---

## Peta Persamaan terhadap Kode

| Persamaan | Rumusan | Berkas | Fungsi |
|---|---|---|---|
| (3.1) | Parameter intrinsik kamera | `src/common/config.py` | `FX`, `FY`, `CX`, `CY` |
| (3.2) | `D = 47410/s + 16,159` | `src/calibration/distance.py` | `estimate_distance` |
| (3.3) | Konversi OBB ke sentimeter | `src/calibration/distance.py` | `pixel_to_cm` |
| (3.4)–(3.6) | Penskalaan kedalaman metrik | `src/waste/pointcloud.py` | `scale_depth_to_metric` |
| (3.7) | Proyeksi balik ke ruang 3D | `src/waste/pointcloud.py` | `unproject_mask_to_points` |
| (3.8) | Statistical Outlier Removal | `src/waste/pointcloud.py` | `remove_statistical_outliers` |
| (3.9) | Volume convex hull | `src/waste/convex_hull.py` | `convex_hull_volume` |
| (3.10) | Koreksi `V_hull / 1,58` | `src/waste/convex_hull.py` | `corrected_volume` |
| (3.11) | Kesebangunan pinhole | `src/calibration/distance.py` | `pixel_to_cm` |
| (3.12)–(3.13) | Kalibrasi `f_px` per zona | `src/seaweed/pipeline.py` | `calibrate` |
| (3.14) | Rentang kedalaman `P95 − P5` | `src/seaweed/pipeline.py` | `depth_span` |
| (3.15)–(3.16) | Faktor skala ketebalan | `src/seaweed/pipeline.py` | `calibrate` |
| (3.17) | Volume PLT | `src/seaweed/pipeline.py` | `estimate_one` |
| (3.18)–(3.19) | Volume ground truth geometri | `src/common/metrics.py` | `cylinder_volume`, `box_volume` |
| (3.20)–(3.22) | MAE, MAPE, akurasi | `src/common/metrics.py` | `evaluate_volume` |

---

## Tahap Bersama

### Instance Segmentation

Empat varian YOLO berukuran small dilatih secara terpisah pada kedua dataset selama 100 epoch dengan transfer learning. Varian small dipilih sebagai kompromi antara akurasi dan kecepatan inferensi, mengingat studi kasus sampah menuntut pemrosesan video yang mendekati waktu nyata.

Keluaran yang dibutuhkan tahap berikutnya adalah **mask biner pada tingkat piksel**, bukan sekadar kotak pembatas. Ketepatan batas mask inilah yang menentukan kualitas titik-titik yang diproyeksikan ke ruang tiga dimensi, sehingga pemilihan model menekankan nilai **mask mAP@50-95**.

### Monocular Depth Estimation

Enam model MDE dievaluasi dengan bobot pre-trained tanpa fine-tuning. Model terbagi menjadi dua jenis keluaran:

- **Metrik** (ZoeDepth-KITTI, ZoeDepth-NYUv2) menghasilkan kedalaman dalam satuan fisik secara langsung.
- **Relatif** (Depth-Anything V2, MiDaS, DPT-Large) menghasilkan nilai tanpa satuan yang memerlukan kalibrasi skala.

Pipeline memperlakukan keduanya seragam dengan melakukan penskalaan berbasis jangkar jarak, sehingga keenam model dapat dipertukarkan tanpa mengubah kode.

---

## Jalur A: Sampah Terapung (Convex Hull)

Objek sampah tergolong **reguler**, sehingga ketiga dimensinya dapat diperoleh melalui rekonstruksi permukaan tiga dimensi.

### A.1 Kalibrasi kamera dan estimasi jarak

Parameter intrinsik diturunkan dari FOV diagonal 60° milik Logitech C310 pada resolusi 1280 × 720:

```
d_diag = sqrt(1280² + 720²) ≈ 1468,6 piksel
f      = (d_diag / 2) / tan(60° / 2) ≈ 1271,8 → dibulatkan 1272 piksel
```

Titik utama ditetapkan pada pusat citra, yaitu (640, 360).

Karena posisi kamera terhadap objek bersifat dinamis, jarak tidak diukur langsung melainkan diestimasi dari ukuran tampak objek. Pada model pinhole, ukuran proyeksi berbanding terbalik terhadap jarak, sehingga jarak dimodelkan sebagai fungsi invers:

```
D = a / s + b       dengan  s = w_obb + h_obb
```

Suku `a/s` menangkap hubungan proyektif terbalik tersebut, dengan `a` pada dasarnya menggabungkan panjang fokus dan ukuran nyata objek referensi. Konstanta aditif `b` mengoreksi penyimpangan pada jarak dekat yang muncul akibat selisih antara ukuran oriented bounding box dan ukuran proyeksi ideal.

Regresi non-linear kuadrat terkecil terhadap 16 pasangan data menghasilkan **a = 47410,68** dan **b = 16,15837**, yang dibulatkan menjadi 47410 dan 16,159.

**Catatan penting mengenai ketahanan terhadap galat kalibrasi.** Volume convex hull bersifat invarian terhadap translasi, sehingga ketidaktepatan penempatan titik utama tidak memengaruhi hasil akhir. Adapun kesalahan skala absolut pada panjang fokus termuat sebagai galat sistematik yang seragam pada seluruh volume mentah, sehingga terserap oleh faktor koreksi empiris. Akurasi akhir pipeline karenanya tidak bergantung pada kesempurnaan kalibrasi intrinsik, melainkan pada konsistensi penskalaan terhadap acuan ground truth.

### A.2 Penskalaan peta kedalaman metrik

Peta kedalaman keluaran MDE diubah menjadi kedalaman metrik dengan jarak hasil kalibrasi sebagai jangkar:

```
d̄_rel  = (1/N) Σ d_rel,i                    (3.4)
k       = D_nyata / d̄_rel                    (3.5)
D_skala = d_rel(x, y) × k                    (3.6)
```

Rata-rata kedalaman objek dipetakan tepat ke jarak `D`, sedangkan variasi kedalaman antar-piksel di dalam objek tetap dipertahankan secara proporsional.

### A.3 Rekonstruksi point cloud

Setiap piksel pada area mask diproyeksikan balik ke ruang tiga dimensi:

```
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy                        (3.7)
Z = kedalaman metrik piksel
```

### A.4 Pembersihan pencilan

Point cloud hasil rekonstruksi mengandung titik bising akibat kesalahan estimasi kedalaman, khususnya di tepi objek. Titik-titik tersebut dibersihkan dengan Statistical Outlier Removal:

```
d_i > μ + α · σ    →  dibuang               (3.8)
```

dengan `k = 20` tetangga terdekat dan `α = 1,5`. Untuk tiap titik dihitung jarak rata-rata terhadap `k` tetangganya, lalu titik yang jarak rata-ratanya melampaui ambang global dianggap pencilan.

### A.5 Estimasi volume convex hull

```
V_hull = (1/6) Σ |(v_a − v_d) · ((v_b − v_d) × (v_c − v_d))|     (3.9)
```

Volume dihitung sebagai penjumlahan volume tetrahedron yang dibentuk tiap facet segitiga terhadap satu titik acuan. Kompleksitas pembentukan convex hull tiga dimensi adalah sekitar `O(n log n)`.

Karena convex hull membungkus seluruh titik terluar secara konveks, volume mentah cenderung melebih-estimasi objek nyata:

```
V_terkoreksi = V_hull / 1,58                 (3.10)
```

### A.6 Konsistensi antar-frame

Masukan berupa video, sehingga objek yang sama harus dikenali sebagai satu entitas di sepanjang frame. Asosiasi dilakukan dengan algoritma penugasan **Hungarian** berbasis kemiripan **IoU**. Matriks biaya disusun sebagai `1 − IoU`, sehingga meminimalkan biaya setara dengan memaksimalkan kemiripan.

Volume tiap track kemudian distabilkan dengan **rata-rata terpangkas** pada jendela lima frame. Penghalusan ini meredam fluktuasi sesaat akibat oklusi maupun kesalahan deteksi pada frame tertentu.

---

## Jalur B: Rumput Laut (Metode PLT)

Objek rumput laut tergolong **ireguler**, sehingga rekonstruksi permukaan tertutup penuh tidak sesuai. Pendekatan kotak pembatas tiga dimensi dipilih karena sebagian besar dimensi objek dapat teramati dari satu sudut pandang, sekaligus konsisten dengan cara pengukuran volume acuan yang lazim dilakukan sebagai perkalian panjang, lebar, dan ketebalan.

Prosedur dijalankan dalam dua tahap karena panjang fokus tidak diketahui, mengingat akuisisi menggunakan dua kamera ponsel yang berbeda.

### B.1 Tahap kalibrasi

**Panjang fokus per zona.** Nilai `f_px` dikalibrasi dengan membalik hubungan kesebangunan pinhole, lalu diambil mediannya per zona:

```
f_px,i    = (ukuran_px,i · jarak_cm,i) / ukuran_GT,i          (3.12)
f_px,zona = median{ f_px,i : i ∈ zona }                      (3.13)
```

Median dipilih alih-alih rata-rata agar hasil kalibrasi tahan terhadap pencilan yang dapat muncul dari kesalahan segmentasi pada sebagian foto.

**Faktor skala ketebalan per grup.** Rentang kedalaman dihitung menggunakan persentil agar tahan terhadap nilai ekstrem:

```
s    = P95(d_rel) − P5(d_rel)                                (3.14)
sz_g = median{ t_GT / s : foto ∈ grup g }                    (3.15)
```

Penggunaan persentil ke-95 dan ke-5, bukan nilai maksimum dan minimum, dilakukan untuk mengabaikan pencilan kedalaman ekstrem di tepi objek. Satu grup didefinisikan sebagai satu sampel yang terdiri atas enam foto, dengan `group_key = (nomor_foto − 1) // 6`.

### B.2 Tahap estimasi volume

Panjang dan lebar dalam piksel diperoleh dari mask hasil segmentasi melalui `minAreaRect`, yaitu sisi terpanjang sebagai `P` dan sisi terpendek sebagai `L`. Keduanya dikonversi ke sentimeter:

```
P = P_px · jarak / f_px,zona
L = L_px · jarak / f_px,zona                                 (3.11)
T = s · sz_g                                                 (3.16)
V = P × L × T                                                (3.17)
```

---

## Evaluasi

### Volume ground truth

Sampah dihitung dari pengukuran dimensi fisik tiap spesimen menggunakan model geometri yang sesuai:

```
V_silinder = π (d/2)² · t                                    (3.18)
V_balok    = p × l × t                                       (3.19)
```

Rumput laut diukur secara fisik menggunakan penggaris sebagai perkalian panjang, lebar, dan ketebalan, sehingga konsisten dengan model PLT.

### Metrik

```
MAE     = (1/n) Σ |V_pred,i − V_gt,i|                        (3.20)
MAPE    = (100%/n) Σ |(V_pred,i − V_gt,i) / V_gt,i|          (3.21)
Akurasi = 100% − MAPE                                        (3.22)
```

Simpangan baku kedua metrik turut dilaporkan untuk menggambarkan konsistensi antar objek.

---

## Keterbatasan

Beberapa keterbatasan yang perlu diperhatikan sebelum memperluas penggunaan pipeline ini.

**Kalibrasi bersifat spesifik perangkat.** Konstanta jarak berlaku untuk Logitech C310, sedangkan panjang fokus per zona berlaku untuk Poco M4 Pro 4G dan Samsung Galaxy A53 5G. Penggunaan perangkat lain memerlukan kalibrasi ulang.

**Faktor koreksi diturunkan dari dataset evaluasi.** Faktor 1,58 diregresikan terhadap volume ground truth pada dataset yang sama dengan dataset evaluasi. Hal ini dinyatakan secara terbuka pada buku Tugas Akhir. Yang menopang kelayakannya adalah konsistensi galat pada rentang volume yang membentang lebih dari 400 kali lipat, yang menunjukkan bahwa faktor ini menangkap bias geometrik sistematik convex hull.

**Metode PLT mengamati satu sudut pandang.** Pendekatan ini kurang menangkap variasi ketebalan yang kompleks pada objek ireguler.

**Model MDE tidak di-fine-tune.** Seluruh model digunakan dengan bobot pre-trained, sehingga belum dioptimalkan untuk karakteristik visual objek di lingkungan perairan.

**Lingkungan akuisisi terkontrol.** Dataset sampah direkam pada wadah berisi air dengan pencahayaan dan latar relatif stabil, bukan di perairan laut terbuka secara langsung.
