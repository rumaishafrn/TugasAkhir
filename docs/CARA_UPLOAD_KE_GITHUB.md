# Cara Mengunggah Repositori Ini ke GitHub

Dokumen ini bersifat sementara. **Hapus berkas ini setelah repositori berhasil diunggah**, karena isinya tidak relevan bagi pembaca repositori.

---

## 1. Membuat repositori kosong di GitHub

1. Buka [github.com/new](https://github.com/new).
2. Isi **Repository name**, misalnya `volumetric-estimation-rgb`.
3. Isi **Description**, misalnya: *Estimasi volumetrik objek dari kamera RGB menggunakan YOLO instance segmentation dan monocular depth estimation.*
4. Pilih **Public** agar dapat diakses dosen penguji.
5. **Jangan** centang "Add a README file", "Add .gitignore", maupun "Choose a license", karena ketiganya sudah tersedia di dalam paket ini.
6. Klik **Create repository**.

---

## 2. Mengunggah dari komputer

Ekstrak berkas zip terlebih dahulu, lalu jalankan perintah berikut dari dalam folder hasil ekstraksi.

```bash
cd volumetric-estimation-rgb

git init
git add .
git commit -m "Kode pendamping Tugas Akhir: estimasi volumetrik objek dari kamera RGB"

git branch -M main
git remote add origin https://github.com/USERNAME/volumetric-estimation-rgb.git
git push -u origin main
```

Ganti `USERNAME` dengan nama pengguna GitHub Anda.

Bila diminta kata sandi, GitHub tidak lagi menerima kata sandi akun. Gunakan **Personal Access Token** yang dapat dibuat melalui **Settings → Developer settings → Personal access tokens → Tokens (classic)**, dengan cakupan `repo`.

---

## 3. Alternatif tanpa Git

1. Buka repositori kosong yang baru dibuat di GitHub.
2. Klik **uploading an existing file**.
3. Seret seluruh isi folder hasil ekstraksi ke area unggah.
4. Isi pesan commit, lalu klik **Commit changes**.

Perlu diperhatikan bahwa cara ini tidak dapat mengunggah folder kosong dan memiliki batas 100 berkas per unggahan.

---

## 4. Setelah repositori terunggah

### Ganti seluruh placeholder `USERNAME`

Placeholder tersebut muncul pada `README.md`, `docs/INSTALASI.md`, dan `notebooks/README.md`.

```bash
grep -rn "USERNAME" --include="*.md" .
```

Ganti dengan nama pengguna GitHub Anda, lalu commit ulang.

### Unggah bobot model melalui Releases

Berkas `.pt` sengaja diabaikan oleh `.gitignore` karena ukurannya besar.

1. Buka tab **Releases → Create a new release**.
2. Isi tag, misalnya `v1.0.0`.
3. Unggah `yolov8s-seg-waste.pt` dan `yolo11s-seg-seaweed.pt` sebagai **assets**.
4. Publikasikan, lalu tambahkan tautan unduhannya pada README.

### Tambahkan topik repositori

Melalui tombol roda gigi di sebelah "About", tambahkan topik seperti `computer-vision`, `yolo`, `instance-segmentation`, `monocular-depth-estimation`, `volume-estimation`, `marine-debris`, dan `point-cloud`. Topik membantu repositori ditemukan melalui pencarian.

### Hapus berkas ini

```bash
git rm docs/CARA_UPLOAD_KE_GITHUB.md
git commit -m "Hapus panduan unggah yang sudah tidak diperlukan"
git push
```

---

## 5. Daftar Periksa Sebelum Diserahkan

- [ ] Seluruh placeholder `USERNAME` sudah diganti
- [ ] `pytest tests/ -v` lolos seluruhnya pada mesin bersih
- [ ] Bobot model tersedia melalui Releases atau tautan alternatif
- [ ] README menampilkan tabel hasil dengan benar
- [ ] Nama kedua dosen pembimbing tertulis lengkap dan benar
- [ ] Deskripsi dan topik repositori sudah diisi
- [ ] Berkas panduan ini sudah dihapus
