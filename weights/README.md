# Folder Bobot Model

Letakkan bobot YOLO hasil pelatihan di sini:

```
weights/
├── yolov8s-seg-waste.pt        # studi kasus sampah terapung
└── yolo11s-seg-seaweed.pt      # studi kasus rumput laut
```

Berkas `.pt` tidak dilacak Git karena ukurannya besar. Distribusikan melalui
GitHub Releases, Google Drive, atau Git LFS.

Model MDE tidak perlu diletakkan di sini. Pustaka `transformers` mengunduhnya
otomatis dari HuggingFace Hub saat pertama kali dipanggil.
