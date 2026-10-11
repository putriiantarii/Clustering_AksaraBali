# Clustering Aksara Bali (DeepLontar)

Proyek ini berfokus pada pengembangan sistem **klasterisasi karakter Aksara Bali** secara otomatis menggunakan dataset **DeepLontar**. Didesain dengan struktur kode modular untuk memfasilitasi riset Machine Learning dan Computer Vision yang terorganisir.

---

## 📁 Struktur Direktori Proyek

```text
Clustering_AksaraBali/
├── data/
│   ├── raw/          # Tempat menyimpan dataset asli (DeepLontar & DeepLontar_Labels)
│   └── processed/    # Hasil ekstraksi/cropping karakter aksara Bali
├── src/
│   ├── __init__.py
│   └── inspect_dataset.py  # Script inspeksi & visualisasi bounding box
├── .gitignore        # Pengecualian file berat & cache agar repo tetap bersih
├── config.py         # Pengaturan path & konfigurasi global proyek
├── requirements.txt  # Daftar pustaka/library Python yang dibutuhkan
└── README.md         # Dokumentasi proyek