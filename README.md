# 🏥 Klasifikasi Balita Stunting Berdasarkan Kelurahan di Kota Bandung

![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Aplikasi klasifikasi stunting balita berbasis Machine Learning dengan antarmuka web Streamlit.

## 📌 Fitur Utama
1. 🏠 *Dashboard*: Ringkasan data, metrik utama, dan visualisasi prevalensi stunting.
2. 📊 *Eksplorasi Data*: Visualisasi interaktif (distribusi, korelasi, scatter plot).
3. 🤖 *Model & Evaluasi*: Latih dan evaluasi model ML (Logistic Regression, Random Forest, Naive Bayes).
4. 🔮 *Prediksi*: Form input untuk prediksi klasifikasi stunting kelurahan.
5. 📁 *Upload Data*: Prediksi batch dengan mengunggah file CSV.

## 📊 Tentang Data

### Sumber Data
Dataset diambil dari *Open Data Jawa Barat (Open Data Jabar)*, yang menyediakan data balita terukur stunting per kelurahan di Kota Bandung. Open Data Jabar dipilih karena merupakan data resmi pemerintah provinsi yang terbuka dan dapat dipertanggungjawabkan, sehingga cocok dijadikan dasar klasifikasi untuk keperluan akademik maupun analisis kesehatan masyarakat.

### Variabel yang Digunakan
| No | Variabel | Keterangan |
|----|----------|------------|
| 1 | jumlah_balita | Total balita yang terdata di suatu kelurahan |
| 2 | jumlah_balita_diukur | Jumlah balita yang benar-benar diukur tinggi badannya |
| 3 | jumlah_balita_stunting_sangat_pendek | Balita dengan kategori severely stunted |
| 4 | jumlah_balita_stunting_pendek | Balita dengan kategori stunted |

Kolom tambahan: nama_kecamatan, nama_kelurahan, tahun.

### Alasan Pemilihan Data
Keempat variabel ini dipilih karena merupakan indikator langsung yang digunakan dalam definisi stunting menurut standar antropometri (tinggi badan menurut usia). Dengan membandingkan jumlah balita yang diukur terhadap jumlah balita kategori pendek/sangat pendek, dapat dihitung *prevalensi stunting per kelurahan*, yang kemudian dijadikan dasar label klasifikasi (stunting / tidak stunting).

### Cakupan Data
Dataset mencakup *150 kelurahan* di Kota Bandung dengan data time-series *tahun 2021–2024, total **604 baris data* (kelurahan × tahun).

## ⚙️ Pengolahan Data (Preprocessing)

1. *Cleaning* — menangani missing value dan memastikan tipe data numerik konsisten di seluruh kolom.
2. *Feature Engineering* — menghitung *persentase prevalensi stunting* per kelurahan dari rasio (jumlah balita pendek + sangat pendek) terhadap jumlah balita diukur.
3. *Labeling* — menentukan kelas target (Stunting / Tidak Stunting) berdasarkan ambang batas prevalensi, default *20%* (dapat diatur dinamis lewat slider Threshold di halaman Model & Evaluasi).
4. *Scaling* — fitur numerik dinormalisasi menggunakan StandardScaler agar skala antar variabel seragam, terutama penting untuk Logistic Regression.
5. *Split Data* — dataset dibagi *80:20* (Train: 483 baris, Test: 120 baris) dengan *5-fold Cross Validation* untuk validasi model yang lebih robust.

## 🤖 Algoritma & Alasan Pemilihan

Proyek ini membandingkan tiga algoritma klasifikasi:

| Algoritma | Alasan Pemilihan |
|-----------|-------------------|
| *Logistic Regression* | Cocok sebagai baseline untuk klasifikasi biner (stunting/tidak), mudah diinterpretasi, dan cepat dilatih meski dataset relatif kecil. |
| *Random Forest* | Mampu menangkap hubungan non-linear antar variabel dan lebih tahan terhadap outlier, cocok karena data kesehatan masyarakat sering memiliki variasi antar kelurahan. |
| *Naive Bayes* | Sederhana, cepat, dan bekerja baik pada dataset kecil dengan fitur yang relatif independen, dijadikan pembanding terhadap dua model lainnya. |

Ketiga model dilatih dan dievaluasi pada data uji yang sama, lalu dibandingkan performanya (akurasi, precision, recall, F1-score) untuk menentukan model terbaik yang digunakan pada fitur *Prediksi*.

## 📁 Struktur Proyek (CCDS)
stunting-bandung-classification/
├── app.py                      # Entry point Streamlit
├── requirements.txt            # Dependencies
├── README.md                   # Dokumentasi
├── .streamlit/
│   └── secrets.toml            # Secrets management (opsional)
├── data/
│   └── stunting_bandung.csv    # Dataset utama
├── models/
│   ├── init.py
│   ├── model_utils.py          # Utilitas model ML
│   ├── train_model.py          # Script pelatihan model
│   ├── trained_model.pkl       # Model terlatih
│   └── scaler.pkl              # StandardScaler
└── utils/
├── init.py
└── data_processing.py      # Preprocessing & feature engineering

## 🚀 Cara Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👥 Anggota Kelompok
| No | Nama | NIM | Peran |
|----|------|-----|-------|
| 1 | Muhamad Rizki Yunara | 20124101 | Data Preprocessing & Modeling |
| 2 | Ikhsan Amrillah Arsy | 20124104 | UI/UX & Streamlit Development |
| 3 | Rizki Fatimah Az-zahra | 20124099 | Data Collection & EDA |
| 4 | Muhamad Luthfi Almanfaluthi | 20124092 | Model Evaluation & Testing |
| 5 | Rio | 20124113 | Documentation & Presentation |

## 📝 Lisensi
MIT License