# 🏥 Klasifikasi Balita Stunting Berdasarkan Kelurahan di Kota Bandung

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyek ini merupakan aplikasi berbasis **Machine Learning** yang dirancang untuk mengklasifikasikan tingkat stunting balita berdasarkan data kelurahan di Kota Bandung. Aplikasi web interaktif dibangun menggunakan **Streamlit**, sedangkan analisis data dan pemodelan dilakukan menggunakan **Scikit-Learn**.

---

## 📌 Fitur Utama

Aplikasi ini memiliki 5 modul utama yang dapat diakses melalui bilah navigasi:

1. **🏠 Dashboard**: Menyajikan ringkasan data, metrik utama (total balita, prevalensi rata-rata, jumlah kelurahan terdampak), serta sebaran kasus stunting.
2. **📊 Eksplorasi Data (EDA)**: Visualisasi interaktif (analisis distribusi, korelasi fitur, scatter plot, dan tren tahunan) menggunakan **Plotly** dan **Matplotlib/Seaborn**.
3. **🤖 Model & Evaluasi**: Melatih dan mengevaluasi performa model Machine Learning secara real-time. Membandingkan algoritma berdasarkan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, *Cross-Validation Score*, serta menampilkan kurva ROC.
4. **🔮 Prediksi Mandiri**: Form interaktif untuk memprediksi status stunting kelurahan secara cepat berdasarkan input manual parameter balita.
5. **📁 Upload Data**: Fitur klasifikasi massal (*batch prediction*) dengan mengunggah file dataset baru dalam format CSV atau Excel.

---

## ⚙️ Model Machine Learning yang Digunakan

Aplikasi ini membandingkan beberapa algoritma klasifikasi populer untuk menentukan model terbaik:
* **Logistic Regression**
* **Random Forest Classifier**
* **Gaussian Naive Bayes (Naive Bayes)**
* **Support Vector Machine (SVM)**
* **K-Nearest Neighbors (KNN)**

*Penentuan kelas (Stunting / Tidak Stunting) didasarkan pada threshold prevalensi stunting kelurahan (default: >= 20%).*

---

## 📁 Struktur Proyek

```text
stunting-bandung-classification/
│
├── .vscode/                 # Konfigurasi workspace dan debugger VS Code
│   ├── launch.json
│   └── settings.json
│
├── data/                    # Dataset proyek
│   ├── raw/
│   │   └── stunting_bandung.csv  # Dataset asli (Open Data Jabar)
│   └── stunting_bandung.csv      # Dataset aktif untuk aplikasi
│
├── models/                  # Modul utilitas model machine learning
│   ├── __init__.py
│   └── model_utils.py
│
├── utils/                   # Modul pemrosesan data & preprocessing
│   ├── __init__.py
│   └── data_processing.py
│
├── app.py                   # Entry point aplikasi web Streamlit
├── process.py               # Pipeline Data Science (Cleaning, EDA, Training, Evaluasi)
├── requirements.txt         # Daftar dependensi library Python
└── README.md                # Dokumentasi proyek
```

---

## 🚀 Cara Instalasi dan Menjalankan

### 1. Prasyarat
Pastikan Anda sudah menginstal Python (versi 3.9 s.d. 3.11 direkomendasikan).

### 2. Kloning Repositori
```bash
git clone https://github.com/Azahra223/stunting-bandung-classification.git
cd stunting-bandung-classification
```

### 3. Buat dan Aktifkan Virtual Environment (Direkomendasikan)
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi

* **Menjalankan Web App Streamlit (Interactive UI):**
  ```bash
  streamlit run app.py
  ```
  *Buka browser Anda dan akses tautan lokal yang muncul (biasanya `http://localhost:8501`).*

* **Menjalankan Pipeline Analisis & Pemodelan (Terminal Output & Chart Generation):**
  ```bash
  python process.py
  ```
  *Perintah ini akan menjalankan pipeline pembersihan data, training model, perbandingan akurasi di terminal, dan menyimpan visualisasi statis (`eda_visualization.png`, `roc_curve.png`, dll.) di folder root.*

---

## 📊 Dataset Sumber
Data didapatkan dari portal **Open Data Jabar** dengan rincian fitur input sebagai berikut:
1. `jumlah_balita`: Total balita terdaftar di kelurahan.
2. `jumlah_balita_diukur`: Jumlah balita yang melakukan pengukuran fisik.
3. `jumlah_balita_stunting_sangat_pendek`: Jumlah balita dengan kategori sangat pendek.
4. `jumlah_balita_stunting_pendek`: Jumlah balita dengan kategori pendek.

---

## 👥 Anggota Kelompok

| No | Nama | NIM | Peran/Kontribusi |
|---|---|---|---|
| 1 | [Muhamad Rizki yunara] | [20124101] | Data Preprocessing & Modeling |
| 2 | [Ikhsan Amrillah Arsy] | [20124104] | UI/UX & Streamlit Development |
| 3 | [Rizki Fatimah Az-zahra] | [20124099] | Data Collection & EDA Analysis |
| 4 | [Muhamad Luthfi Almanfaluthi] | [20124092] | Model Evaluation & Testing |
| 5 | [Rio Aji Pamungkas] | [20124113] | Documentation & Presentation |



---

## 📝 Lisensi
Proyek ini dilisensikan di bawah **MIT License** - lihat file [LICENSE](LICENSE) jika tersedia untuk detail selengkapnya.
