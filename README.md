# 🏥 Klasifikasi Balita Stunting Berdasarkan Kelurahan di Kota Bandung

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aplikasi klasifikasi stunting balita berbasis **Machine Learning** dengan antarmuka web **Streamlit**.

---

## 📌 Fitur Utama

1. **🏠 Dashboard**: Ringkasan data, metrik utama, dan visualisasi prevalensi stunting.
2. **📊 Eksplorasi Data**: Visualisasi interaktif (distribusi, korelasi, scatter plot).
3. **🤖 Model & Evaluasi**: Latih dan evaluasi model ML (Logistic Regression, Random Forest, Naive Bayes).
4. **🔮 Prediksi**: Form input untuk prediksi klasifikasi stunting kelurahan.
5. **📁 Upload Data**: Prediksi batch dengan mengunggah file CSV.

---

## 📁 Struktur Proyek (CCDS)

```
stunting-bandung-classification/
├── app.py                      # Entry point Streamlit
├── requirements.txt            # Dependencies
├── README.md                   # Dokumentasi
├── .streamlit/
│   └── secrets.toml            # Secrets management (opsional)
├── data/
│   └── stunting_bandung.csv    # Dataset utama
├── models/
│   ├── __init__.py
│   ├── model_utils.py          # Utilitas model ML
│   ├── train_model.py          # Script pelatihan model
│   ├── trained_model.pkl       # Model terlatih
│   └── scaler.pkl              # StandardScaler
└── utils/
    ├── __init__.py
    └── data_processing.py      # Preprocessing & feature engineering
```

---

## 🚀 Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👥 Anggota Kelompok

| No | Nama | NIM | Peran |
|---|---|---|---|
| 1 | Muhamad Rizki Yunara | 20124101 | Data Preprocessing & Modeling |
| 2 | Ikhsan Amrillah Arsy | 20124104 | UI/UX & Streamlit Development |
| 3 | Rizki Fatimah Az-zahra | 20124099 | Data Collection & EDA |
| 4 | Muhamad Luthfi Almanfaluthi | 20124092 | Model Evaluation & Testing |
| 5 | Rio Aji Pamungkas | 20124113 | Documentation & Presentation |

---

## 📝 Lisensi

MIT License
