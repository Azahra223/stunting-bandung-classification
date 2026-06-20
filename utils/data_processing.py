"""
Data Processing Module — Klasifikasi Stunting Kota Bandung
==========================================================
Handles dataset generation, loading, preprocessing, and feature engineering.
Dataset berdasarkan format Open Data Jabar: Jumlah Balita Stunting
Berdasarkan Kelurahan di Kota Bandung.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# Data Administratif Kota Bandung (30 Kecamatan, ~151 Kelurahan)
# Sumber referensi: BPS Kota Bandung
# ============================================================
KECAMATAN_KELURAHAN = {
    "Andir": [
        "Campaka", "Ciroyom", "Dunguscariang", "Garuda", "Kebon Jeruk", "Maleber"
    ],
    "Antapani": [
        "Antapani Kidul", "Antapani Kulon", "Antapani Tengah", "Antapani Wetan"
    ],
    "Arcamanik": [
        "Cisaranten Endah", "Cisaranten Kulon", "Sukamiskin",
        "Cisaranten Bina Harapan"
    ],
    "Astanaanyar": [
        "Cibadak", "Karang Anyar", "Karasak", "Nyengseret",
        "Panjunan", "Pelindung Hewan"
    ],
    "Babakan Ciparay": [
        "Babakan", "Babakan Ciparay", "Cirangrang",
        "Margahayu Utara", "Margasuka", "Sukahaji"
    ],
    "Bandung Kidul": [
        "Batununggal", "Mengger", "Wates", "Kujangsari"
    ],
    "Bandung Kulon": [
        "Caringin", "Cibuntu", "Cigondewah Kaler", "Cigondewah Rahayu",
        "Cijerah", "Gempolsari", "Warungmuncang", "Cigondewah Hilir"
    ],
    "Bandung Wetan": [
        "Cihapit", "Citarum", "Tamansari"
    ],
    "Batununggal": [
        "Binong", "Cibangkong", "Gumuruh", "Kacapiring",
        "Kebonwaru", "Maleer", "Samoja", "Kebon Gedang"
    ],
    "Bojongloa Kaler": [
        "Babakan Asih", "Babakan Tarogong", "Jamika", "Kopo", "Suka Asih"
    ],
    "Bojongloa Kidul": [
        "Cibaduyut", "Cibaduyut Kidul", "Cibaduyut Wetan",
        "Kebon Lega", "Mekar Wangi", "Situsaeur"
    ],
    "Buahbatu": [
        "Cijagra", "Jatisari", "Margasari", "Sekejati"
    ],
    "Cibeunying Kaler": [
        "Cigadung", "Cihaurgeulis", "Neglasari", "Sukaluyu"
    ],
    "Cibeunying Kidul": [
        "Cicadas", "Cikutra", "Padasuka", "Sukamaju",
        "Sukapada", "Pasirlayung"
    ],
    "Cibiru": [
        "Cipadung", "Cisurupan", "Palasari", "Pasir Biru"
    ],
    "Cicendo": [
        "Arjuna", "Husein Sastranegara", "Pajajaran",
        "Pamoyanan", "Pasirkaliki", "Sukaraja"
    ],
    "Cidadap": [
        "Ciumbuleuit", "Hegarmanah", "Ledeng"
    ],
    "Cinambo": [
        "Cisaranten Wetan", "Pakemitan", "Sukamulya", "Babakan Penghulu"
    ],
    "Coblong": [
        "Cipaganti", "Dago", "Lebak Gede", "Lebak Siliwangi",
        "Sadang Serang", "Sekeloa"
    ],
    "Gedebage": [
        "Cimincrang", "Cisaranten Kidul", "Rancabolang", "Rancanumpang"
    ],
    "Kiaracondong": [
        "Babakan Sari", "Babakan Surabaya", "Cicaheum",
        "Kebon Jayanti", "Kebon Kangkung", "Sukapura"
    ],
    "Lengkong": [
        "Burangrang", "Cijagra", "Cikawao", "Lingkar Selatan",
        "Malabar", "Paledang", "Turangga"
    ],
    "Mandalajati": [
        "Jatihandap", "Karang Pamulang", "Pasir Impun", "Sindang Jaya"
    ],
    "Panyileukan": [
        "Cipadung Kidul", "Cipadung Kulon", "Cipadung Wetan", "Mekarmulya"
    ],
    "Rancasari": [
        "Cipamokolan", "Derwati", "Manjahlega", "Mekarjaya"
    ],
    "Regol": [
        "Ancol", "Balonggede", "Ciateul", "Cigereleng",
        "Cisalak", "Pungkur", "Pasirluyu"
    ],
    "Sukajadi": [
        "Cipedes", "Pasteur", "Sukabungah", "Sukagalih", "Sukawarna"
    ],
    "Sukasari": [
        "Gegerkalong", "Isola", "Sarijadi", "Sukarasa"
    ],
    "Sumur Bandung": [
        "Babakan Ciamis", "Braga", "Kebon Pisang", "Merdeka"
    ],
    "Ujung Berung": [
        "Cigending", "Pasanggrahan", "Pasir Endah", "Pasir Jati", "Pasir Wangi"
    ],
}


def generate_sample_dataset(output_path='data/stunting_bandung.csv', seed=42):
    """
    Generate dataset simulasi yang realistis untuk klasifikasi stunting
    berdasarkan struktur administratif Kota Bandung.

    Kolom yang dihasilkan sesuai format Open Data Jabar:
    - jumlah_balita
    - jumlah_balita_diukur
    - jumlah_balita_stunting_sangat_pendek (TB sangat pendek)
    - jumlah_balita_stunting_pendek (TB pendek)
    - tahun, nama_kecamatan, nama_kelurahan, dll.
    """
    np.random.seed(seed)

    records = []
    years = [2021, 2022, 2023, 2024]

    for kecamatan, kelurahan_list in KECAMATAN_KELURAHAN.items():
        for kelurahan in kelurahan_list:
            # Nilai dasar per kelurahan (konsisten antar tahun + variasi kecil)
            base_balita = np.random.randint(200, 1500)
            base_coverage = np.random.uniform(0.60, 0.95)

            # Beberapa area memiliki tingkat stunting lebih tinggi
            # (simulasi ketimpangan antar kelurahan)
            stunting_factor = np.random.choice(
                [0.4, 0.7, 1.0, 1.5, 2.2],
                p=[0.20, 0.25, 0.25, 0.20, 0.10]
            )

            for tahun in years:
                # Variasi tahunan
                year_noise = 1 + np.random.uniform(-0.10, 0.10)

                jumlah_balita = max(100, int(base_balita * year_noise))

                coverage_noise = base_coverage + np.random.uniform(-0.05, 0.05)
                coverage_noise = np.clip(coverage_noise, 0.50, 1.0)
                jumlah_balita_diukur = max(
                    50, int(jumlah_balita * coverage_noise)
                )
                jumlah_balita_diukur = min(jumlah_balita_diukur, jumlah_balita)

                # Tingkat stunting
                rate_sangat_pendek = np.clip(
                    np.random.uniform(0.01, 0.08) * stunting_factor, 0, 0.20
                )
                rate_pendek = np.clip(
                    np.random.uniform(0.03, 0.15) * stunting_factor, 0, 0.30
                )

                jumlah_sangat_pendek = max(
                    0, int(jumlah_balita_diukur * rate_sangat_pendek)
                )
                jumlah_pendek = max(
                    0, int(jumlah_balita_diukur * rate_pendek)
                )

                # Pastikan total stunting tidak melebihi jumlah yang diukur
                if (jumlah_sangat_pendek + jumlah_pendek) > jumlah_balita_diukur:
                    jumlah_pendek = max(
                        0, jumlah_balita_diukur - jumlah_sangat_pendek
                    )

                records.append({
                    'kode_provinsi': 32,
                    'nama_provinsi': 'Jawa Barat',
                    'kode_kabupaten_kota': 3273,
                    'nama_kabupaten_kota': 'Kota Bandung',
                    'nama_kecamatan': kecamatan,
                    'nama_kelurahan': kelurahan,
                    'jumlah_balita': jumlah_balita,
                    'jumlah_balita_diukur': jumlah_balita_diukur,
                    'jumlah_balita_stunting_sangat_pendek': jumlah_sangat_pendek,
                    'jumlah_balita_stunting_pendek': jumlah_pendek,
                    'tahun': tahun,
                })

    df = pd.DataFrame(records)

    # Buat direktori output jika belum ada
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def load_data(filepath='data/stunting_bandung.csv'):
    """
    Load data dari file CSV.
    Jika file tidak ditemukan, generate dataset sampel secara otomatis.
    """
    if not os.path.exists(filepath):
        print(f"[INFO] File '{filepath}' tidak ditemukan. Membuat dataset sampel...")
        return generate_sample_dataset(filepath)
    return pd.read_csv(filepath)


def create_features_and_label(df, threshold=20.0):
    """
    Hitung fitur turunan dan buat label klasifikasi biner.

    Fitur Input (sesuai permintaan):
        1. jumlah_balita
        2. jumlah_balita_diukur
        3. jumlah_balita_stunting_sangat_pendek
        4. jumlah_balita_stunting_pendek

    Label (Target):
        prevalensi = (sangat_pendek + pendek) / diukur × 100
        - 'Stunting'       jika prevalensi >= threshold (default 20%)
        - 'Tidak Stunting'  jika prevalensi <  threshold

    Parameter threshold 20% berdasarkan standar WHO:
    prevalensi ≥ 20% dianggap masalah kesehatan masyarakat.
    """
    df = df.copy()

    # Hitung kolom turunan
    df['total_stunting'] = (
        df['jumlah_balita_stunting_sangat_pendek'] +
        df['jumlah_balita_stunting_pendek']
    )
    df['prevalensi_stunting'] = np.where(
        df['jumlah_balita_diukur'] > 0,
        (df['total_stunting'] / df['jumlah_balita_diukur']) * 100,
        0
    )
    df['rasio_pengukuran'] = np.where(
        df['jumlah_balita'] > 0,
        (df['jumlah_balita_diukur'] / df['jumlah_balita']) * 100,
        0
    )

    # Label biner
    df['label'] = np.where(
        df['prevalensi_stunting'] >= threshold,
        'Stunting',
        'Tidak Stunting'
    )

    return df


def get_feature_columns():
    """Daftar kolom fitur input untuk model ML."""
    return [
        'jumlah_balita',
        'jumlah_balita_diukur',
        'jumlah_balita_stunting_sangat_pendek',
        'jumlah_balita_stunting_pendek',
    ]


def prepare_ml_data(df, test_size=0.2, random_state=42):
    """
    Siapkan data untuk pelatihan ML:
    - Pilih fitur & target
    - Split train / test
    - Scaling fitur (StandardScaler)

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test
    """
    feature_cols = get_feature_columns()

    X = df[feature_cols].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test


def validate_uploaded_data(df):
    """
    Validasi apakah DataFrame yang di-upload memiliki kolom yang dibutuhkan.
    Returns (is_valid, missing_columns)
    """
    required = get_feature_columns()
    missing = [col for col in required if col not in df.columns]
    return len(missing) == 0, missing
