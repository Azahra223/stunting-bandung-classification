"""
Proses Analisis Data Stunting Kota Bandung
=======================================
Pipeline lengkap: Data Cleaning, Preprocessing, Feature Engineering, EDA,
Pemilihan Algoritma, Training, dan Evaluasi Model.

Data Input:
- jumlah_balita
- jumlah_balita_diukur  
- jumlah_balita_stunting_sangat_pendek
- jumlah_balita_stunting_pendek
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA CLEANING
# ============================================================
def data_cleaning(df):
    """Cleaning data: handle missing values, duplicates, outliers."""
    print("=" * 60)
    print("1. DATA CLEANING")
    print("=" * 60)
    
    df_clean = df.copy()
    
    print(f"Shape sebelum cleaning: {df.shape}")
    print(f"\nMissing values per kolom:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print(f"\nDuplikasi data: {df.duplicated().sum()} baris")
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle missing values - fill dengan median untuk numerik
    numeric_cols = [
        'jumlah_balita', 'jumlah_balita_diukur',
        'jumlah_balita_stunting_sangat_pendek', 'jumlah_balita_stunting_pendek'
    ]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Validasi nilai negatif
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=0)
    
    print(f"Shape setelah cleaning: {df_clean.shape}")
    
    # Statistik outlier detection
    print(f"\nStatistik deskriptif numerik:")
    print(df_clean[numeric_cols].describe())
    
    return df_clean


# ============================================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ============================================================
def feature_engineering(df, threshold=20.0):
    """Buat fitur turunan dan label klasifikasi."""
    print("\n" + "=" * 60)
    print("2. FEATURE ENGINEERING")
    print("=" * 60)
    
    df_fe = df.copy()
    
    # Fitur turunan
    df_fe['total_stunting'] = (
        df_fe['jumlah_balita_stunting_sangat_pendek'] + 
        df_fe['jumlah_balita_stunting_pendek']
    )
    
    df_fe['prevalensi_stunting'] = np.where(
        df_fe['jumlah_balita_diukur'] > 0,
        (df_fe['total_stunting'] / df_fe['jumlah_balita_diukur']) * 100,
        0
    )
    
    df_fe['rasio_pengukuran'] = np.where(
        df_fe['jumlah_balita'] > 0,
        (df_fe['jumlah_balita_diukur'] / df_fe['jumlah_balita']) * 100,
        0
    )
    
    df_fe['rasio_sangat_pendek'] = np.where(
        df_fe['jumlah_balita_diukur'] > 0,
        (df_fe['jumlah_balita_stunting_sangat_pendek'] / df_fe['jumlah_balita_diukur']) * 100,
        0
    )
    
    df_fe['rasio_pendek'] = np.where(
        df_fe['jumlah_balita_diukur'] > 0,
        (df_fe['jumlah_balita_stunting_pendek'] / df_fe['jumlah_balita_diukur']) * 100,
        0
    )
    
    # Label biner
    df_fe['label'] = np.where(
        df_fe['prevalensi_stunting'] >= threshold,
        'Stunting',
        'Tidak Stunting'
    )
    
    label_dist = df_fe['label'].value_counts()
    print(f"\nDistribusi Label (threshold={threshold}%):")
    print(f"  Stunting: {label_dist.get('Stunting', 0)}")
    print(f"  Tidak Stunting: {label_dist.get('Tidak Stunting', 0)}")
    print(f"  Total: {len(df_fe)}")
    
    return df_fe


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
def perform_eda(df):
    """Eksplorasi data: visualisasi dan analisis statistik."""
    print("\n" + "=" * 60)
    print("3. EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    feature_cols = [
        'jumlah_balita', 'jumlah_balita_diukur',
        'jumlah_balita_stunting_sangat_pendek', 'jumlah_balita_stunting_pendek',
        'prevalensi_stunting', 'rasio_pengukuran'
    ]
    
    # Statistik deskriptif per label
    print("\nStatistik deskriptif per kelas:")
    print(df.groupby('label')[feature_cols].describe().T)
    
    # Correlation matrix
    corr_matrix = df[feature_cols].corr()
    print("\nKorelasi antar fitur:")
    print(corr_matrix.round(3))
    
    # Visualisasi
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribusi prevalensi stunting
    axes[0, 0].hist(df['prevalensi_stunting'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(20, color='red', linestyle='--', label='Threshold 20%')
    axes[0, 0].set_xlabel('Prevalensi Stunting (%)')
    axes[0, 0].set_ylabel('Frekuensi')
    axes[0, 0].set_title('Distribusi Prevalensi Stunting')
    axes[0, 0].legend()
    
    # 2. Boxplot prevalensi per label
    df.boxplot(column='prevalensi_stunting', by='label', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Label')
    axes[0, 1].set_ylabel('Prevalensi Stunting (%)')
    axes[0, 1].set_title('Boxplot Prevalensi per Label')
    
    # 3. Scatter plot: sangat pendek vs pendek
    axes[1, 0].scatter(
        df['jumlah_balita_stunting_sangat_pendek'],
        df['jumlah_balita_stunting_pendek'],
        c=df['label'].map({'Stunting': 'red', 'Tidak Stunting': 'green'}),
        alpha=0.6
    )
    axes[1, 0].set_xlabel('Jumlah Sangat Pendek')
    axes[1, 0].set_ylabel('Jumlah Pendek')
    axes[1, 0].set_title('Sangat Pendek vs Pendek')
    
    # 4. Trend tahunan
    yearly = df.groupby('tahun')['prevalensi_stunting'].mean()
    axes[1, 1].plot(yearly.index, yearly.values, marker='o', linewidth=2)
    axes[1, 1].set_xlabel('Tahun')
    axes[1, 1].set_ylabel('Rata-rata Prevalensi (%)')
    axes[1, 1].set_title('Tren Prevalensi Stunting Tahunan')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nVisualisasi EDA disimpan: eda_visualization.png")
    
    return feature_cols


# ============================================================
# 4. PEMILIHAN ALGORITMA & TRAINING
# ============================================================
def train_and_evaluate(df, test_size=0.2, cv_folds=5):
    """Training dan evaluasi berbagai algoritma klasifikasi."""
    print("\n" + "=" * 60)
    print("4. PEMILIHAN ALGORITMA & EVALUASI")
    print("=" * 60)
    
    feature_cols = [
        'jumlah_balita',
        'jumlah_balita_diukur',
        'jumlah_balita_stunting_sangat_pendek',
        'jumlah_balita_stunting_pendek',
    ]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Convert to numpy array (ensure proper format)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nData split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Model definitions
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Training
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='Stunting', zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label='Stunting', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label='Stunting', zero_division=0)
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  CV Mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=['Stunting', 'Tidak Stunting'])
        print(f"\n  Confusion Matrix:")
        print(f"    [[TN, FP], [FN, TP]]")
        print(f"    {cm}")
    
    return results, X_test_scaled, y_test


# ============================================================
# 5. MODEL COMPARISON & BEST SELECTION
# ============================================================
def compare_models(results, X_test_scaled, y_test):
    """Bandingkan performa model dan pilih terbaik."""
    print("\n" + "=" * 60)
    print("5. PERBANDINGAN MODEL")
    print("=" * 60)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [f"{r['accuracy']:.4f}" for r in results.values()],
        'Precision': [f"{r['precision']:.4f}" for r in results.values()],
        'Recall': [f"{r['recall']:.4f}" for r in results.values()],
        'F1-Score': [f"{r['f1']:.4f}" for r in results.values()],
        'CV_Mean': [f"{r['cv_mean']:.4f}" for r in results.values()],
    })
    
    print("\nTabel Perbandingan Model:")
    print(comparison_df.to_string(index=False))
    
    # Best model berdasarkan F1-Score
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    print(f"\n[BEST] Model Terbaik: {best_model_name}")
    print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
    
# ROC Curve
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        if hasattr(res['model'], 'predict_proba'):
            y_proba = res['model'].predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label='Stunting')
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Semua Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return best_model_name, comparison_df


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
def show_feature_importance(model, feature_names, model_name):
    """Tampilkan feature importance."""
    print("\n" + "=" * 60)
    print(f"6. FEATURE IMPORTANCE - {model_name}")
    print("=" * 60)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model tidak memiliki feature importance")
        return
    
    imp_df = pd.DataFrame({
        'Fitur': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(imp_df.to_string(index=False))
    
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df['Fitur'], imp_df['Importance'], color='steelblue')
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/raw/stunting_bandung.csv')
    
    print("DATA STUNTING KOTA BANDUNG")
    print(f"Total records: {len(df)}")
    
    # 1. Data Cleaning
    df_clean = data_cleaning(df)
    
    # 2. Feature Engineering
    df_fe = feature_engineering(df_clean, threshold=20.0)
    
    # 3. EDA
    feature_cols = perform_eda(df_fe)
    
    # 4. Training & Evaluation
    results, X_test_scaled, y_test = train_and_evaluate(df_fe)
    
    # 5. Model Comparison
    best_model, comparison_df = compare_models(results, X_test_scaled, y_test)
    
    # 6. Feature Importance Best Model
    feature_names = [
        'jumlah_balita', 'jumlah_balita_diukur',
        'jumlah_balita_stunting_sangat_pendek', 'jumlah_balita_stunting_pendek'
    ]
    show_feature_importance(results[best_model]['model'], feature_names, best_model)
    
    print("\n" + "=" * 60)
    print("PIPELINE SELESAI")
    print("=" * 60)
    print(f"\nFile output:")
    print("  - eda_visualization.png")
    print("  - roc_curve.png")
    print(f"  - feature_importance_{best_model.lower().replace(' ', '_')}.png")