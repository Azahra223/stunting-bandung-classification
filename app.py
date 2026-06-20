"""
Klasifikasi Balita Stunting Berdasarkan Kelurahan di Kota Bandung
=================================================================
Aplikasi Streamlit untuk klasifikasi tingkat stunting menggunakan
Machine Learning (Logistic Regression, Random Forest, Naive Bayes).

Sumber Data  : Open Data Jabar
Teknologi    : Python · Pandas · Scikit-Learn · Streamlit · Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Tambahkan root project ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_processing import (
    generate_sample_dataset,
    load_data,
    create_features_and_label,
    get_feature_columns,
    prepare_ml_data,
    validate_uploaded_data,
)
from models.model_utils import (
    AVAILABLE_MODELS,
    get_model,
    train_model,
    evaluate_model,
    compare_all_models,
    get_comparison_dataframe,
    predict_single,
    get_feature_importance,
    cross_validate_model,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Stunting — Kota Bandung",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* ---- Import Fonts ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #a78bfa;
    }

    /* ---- Headers ---- */
    h1 { color: #e0e7ff !important; font-weight: 800 !important; }
    h2 { color: #c7d2fe !important; font-weight: 700 !important; }
    h3 { color: #a5b4fc !important; font-weight: 600 !important; }

    /* ---- Metric Cards ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102,126,234,0.12) 0%, rgba(118,75,162,0.12) 100%);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.2);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e0e7ff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* ---- DataFrames ---- */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(102,126,234,0.08);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102,126,234,0.25) 0%, rgba(118,75,162,0.25) 100%);
        color: #e0e7ff !important;
        border-bottom: 2px solid #667eea;
    }

    /* ---- Selectbox / Input ---- */
    div[data-baseweb="select"] > div {
        background-color: rgba(102,126,234,0.08);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 10px;
    }

    /* ---- Info boxes ---- */
    .info-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.10) 0%, rgba(118,75,162,0.10) 100%);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
    }
    .success-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.10) 0%, rgba(5,150,105,0.10) 100%);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(245,158,11,0.10) 0%, rgba(217,119,6,0.10) 100%);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }
    .danger-box {
        background: linear-gradient(135deg, rgba(239,68,68,0.10) 0%, rgba(220,38,38,0.10) 100%);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }

    /* ---- Prediction Result Card ---- */
    .prediction-card {
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
        backdrop-filter: blur(12px);
    }
    .prediction-stunting {
        background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(220,38,38,0.15) 100%);
        border: 2px solid rgba(239,68,68,0.4);
    }
    .prediction-tidak {
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.15) 100%);
        border: 2px solid rgba(16,185,129,0.4);
    }

    /* ---- Divider ---- */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(102,126,234,0.3) 50%, transparent 100%);
        margin: 2rem 0;
    }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PLOTLY THEME
# ============================================================
PLOTLY_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#c7d2fe'),
    title_font=dict(size=18, color='#e0e7ff'),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(102,126,234,0.2)',
        borderwidth=1,
        font=dict(color='#a5b4fc'),
    ),
    xaxis=dict(
        gridcolor='rgba(102,126,234,0.08)',
        zerolinecolor='rgba(102,126,234,0.15)',
    ),
    yaxis=dict(
        gridcolor='rgba(102,126,234,0.08)',
        zerolinecolor='rgba(102,126,234,0.15)',
    ),
    margin=dict(l=40, r=40, t=60, b=40),
)

COLOR_PALETTE = [
    '#667eea', '#764ba2', '#f093fb', '#4fd1c5',
    '#f6ad55', '#fc8181', '#63b3ed', '#68d391',
    '#b794f4', '#feb2b2',
]
COLOR_STUNTING = '#ef4444'
COLOR_TIDAK = '#10b981'


# ============================================================
# DATA LOADING (cached)
# ============================================================
@st.cache_data
def load_and_process(filepath='data/stunting_bandung.csv', threshold=20.0):
    """Load data dan buat fitur + label."""
    df = load_data(filepath)
    df = create_features_and_label(df, threshold=threshold)
    return df


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🏥 Stunting Bandung")
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.85rem;'>"
        "Klasifikasi Balita Stunting<br>Berdasarkan Kelurahan<br>Kota Bandung"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "📌 Navigasi",
        [
            "🏠 Dashboard",
            "📊 Eksplorasi Data",
            "🤖 Model & Evaluasi",
            "🔮 Prediksi",
            "📁 Upload Data",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#64748b; font-size:0.75rem;'>"
        "Sumber: Open Data Jabar<br>"
        "Model: Logistic Regression,<br>"
        "Random Forest, Naive Bayes<br><br>"
        "© 2024 — Klasifikasi Stunting"
        "</p>",
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD DATA
# ============================================================
df = load_and_process()


# ====================================================================
# PAGE 1: DASHBOARD
# ====================================================================
if page == "🏠 Dashboard":
    st.markdown("# 🏠 Dashboard Stunting Kota Bandung")
    st.markdown(
        "<p style='color:#94a3b8; margin-top:-10px;'>"
        "Overview data balita stunting berdasarkan kelurahan di Kota Bandung"
        "</p>",
        unsafe_allow_html=True,
    )

    # ---- Tahun filter ----
    years = sorted(df['tahun'].unique())
    selected_year = st.selectbox(
        "📅 Pilih Tahun", years, index=len(years) - 1, key="dash_year"
    )
    df_year = df[df['tahun'] == selected_year]

    # ---- KPI Cards ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏘️ Jumlah Kelurahan", f"{df_year['nama_kelurahan'].nunique()}")
    with col2:
        st.metric("👶 Total Balita", f"{df_year['jumlah_balita'].sum():,}")
    with col3:
        st.metric("📏 Total Diukur", f"{df_year['jumlah_balita_diukur'].sum():,}")
    with col4:
        st.metric(
            "⚠️ Total Stunting",
            f"{df_year['total_stunting'].sum():,}",
        )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_prev = df_year['prevalensi_stunting'].mean()
        st.metric("📊 Rata-rata Prevalensi", f"{avg_prev:.1f}%")
    with col2:
        stunting_count = (df_year['label'] == 'Stunting').sum()
        st.metric("🔴 Kelurahan Stunting", f"{stunting_count}")
    with col3:
        tidak_count = (df_year['label'] == 'Tidak Stunting').sum()
        st.metric("🟢 Kelurahan Tidak Stunting", f"{tidak_count}")
    with col4:
        max_prev = df_year['prevalensi_stunting'].max()
        st.metric("🔺 Prevalensi Tertinggi", f"{max_prev:.1f}%")

    st.markdown("---")

    # ---- Charts ----
    c1, c2 = st.columns(2)

    with c1:
        # Pie chart: distribusi label
        label_counts = df_year['label'].value_counts().reset_index()
        label_counts.columns = ['Klasifikasi', 'Jumlah']
        fig_pie = px.pie(
            label_counts,
            values='Jumlah',
            names='Klasifikasi',
            title='Distribusi Klasifikasi Kelurahan',
            color='Klasifikasi',
            color_discrete_map={
                'Stunting': COLOR_STUNTING,
                'Tidak Stunting': COLOR_TIDAK,
            },
            hole=0.45,
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT)
        fig_pie.update_traces(
            textposition='outside',
            textinfo='label+percent+value',
            textfont_size=12,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        # Bar chart: top 10 kelurahan prevalensi tertinggi
        top10 = df_year.nlargest(10, 'prevalensi_stunting')[
            ['nama_kelurahan', 'nama_kecamatan', 'prevalensi_stunting']
        ]
        fig_bar = px.bar(
            top10,
            x='prevalensi_stunting',
            y='nama_kelurahan',
            orientation='h',
            title='Top 10 Kelurahan — Prevalensi Stunting Tertinggi',
            color='prevalensi_stunting',
            color_continuous_scale=['#667eea', '#ef4444'],
            text='prevalensi_stunting',
            hover_data=['nama_kecamatan'],
        )
        fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        fig_bar.update_yaxes(autorange='reversed')
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bar.update_coloraxes(showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Trend chart (multi-year) ----
    st.markdown("### 📈 Tren Tahunan Stunting")
    trend = df.groupby('tahun').agg(
        total_balita=('jumlah_balita', 'sum'),
        total_diukur=('jumlah_balita_diukur', 'sum'),
        total_stunting=('total_stunting', 'sum'),
        avg_prevalensi=('prevalensi_stunting', 'mean'),
    ).reset_index()

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(
            x=trend['tahun'], y=trend['total_stunting'],
            name='Total Stunting',
            marker_color='rgba(239,68,68,0.7)',
            text=trend['total_stunting'],
            textposition='outside',
        ),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Scatter(
            x=trend['tahun'], y=trend['avg_prevalensi'],
            name='Rata-rata Prevalensi (%)',
            mode='lines+markers+text',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10),
            text=[f'{v:.1f}%' for v in trend['avg_prevalensi']],
            textposition='top center',
            textfont=dict(color='#a5b4fc'),
        ),
        secondary_y=True,
    )
    fig_trend.update_layout(
        **PLOTLY_LAYOUT,
        title='Tren Total Stunting & Prevalensi per Tahun',
        barmode='group',
    )
    fig_trend.update_yaxes(
        title_text="Total Stunting", secondary_y=False,
        gridcolor='rgba(102,126,234,0.08)',
    )
    fig_trend.update_yaxes(
        title_text="Prevalensi (%)", secondary_y=True,
        gridcolor='rgba(102,126,234,0.08)',
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ---- Per-Kecamatan Summary ----
    st.markdown("### 🗺️ Ringkasan per Kecamatan")
    kec_summary = df_year.groupby('nama_kecamatan').agg(
        jumlah_kelurahan=('nama_kelurahan', 'nunique'),
        total_balita=('jumlah_balita', 'sum'),
        total_diukur=('jumlah_balita_diukur', 'sum'),
        total_stunting=('total_stunting', 'sum'),
        avg_prevalensi=('prevalensi_stunting', 'mean'),
        kelurahan_stunting=('label', lambda x: (x == 'Stunting').sum()),
    ).reset_index().sort_values('avg_prevalensi', ascending=False)

    fig_kec = px.bar(
        kec_summary,
        x='nama_kecamatan',
        y='avg_prevalensi',
        title=f'Rata-rata Prevalensi Stunting per Kecamatan ({selected_year})',
        color='avg_prevalensi',
        color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
        text='avg_prevalensi',
    )
    fig_kec.update_layout(**PLOTLY_LAYOUT, showlegend=False, xaxis_tickangle=-45)
    fig_kec.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_kec.update_coloraxes(showscale=False)
    st.plotly_chart(fig_kec, use_container_width=True)


# ====================================================================
# PAGE 2: EKSPLORASI DATA
# ====================================================================
elif page == "📊 Eksplorasi Data":
    st.markdown("# 📊 Eksplorasi Data")
    st.markdown(
        "<p style='color:#94a3b8; margin-top:-10px;'>"
        "Jelajahi dataset stunting Kota Bandung secara interaktif"
        "</p>",
        unsafe_allow_html=True,
    )

    # ---- Filters ----
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        years = sorted(df['tahun'].unique())
        sel_years = st.multiselect("📅 Tahun", years, default=years, key="exp_years")
    with fc2:
        kecamatans = sorted(df['nama_kecamatan'].unique())
        sel_kec = st.multiselect("🏘️ Kecamatan", kecamatans, default=[], key="exp_kec")
    with fc3:
        sel_label = st.multiselect(
            "🏷️ Label", ['Stunting', 'Tidak Stunting'], default=['Stunting', 'Tidak Stunting'],
            key="exp_label",
        )

    # Apply filters
    df_filtered = df[df['tahun'].isin(sel_years)]
    if sel_kec:
        df_filtered = df_filtered[df_filtered['nama_kecamatan'].isin(sel_kec)]
    df_filtered = df_filtered[df_filtered['label'].isin(sel_label)]

    st.markdown(f"**Menampilkan {len(df_filtered):,} baris data**")

    # ---- Data Table ----
    tab1, tab2, tab3 = st.tabs(["📋 Tabel Data", "📊 Distribusi", "🔗 Korelasi"])

    with tab1:
        display_cols = [
            'nama_kecamatan', 'nama_kelurahan', 'tahun',
            'jumlah_balita', 'jumlah_balita_diukur',
            'jumlah_balita_stunting_sangat_pendek',
            'jumlah_balita_stunting_pendek',
            'prevalensi_stunting', 'label',
        ]
        st.dataframe(
            df_filtered[display_cols].style.format({
                'prevalensi_stunting': '{:.2f}%',
            }),
            use_container_width=True,
            height=500,
        )

        # Statistics
        st.markdown("#### 📈 Statistik Deskriptif")
        numeric_cols = [
            'jumlah_balita', 'jumlah_balita_diukur',
            'jumlah_balita_stunting_sangat_pendek',
            'jumlah_balita_stunting_pendek',
            'prevalensi_stunting',
        ]
        st.dataframe(
            df_filtered[numeric_cols].describe().T.style.format('{:.2f}'),
            use_container_width=True,
        )

    with tab2:
        st.markdown("#### Distribusi Fitur")

        d1, d2 = st.columns(2)

        with d1:
            fig_hist = px.histogram(
                df_filtered,
                x='prevalensi_stunting',
                color='label',
                nbins=30,
                title='Distribusi Prevalensi Stunting',
                color_discrete_map={
                    'Stunting': COLOR_STUNTING,
                    'Tidak Stunting': COLOR_TIDAK,
                },
                barmode='overlay',
                opacity=0.75,
            )
            fig_hist.update_layout(**PLOTLY_LAYOUT)
            fig_hist.add_vline(
                x=20, line_dash="dash", line_color="#f59e0b",
                annotation_text="Threshold 20%",
                annotation_font_color="#f59e0b",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with d2:
            fig_box = px.box(
                df_filtered,
                y='prevalensi_stunting',
                x='label',
                color='label',
                title='Boxplot Prevalensi per Label',
                color_discrete_map={
                    'Stunting': COLOR_STUNTING,
                    'Tidak Stunting': COLOR_TIDAK,
                },
                points='outliers',
            )
            fig_box.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_box, use_container_width=True)

        # Histogram per feature
        feat_select = st.selectbox(
            "Pilih fitur untuk histogram",
            get_feature_columns(),
            key="hist_feat",
        )
        fig_feat = px.histogram(
            df_filtered,
            x=feat_select,
            color='label',
            nbins=30,
            title=f'Distribusi: {feat_select}',
            color_discrete_map={
                'Stunting': COLOR_STUNTING,
                'Tidak Stunting': COLOR_TIDAK,
            },
            barmode='overlay',
            opacity=0.7,
        )
        fig_feat.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_feat, use_container_width=True)

    with tab3:
        st.markdown("#### Matriks Korelasi Fitur")
        corr_cols = numeric_cols
        corr_matrix = df_filtered[corr_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title='Heatmap Korelasi antar Fitur',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
        )
        fig_corr.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Scatter plot
        st.markdown("#### Scatter Plot")
        sc1, sc2 = st.columns(2)
        with sc1:
            x_feat = st.selectbox("Sumbu X", numeric_cols, index=0, key="sc_x")
        with sc2:
            y_feat = st.selectbox("Sumbu Y", numeric_cols, index=4, key="sc_y")

        fig_scatter = px.scatter(
            df_filtered,
            x=x_feat, y=y_feat,
            color='label',
            color_discrete_map={
                'Stunting': COLOR_STUNTING,
                'Tidak Stunting': COLOR_TIDAK,
            },
            title=f'{x_feat} vs {y_feat}',
            opacity=0.65,
            hover_data=['nama_kelurahan', 'nama_kecamatan'],
        )
        fig_scatter.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)


# ====================================================================
# PAGE 3: MODEL & EVALUASI
# ====================================================================
elif page == "🤖 Model & Evaluasi":
    st.markdown("# 🤖 Model & Evaluasi")
    st.markdown(
        "<p style='color:#94a3b8; margin-top:-10px;'>"
        "Latih, evaluasi, dan bandingkan model klasifikasi Machine Learning"
        "</p>",
        unsafe_allow_html=True,
    )

    # ---- Settings ----
    st.markdown("### ⚙️ Pengaturan")
    s1, s2, s3 = st.columns(3)
    with s1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5, key="ts") / 100
    with s2:
        threshold = st.slider("Threshold Stunting (%)", 10, 40, 20, 1, key="thresh")
    with s3:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5, 1, key="cv")

    # Re-process with updated threshold
    df_ml = create_features_and_label(
        load_data(), threshold=float(threshold)
    )

    # Data split info
    label_dist = df_ml['label'].value_counts()
    lc1, lc2 = st.columns(2)
    with lc1:
        st.info(
            f"📊 **Distribusi Label** — "
            f"Stunting: {label_dist.get('Stunting', 0)} | "
            f"Tidak Stunting: {label_dist.get('Tidak Stunting', 0)} | "
            f"Total: {len(df_ml)}"
        )
    with lc2:
        st.info(
            f"📐 **Split Data** — "
            f"Train: {int(len(df_ml) * (1 - test_size))} | "
            f"Test: {int(len(df_ml) * test_size)} | "
            f"CV Folds: {cv_folds}"
        )

    st.markdown("---")

    # ---- Model Selection ----
    selected_models = st.multiselect(
        "🎯 Pilih Model untuk Dilatih",
        list(AVAILABLE_MODELS.keys()),
        default=list(AVAILABLE_MODELS.keys()),
        key="models",
    )

    # Model descriptions
    for name in selected_models:
        info = AVAILABLE_MODELS[name]
        st.markdown(
            f"<div class='info-box'>"
            f"<strong style='color:#a78bfa;'>📘 {name}</strong><br>"
            f"<span style='color:#94a3b8;'>{info['description']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if st.button("🚀 Latih Semua Model", use_container_width=True, key="train_btn"):
        if not selected_models:
            st.error("Pilih minimal satu model!")
        else:
            with st.spinner("⏳ Melatih model..."):
                # Prepare data
                X_train, X_test, y_train, y_test, scaler, _, _ = prepare_ml_data(
                    df_ml, test_size=test_size
                )

                # Store in session state
                st.session_state['scaler'] = scaler
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test

                results = {}
                for name in selected_models:
                    model = get_model(name)
                    train_model(model, X_train, y_train)
                    eval_res = evaluate_model(model, X_test, y_test)
                    cv_res = cross_validate_model(
                        get_model(name), X_train, y_train, cv=cv_folds
                    )
                    results[name] = {
                        'model': model,
                        'test_results': eval_res,
                        'cv_results': cv_res,
                    }

                st.session_state['trained_models'] = results

            st.success("✅ Semua model berhasil dilatih!")

    # ---- Display results ----
    if 'trained_models' in st.session_state:
        results = st.session_state['trained_models']

        st.markdown("---")
        st.markdown("### 📊 Perbandingan Model")

        comp_df = get_comparison_dataframe(results)
        st.dataframe(
            comp_df.set_index('Model'),
            use_container_width=True,
        )

        # Bar chart comparison
        metrics_data = []
        for name, data in results.items():
            r = data['test_results']
            for metric_name, value in [
                ('Accuracy', r['accuracy']),
                ('Precision', r['precision']),
                ('Recall', r['recall']),
                ('F1-Score', r['f1']),
            ]:
                metrics_data.append({
                    'Model': name,
                    'Metrik': metric_name,
                    'Nilai': value,
                })
        metrics_df = pd.DataFrame(metrics_data)

        fig_comp = px.bar(
            metrics_df,
            x='Metrik', y='Nilai',
            color='Model',
            barmode='group',
            title='Perbandingan Performa Model',
            color_discrete_sequence=COLOR_PALETTE,
            text='Nilai',
        )
        fig_comp.update_layout(**PLOTLY_LAYOUT)
        fig_comp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_comp.update_yaxes(range=[0, 1.15])
        st.plotly_chart(fig_comp, use_container_width=True)

        # ---- Per-model details ----
        st.markdown("---")
        st.markdown("### 🔍 Detail per Model")

        model_tabs = st.tabs([f"📘 {name}" for name in results])

        for i, (name, data) in enumerate(results.items()):
            with model_tabs[i]:
                r = data['test_results']
                cv = data['cv_results']

                # Metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Accuracy", f"{r['accuracy']:.4f}")
                with m2:
                    st.metric("Precision", f"{r['precision']:.4f}")
                with m3:
                    st.metric("Recall", f"{r['recall']:.4f}")
                with m4:
                    st.metric("F1-Score", f"{r['f1']:.4f}")
                with m5:
                    st.metric("CV Mean", f"{cv['mean_score']:.4f}")

                det1, det2 = st.columns(2)

                with det1:
                    # Confusion Matrix
                    cm = r['confusion_matrix']
                    labels = ['Stunting', 'Tidak Stunting']
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        x=labels, y=labels,
                        title=f'Confusion Matrix — {name}',
                        color_continuous_scale='Blues',
                        labels=dict(x="Prediksi", y="Aktual"),
                    )
                    fig_cm.update_layout(**PLOTLY_LAYOUT)
                    fig_cm.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_cm, use_container_width=True)

                with det2:
                    # Feature Importance
                    model_obj = data['model']
                    feat_imp = get_feature_importance(
                        model_obj, get_feature_columns()
                    )
                    if feat_imp is not None:
                        fig_imp = px.bar(
                            feat_imp,
                            x='Importance', y='Fitur',
                            orientation='h',
                            title=f'Feature Importance — {name}',
                            color='Importance',
                            color_continuous_scale=['#667eea', '#f093fb'],
                            text='Importance',
                        )
                        fig_imp.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                        fig_imp.update_traces(
                            texttemplate='%{text:.4f}', textposition='outside'
                        )
                        fig_imp.update_coloraxes(showscale=False)
                        st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.info(
                            f"Feature importance tidak tersedia untuk {name}."
                        )

                # ROC Curve
                if 'roc_auc' in r:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=r['fpr'], y=r['tpr'],
                        mode='lines',
                        name=f'{name} (AUC={r["roc_auc"]:.4f})',
                        line=dict(color='#667eea', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(102,126,234,0.1)',
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random',
                        line=dict(color='#64748b', dash='dash'),
                    ))
                    fig_roc.update_layout(
                        **PLOTLY_LAYOUT,
                        title=f'ROC Curve — {name}',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                # Classification Report
                st.markdown(f"**Classification Report — {name}**")
                report = r['classification_report']
                report_df = pd.DataFrame(report).T
                st.dataframe(
                    report_df.style.format('{:.4f}', na_rep='-'),
                    use_container_width=True,
                )

        # ---- Best model ----
        st.markdown("---")
        best_name = max(
            results,
            key=lambda n: results[n]['test_results']['f1'],
        )
        best_f1 = results[best_name]['test_results']['f1']
        st.markdown(
            f"<div class='success-box'>"
            f"<h3 style='color:#10b981; margin:0;'>🏆 Model Terbaik: {best_name}</h3>"
            f"<p style='color:#a7f3d0; margin:5px 0 0 0;'>"
            f"F1-Score: <strong>{best_f1:.4f}</strong> — "
            f"Model ini dipilih otomatis untuk halaman Prediksi.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.session_state['best_model_name'] = best_name


# ====================================================================
# PAGE 4: PREDIKSI
# ====================================================================
elif page == "🔮 Prediksi":
    st.markdown("# 🔮 Prediksi Klasifikasi Stunting")
    st.markdown(
        "<p style='color:#94a3b8; margin-top:-10px;'>"
        "Masukkan data kelurahan untuk memprediksi klasifikasi stunting"
        "</p>",
        unsafe_allow_html=True,
    )

    if 'trained_models' not in st.session_state:
        st.warning(
            "⚠️ Belum ada model yang dilatih. Silakan ke halaman "
            "**🤖 Model & Evaluasi** terlebih dahulu untuk melatih model."
        )
    else:
        results = st.session_state['trained_models']
        scaler = st.session_state['scaler']

        # Model selection
        best_name = st.session_state.get('best_model_name', list(results.keys())[0])
        model_names = list(results.keys())
        default_idx = model_names.index(best_name) if best_name in model_names else 0

        selected_model_name = st.selectbox(
            "🎯 Pilih Model untuk Prediksi",
            model_names,
            index=default_idx,
            key="pred_model",
        )
        model = results[selected_model_name]['model']

        st.markdown("---")
        st.markdown("### 📝 Input Data Kelurahan")

        # Input form
        f1, f2 = st.columns(2)
        with f1:
            jumlah_balita = st.number_input(
                "👶 Jumlah Balita",
                min_value=0, max_value=5000, value=500, step=10,
                help="Total jumlah balita di kelurahan",
                key="inp_balita",
            )
            jumlah_sangat_pendek = st.number_input(
                "📏 Jumlah Balita TB Sangat Pendek",
                min_value=0, max_value=2000, value=30, step=1,
                help="Jumlah balita yang diukur tinggi badan sangat pendek",
                key="inp_sp",
            )
        with f2:
            jumlah_diukur = st.number_input(
                "📐 Jumlah Balita Diukur",
                min_value=0, max_value=5000, value=400, step=10,
                help="Jumlah balita yang diukur tinggi badannya",
                key="inp_diukur",
            )
            jumlah_pendek = st.number_input(
                "📏 Jumlah Balita TB Pendek",
                min_value=0, max_value=2000, value=50, step=1,
                help="Jumlah balita yang diukur tinggi badan pendek",
                key="inp_p",
            )

        # Validasi input
        input_valid = True
        warnings = []
        if jumlah_diukur > jumlah_balita:
            warnings.append("⚠️ Jumlah balita diukur melebihi total balita.")
            input_valid = False
        if (jumlah_sangat_pendek + jumlah_pendek) > jumlah_diukur:
            warnings.append("⚠️ Total stunting melebihi jumlah balita diukur.")
            input_valid = False

        for w in warnings:
            st.warning(w)

        # Preview
        if jumlah_diukur > 0:
            preview_prev = ((jumlah_sangat_pendek + jumlah_pendek) / jumlah_diukur) * 100
        else:
            preview_prev = 0

        st.markdown(
            f"<div class='info-box'>"
            f"<strong style='color:#a78bfa;'>📊 Preview</strong><br>"
            f"<span style='color:#94a3b8;'>"
            f"Total Stunting: <strong>{jumlah_sangat_pendek + jumlah_pendek}</strong> | "
            f"Prevalensi: <strong>{preview_prev:.1f}%</strong> | "
            f"Rasio Pengukuran: <strong>"
            f"{(jumlah_diukur / jumlah_balita * 100) if jumlah_balita > 0 else 0:.1f}%"
            f"</strong>"
            f"</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("")

        if st.button(
            "🔮 Prediksi Sekarang", use_container_width=True,
            disabled=not input_valid, key="predict_btn",
        ):
            input_data = [
                jumlah_balita,
                jumlah_diukur,
                jumlah_sangat_pendek,
                jumlah_pendek,
            ]

            prediction, probability = predict_single(model, scaler, input_data)

            # Result display
            if prediction == 'Stunting':
                card_class = 'prediction-stunting'
                icon = '🔴'
                color = '#ef4444'
            else:
                card_class = 'prediction-tidak'
                icon = '🟢'
                color = '#10b981'

            st.markdown(
                f"<div class='prediction-card {card_class}'>"
                f"<h1 style='color:{color}; font-size:3rem; margin:0;'>{icon} {prediction}</h1>"
                f"<p style='color:#c7d2fe; font-size:1.1rem; margin-top:10px;'>"
                f"Model: <strong>{selected_model_name}</strong> | "
                f"Prevalensi Hitung: <strong>{preview_prev:.1f}%</strong>"
                f"</p></div>",
                unsafe_allow_html=True,
            )

            # Probability bar
            if probability:
                st.markdown("#### 📊 Probabilitas Prediksi")
                prob_df = pd.DataFrame([
                    {'Kelas': k, 'Probabilitas': v} for k, v in probability.items()
                ])
                fig_prob = px.bar(
                    prob_df,
                    x='Kelas', y='Probabilitas',
                    color='Kelas',
                    color_discrete_map={
                        'Stunting': COLOR_STUNTING,
                        'Tidak Stunting': COLOR_TIDAK,
                    },
                    text='Probabilitas',
                    title='Distribusi Probabilitas Prediksi',
                )
                fig_prob.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                fig_prob.update_traces(
                    texttemplate='%{text:.2%}', textposition='outside'
                )
                fig_prob.update_yaxes(range=[0, 1.15])
                st.plotly_chart(fig_prob, use_container_width=True)

        # ---- Batch Prediction ----
        st.markdown("---")
        st.markdown("### 📋 Prediksi Batch (Multi Data)")
        st.markdown(
            "<p style='color:#94a3b8;'>"
            "Upload file CSV dengan kolom yang sesuai untuk prediksi massal."
            "</p>",
            unsafe_allow_html=True,
        )

        batch_file = st.file_uploader(
            "Upload CSV untuk prediksi batch",
            type=['csv'],
            key="batch_upload",
        )

        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            is_valid, missing = validate_uploaded_data(batch_df)

            if is_valid:
                feature_cols = get_feature_columns()
                X_batch = batch_df[feature_cols].values
                X_batch_scaled = scaler.transform(X_batch)
                batch_predictions = model.predict(X_batch_scaled)
                batch_df['prediksi'] = batch_predictions

                if hasattr(model, 'predict_proba'):
                    classes = list(model.classes_)
                    if 'Stunting' in classes:
                        pos_idx = classes.index('Stunting')
                        batch_df['probabilitas_stunting'] = model.predict_proba(
                            X_batch_scaled
                        )[:, pos_idx]

                st.success(f"✅ Prediksi berhasil untuk {len(batch_df)} data!")
                st.dataframe(batch_df, use_container_width=True)

                # Summary
                pred_counts = batch_df['prediksi'].value_counts()
                st.markdown(
                    f"**Ringkasan:** Stunting: {pred_counts.get('Stunting', 0)} | "
                    f"Tidak Stunting: {pred_counts.get('Tidak Stunting', 0)}"
                )
            else:
                st.error(f"❌ Kolom berikut tidak ditemukan: {', '.join(missing)}")


# ====================================================================
# PAGE 5: UPLOAD DATA
# ====================================================================
elif page == "📁 Upload Data":
    st.markdown("# 📁 Upload Data Sendiri")
    st.markdown(
        "<p style='color:#94a3b8; margin-top:-10px;'>"
        "Upload dataset CSV Anda sendiri untuk analisis dan klasifikasi"
        "</p>",
        unsafe_allow_html=True,
    )

    # Format info
    st.markdown(
        "<div class='info-box'>"
        "<strong style='color:#a78bfa;'>📋 Format CSV yang Dibutuhkan</strong><br>"
        "<span style='color:#94a3b8;'>"
        "File CSV harus memiliki minimal kolom berikut:</span>"
        "<ul style='color:#c7d2fe; margin-top:8px;'>"
        "<li><code>jumlah_balita</code> — Jumlah total balita</li>"
        "<li><code>jumlah_balita_diukur</code> — Jumlah balita yang diukur</li>"
        "<li><code>jumlah_balita_stunting_sangat_pendek</code> — TB sangat pendek</li>"
        "<li><code>jumlah_balita_stunting_pendek</code> — TB pendek</li>"
        "</ul>"
        "<span style='color:#94a3b8;'>"
        "Kolom tambahan (nama_kecamatan, nama_kelurahan, tahun) opsional.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "📎 Upload File CSV",
        type=['csv'],
        key="main_upload",
    )

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            st.success(f"✅ File berhasil dimuat: {len(df_upload)} baris, {len(df_upload.columns)} kolom")

            # Validate
            is_valid, missing = validate_uploaded_data(df_upload)

            if not is_valid:
                st.error(
                    f"❌ Kolom yang diperlukan tidak ditemukan: {', '.join(missing)}"
                )
                st.markdown("**Kolom yang tersedia dalam file Anda:**")
                st.code(", ".join(df_upload.columns.tolist()))
            else:
                st.markdown("### 👀 Preview Data")
                st.dataframe(df_upload.head(20), use_container_width=True)

                # Process
                st.markdown("---")
                threshold_up = st.slider(
                    "Threshold Stunting (%)",
                    10, 40, 20, 1, key="up_thresh",
                )
                df_processed = create_features_and_label(
                    df_upload, threshold=float(threshold_up)
                )

                # Stats
                st.markdown("### 📈 Statistik Data Anda")
                uc1, uc2, uc3, uc4 = st.columns(4)
                with uc1:
                    st.metric("📋 Total Data", len(df_processed))
                with uc2:
                    st.metric("👶 Total Balita", f"{df_processed['jumlah_balita'].sum():,}")
                with uc3:
                    stunting_n = (df_processed['label'] == 'Stunting').sum()
                    st.metric("🔴 Stunting", stunting_n)
                with uc4:
                    tidak_n = (df_processed['label'] == 'Tidak Stunting').sum()
                    st.metric("🟢 Tidak Stunting", tidak_n)

                # Distribution
                fig_up_pie = px.pie(
                    df_processed['label'].value_counts().reset_index(),
                    values='count', names='label',
                    title='Distribusi Label pada Data Upload',
                    color='label',
                    color_discrete_map={
                        'Stunting': COLOR_STUNTING,
                        'Tidak Stunting': COLOR_TIDAK,
                    },
                    hole=0.45,
                )
                fig_up_pie.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_up_pie, use_container_width=True)

                # Train model on uploaded data
                st.markdown("---")
                st.markdown("### 🤖 Training Model dengan Data Upload")

                if st.button(
                    "🚀 Latih Model dengan Data Ini",
                    use_container_width=True,
                    key="up_train_btn",
                ):
                    with st.spinner("⏳ Melatih model..."):
                        X_train, X_test, y_train, y_test, scaler, _, _ = prepare_ml_data(
                            df_processed, test_size=0.2
                        )

                        st.session_state['scaler'] = scaler

                        up_results = {}
                        for name in AVAILABLE_MODELS:
                            m = get_model(name)
                            train_model(m, X_train, y_train)
                            eval_r = evaluate_model(m, X_test, y_test)
                            cv_r = cross_validate_model(
                                get_model(name), X_train, y_train, cv=5
                            )
                            up_results[name] = {
                                'model': m,
                                'test_results': eval_r,
                                'cv_results': cv_r,
                            }

                        st.session_state['trained_models'] = up_results

                    st.success("✅ Model berhasil dilatih dengan data Anda!")

                    comp = get_comparison_dataframe(up_results)
                    st.dataframe(comp.set_index('Model'), use_container_width=True)

                    best = max(
                        up_results,
                        key=lambda n: up_results[n]['test_results']['f1'],
                    )
                    st.session_state['best_model_name'] = best
                    st.markdown(
                        f"<div class='success-box'>"
                        f"<h3 style='color:#10b981; margin:0;'>"
                        f"🏆 Model Terbaik: {best}</h3>"
                        f"<p style='color:#a7f3d0;'>"
                        f"F1-Score: {up_results[best]['test_results']['f1']:.4f}"
                        f"</p></div>",
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"❌ Error saat membaca file: {str(e)}")

    # ---- Download sample dataset ----
    st.markdown("---")
    st.markdown("### 📥 Download Dataset Sampel")
    st.markdown(
        "<p style='color:#94a3b8;'>"
        "Belum punya dataset? Download dataset sampel Kota Bandung di bawah ini "
        "sebagai referensi format atau untuk latihan."
        "</p>",
        unsafe_allow_html=True,
    )

    if st.button("📥 Generate & Download Dataset Sampel", key="gen_sample"):
        sample_df = load_data()
        csv_data = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download stunting_bandung.csv",
            data=csv_data,
            file_name="stunting_bandung.csv",
            mime="text/csv",
            key="dl_sample",
        )
        st.success(f"✅ Dataset sampel siap: {len(sample_df)} baris data")
        st.dataframe(sample_df.head(10), use_container_width=True)
