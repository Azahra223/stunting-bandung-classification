"""
Model Utilities — Klasifikasi Stunting Kota Bandung
====================================================
Training, evaluasi, dan perbandingan model Machine Learning
untuk klasifikasi biner: Stunting vs Tidak Stunting.

Model yang tersedia:
  1. Logistic Regression
  2. Random Forest Classifier
  3. Naive Bayes (Gaussian)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score


# ============================================================
# Registry model yang tersedia
# ============================================================
AVAILABLE_MODELS = {
    'Logistic Regression': {
        'class': LogisticRegression,
        'default_params': {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs',
        },
        'description': (
            'Model linear yang memprediksi probabilitas kelas '
            'menggunakan fungsi logistik (sigmoid). Cocok untuk '
            'klasifikasi biner dengan fitur numerik.'
        ),
    },
    'Random Forest': {
        'class': RandomForestClassifier,
        'default_params': {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
        },
        'description': (
            'Ensemble dari banyak Decision Tree yang digabungkan '
            'melalui voting mayoritas. Robust terhadap overfitting '
            'dan mampu menangkap pola non-linear.'
        ),
    },
    'Naive Bayes': {
        'class': GaussianNB,
        'default_params': {},
        'description': (
            'Model probabilistik berdasarkan teorema Bayes dengan '
            'asumsi independensi antar fitur. Cepat, sederhana, dan '
            'efektif untuk baseline klasifikasi.'
        ),
    },
}


def get_model(model_name, **custom_params):
    """
    Buat instance model berdasarkan nama.

    Parameters
    ----------
    model_name : str
        Nama model (key di AVAILABLE_MODELS).
    **custom_params
        Parameter tambahan untuk override default.

    Returns
    -------
    model : sklearn estimator
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' tidak tersedia. "
            f"Pilihan: {list(AVAILABLE_MODELS.keys())}"
        )

    model_info = AVAILABLE_MODELS[model_name]
    params = {**model_info['default_params'], **custom_params}
    return model_info['class'](**params)


def train_model(model, X_train, y_train):
    """Latih model dan kembalikan model yang sudah dilatih."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluasi model pada data test.

    Returns
    -------
    results : dict
        accuracy, precision, recall, f1, confusion_matrix,
        classification_report, y_pred
    """
    y_pred = model.predict(X_test)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(
            y_test, y_pred, pos_label='Stunting', zero_division=0
        ),
        'recall': recall_score(
            y_test, y_pred, pos_label='Stunting', zero_division=0
        ),
        'f1': f1_score(
            y_test, y_pred, pos_label='Stunting', zero_division=0
        ),
        'confusion_matrix': confusion_matrix(
            y_test, y_pred, labels=['Stunting', 'Tidak Stunting']
        ),
        'classification_report': classification_report(
            y_test, y_pred, labels=['Stunting', 'Tidak Stunting'],
            output_dict=True, zero_division=0
        ),
        'y_pred': y_pred,
    }

    # Probabilitas prediksi (untuk ROC curve) — jika model mendukung
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        # Ambil probabilitas kelas positif ('Stunting')
        classes = list(model.classes_)
        if 'Stunting' in classes:
            pos_idx = classes.index('Stunting')
            results['y_proba'] = y_proba[:, pos_idx]

            fpr, tpr, _ = roc_curve(
                y_test, y_proba[:, pos_idx], pos_label='Stunting'
            )
            results['roc_auc'] = auc(fpr, tpr)
            results['fpr'] = fpr
            results['tpr'] = tpr

    return results


def cross_validate_model(model, X, y, cv=5):
    """
    Lakukan cross-validation dan kembalikan skor rata-rata.

    Returns
    -------
    cv_results : dict
        mean_score, std_score, scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'scores': scores,
    }


def compare_all_models(X_train, y_train, X_test, y_test, cv=5):
    """
    Latih dan evaluasi semua model yang tersedia, lalu bandingkan.

    Returns
    -------
    comparison : dict[model_name -> {model, train_results, test_results, cv_results}]
    """
    comparison = {}

    for name in AVAILABLE_MODELS:
        model = get_model(name)
        train_model(model, X_train, y_train)
        test_results = evaluate_model(model, X_test, y_test)
        cv_results = cross_validate_model(
            get_model(name), X_train, y_train, cv=cv
        )

        comparison[name] = {
            'model': model,
            'test_results': test_results,
            'cv_results': cv_results,
        }

    return comparison


def get_comparison_dataframe(comparison):
    """
    Ubah hasil perbandingan menjadi DataFrame yang rapi untuk ditampilkan.
    """
    rows = []
    for name, data in comparison.items():
        r = data['test_results']
        cv = data['cv_results']
        rows.append({
            'Model': name,
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1']:.4f}",
            'CV Mean': f"{cv['mean_score']:.4f}",
            'CV Std': f"±{cv['std_score']:.4f}",
        })
    return pd.DataFrame(rows)


def predict_single(model, scaler, input_data):
    """
    Prediksi untuk satu input data.

    Parameters
    ----------
    model : trained sklearn model
    scaler : fitted StandardScaler
    input_data : list atau array [jumlah_balita, jumlah_balita_diukur,
                                   sangat_pendek, pendek]

    Returns
    -------
    prediction : str  ('Stunting' atau 'Tidak Stunting')
    probability : dict atau None
    """
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    probability = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_scaled)[0]
        classes = list(model.classes_)
        probability = {cls: float(p) for cls, p in zip(classes, proba)}

    return prediction, probability


def get_feature_importance(model, feature_names):
    """
    Ambil feature importance dari model (jika tersedia).

    Returns
    -------
    importance_df : pd.DataFrame atau None
    """
    if hasattr(model, 'feature_importances_'):
        # Random Forest, Decision Tree, dll.
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Logistic Regression, SVM linear, dll.
        importances = np.abs(model.coef_[0])
    else:
        return None

    df = pd.DataFrame({
        'Fitur': feature_names,
        'Importance': importances,
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    return df
