"""
Model Training Script — Klasifikasi Stunting Kota Bandung
================================================================================
Script untuk melatih dan menyimpan model ML ke file .pkl untuk deployment.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from utils.data_processing import load_data, create_features_and_label, prepare_ml_data, get_feature_columns


def create_and_save_model(output_path='models/trained_model.pkl', scaler_path='models/scaler.pkl'):
    """Latih model dengan data default dan simpan ke file."""
    df = load_data()
    df = create_features_and_label(df, threshold=20.0)
    
    X_train, X_test, y_train, y_test, scaler, _, _ = prepare_ml_data(df, test_size=0.2)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model tersimpan: {output_path}")
    print(f"Scaler tersimpan: {scaler_path}")
    
    return model, scaler


if __name__ == "__main__":
    create_and_save_model()