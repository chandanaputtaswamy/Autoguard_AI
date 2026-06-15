import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLS = [
    "engine_temp_C", "vibration_ms2", "battery_voltage_V",
    "engine_load_pct", "rpm", "oil_pressure_psi", "coolant_temp_C",
    "vehicle_age_years", "mileage_km",
    # Engineered
    "temp_load_ratio", "vib_rpm_ratio", "batt_age_interaction",
    "thermal_stress", "is_overheating", "is_low_battery", "is_high_vib"
]

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Fill missing with median
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_load_ratio"] = df["engine_temp_C"] / (df["engine_load_pct"] + 1)
    df["vib_rpm_ratio"] = df["vibration_ms2"] / (df["rpm"] / 1000 + 0.01)
    df["batt_age_interaction"] = df["battery_voltage_V"] * (1 / (df["vehicle_age_years"] + 1))
    df["thermal_stress"] = (df["engine_temp_C"] + df["coolant_temp_C"]) / 2
    df["is_overheating"] = (df["engine_temp_C"] > 110).astype(int)
    df["is_low_battery"] = (df["battery_voltage_V"] < 11.5).astype(int)
    df["is_high_vib"] = (df["vibration_ms2"] > 0.9).astype(int)
    return df

def get_X_y(df: pd.DataFrame):
    df = engineer_features(df)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df["failure"] if "failure" in df.columns else None
    return X, y

def fit_scaler(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    return X_scaled, scaler

def apply_scaler(X):
    scaler = joblib.load("models/scaler.pkl")
    return scaler.transform(X)
