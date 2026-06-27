import numpy as np
import pandas as pd
import joblib

# Fixed normalization constants (domain knowledge)
NORM = {
    "engine_temp_C":      {"min": 60,  "max": 160, "danger": 110},
    "vibration_ms2":      {"min": 0.1, "max": 2.0,  "danger": 0.9},
    "battery_voltage_V":  {"min": 9.0, "max": 15.0, "danger": 11.5},
    "engine_load_pct":    {"min": 10,  "max": 100,  "danger": 85},
    "oil_pressure_psi":   {"min": 10,  "max": 80,   "danger": 25},
    "coolant_temp_C":     {"min": 70,  "max": 140,  "danger": 110},
}

FRIENDLY = {
    "engine_temp_C":     "Engine Temperature",
    "vibration_ms2":     "Vibration Level",
    "battery_voltage_V": "Battery Voltage",
    "engine_load_pct":   "Engine Load",
    "rpm":               "RPM",
    "oil_pressure_psi":  "Oil Pressure",
    "coolant_temp_C":    "Coolant Temperature",
    "vehicle_age_years": "Vehicle Age",
    "mileage_km":        "Mileage",
    "temp_load_ratio":   "Temp/Load Ratio",
    "vib_rpm_ratio":     "Vibration/RPM Ratio",
    "batt_age_interaction": "Battery×Age Factor",
    "thermal_stress":    "Thermal Stress Index",
    "is_overheating":    "Overheating Flag",
    "is_low_battery":    "Low Battery Flag",
    "is_high_vib":       "High Vibration Flag",
}

def load_model():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

def engineer_single(row: dict) -> dict:
    row = row.copy()
    row["temp_load_ratio"] = row["engine_temp_C"] / (row["engine_load_pct"] + 1)
    row["vib_rpm_ratio"] = row["vibration_ms2"] / (row["rpm"] / 1000 + 0.01)
    row["batt_age_interaction"] = row["battery_voltage_V"] * (1 / (row["vehicle_age_years"] + 1))
    row["thermal_stress"] = (row["engine_temp_C"] + row["coolant_temp_C"]) / 2
    row["is_overheating"] = int(row["engine_temp_C"] > 110)
    row["is_low_battery"] = int(row["battery_voltage_V"] < 11.5)
    row["is_high_vib"] = int(row["vibration_ms2"] > 0.9)
    return row

def compute_health_score(row: dict) -> float:
    """0-100, higher = healthier."""
    penalties = 0.0
    # Each sensor: penalty proportional to how far into danger zone
    for col, norms in NORM.items():
        val = row.get(col, None)
        if val is None:
            continue
        rng = norms["max"] - norms["min"]
        if col == "battery_voltage_V" or col == "oil_pressure_psi":
            # Lower is worse
            if val < norms["danger"]:
                severity = (norms["danger"] - val) / (norms["danger"] - norms["min"])
                penalties += min(severity, 1.0) * 20
        else:
            # Higher is worse
            if val > norms["danger"]:
                severity = (val - norms["danger"]) / (norms["max"] - norms["danger"])
                penalties += min(severity, 1.0) * 20

    score = max(0.0, 100.0 - penalties)
    return round(score, 1)

def estimate_rul(row: dict, failure_prob: float) -> int:
    """Remaining Useful Life in days."""
    base_rul = 365  # 1 year baseline
    age_factor = max(0.3, 1 - row.get("vehicle_age_years", 5) / 20)
    mileage_factor = max(0.3, 1 - row.get("mileage_km", 100000) / 400000)
    risk_factor = max(0.05, 1 - failure_prob)
    rul = int(base_rul * age_factor * mileage_factor * risk_factor)
    return max(1, rul)

def get_recommendations(row: dict, failure_prob: float) -> list:
    recs = []
    if row.get("engine_temp_C", 0) > 110:
        recs.append({
            "text": "Engine overheating detected — check coolant level and thermostat immediately.",
            "priority": "CRITICAL",
            "code": "P0217",
            "icon": "🌡️"
        })
    if row.get("vibration_ms2", 0) > 0.9:
        recs.append({
            "text": "High vibration — inspect engine mounts, driveshaft, and wheel balance.",
            "priority": "CRITICAL",
            "code": "P0300",
            "icon": "🔧"
        })
    if row.get("battery_voltage_V", 15) < 11.5:
        recs.append({
            "text": "Low battery voltage — test and replace battery; check alternator output.",
            "priority": "WARNING",
            "code": "P0562",
            "icon": "🔋"
        })
    if row.get("engine_load_pct", 0) > 85:
        recs.append({
            "text": "High engine load — reduce load; check for clogged air filter or fuel issues.",
            "priority": "WARNING",
            "code": "P115D",
            "icon": "⚙️"
        })
    if row.get("oil_pressure_psi", 80) < 25:
        recs.append({
            "text": "Low oil pressure — check oil level and pump; do not drive until resolved.",
            "priority": "CRITICAL",
            "code": "P0522",
            "icon": "🛢️"
        })
    if row.get("coolant_temp_C", 0) > 110:
        recs.append({
            "text": "Coolant overheating — inspect radiator, water pump, and hoses.",
            "priority": "CRITICAL",
            "code": "P0117",
            "icon": "💧"
        })
    if row.get("mileage_km", 0) > 150000:
        recs.append({
            "text": "High mileage — schedule a full service: belts, plugs, filters, and fluids.",
            "priority": "ROUTINE",
            "code": "DTC-INFO",
            "icon": "📅"
        })
    
    if not recs:
        # failure_prob is a raw fraction (0.0–1.0) from predict_proba
        if failure_prob < 0.15:
            recs.append({
                "text": "Vehicle in good health — continue regular maintenance schedule.",
                "priority": "ROUTINE",
                "code": "SYSTEM-OK",
                "icon": "✅"
            })
        else:
            recs.append({
                "text": "Moderate risk detected — schedule preventive inspection within 2 weeks.",
                "priority": "WARNING",
                "code": "PREV-MAINT",
                "icon": "⚠️"
            })
    return recs

def get_risk_factors(model, feature_names: list, row_engineered: dict) -> list:
    """Return top contributing features to the failure prediction."""
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    paired = sorted(zip(importances, feature_names), reverse=True)
    top = [(FRIENDLY.get(name, name), round(float(imp * 100), 1))
           for imp, name in paired[:6]]
    return top

def predict(row: dict):
    """Full prediction pipeline. row: dict of raw sensor values."""
    model, scaler, feature_names = load_model()
    eng = engineer_single(row)
    X = pd.DataFrame([[eng.get(f, 0) for f in feature_names]], columns=feature_names)
    X_scaled = scaler.transform(X)
    failure_prob = float(model.predict_proba(X_scaled)[0][1])
    prediction = int(failure_prob > 0.5)
    health_score = compute_health_score(row)
    rul = estimate_rul(row, failure_prob)
    recs = get_recommendations(row, failure_prob)
    risk_factors = get_risk_factors(model, feature_names, eng)
    return {
        "failure_prob": round(failure_prob * 100, 1),
        "prediction": prediction,
        "health_score": health_score,
        "rul_days": rul,
        "recommendations": recs,
        "risk_factors": risk_factors,
    }

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict for a full DataFrame."""
    model, scaler, feature_names = load_model()
    from preprocess import engineer_features
    df_eng = engineer_features(df)
    available = [c for c in feature_names if c in df_eng.columns]
    X = df_eng[available]
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    df = df.copy()
    df["failure_prob_pct"] = (probs * 100).round(1)
    df["prediction"] = (probs > 0.5).astype(int)
    df["health_score"] = [compute_health_score(row) for row in df.to_dict("records")]
    df["rul_days"] = [estimate_rul(row, p) for row, p in zip(df.to_dict("records"), probs)]
    return df
