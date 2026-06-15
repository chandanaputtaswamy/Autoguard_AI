import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 5000

# Normal operating ranges
temp_normal = np.random.normal(85, 10, N)
vib_normal = np.random.normal(0.4, 0.1, N)
batt_normal = np.random.normal(13.5, 0.5, N)
load_normal = np.random.normal(55, 15, N)
rpm_normal = np.random.normal(2500, 400, N)
oil_normal = np.random.normal(45, 8, N)
coolant_normal = np.random.normal(90, 8, N)

# Inject failure patterns (30% of records)
fail_idx = np.random.choice(N, int(N * 0.30), replace=False)

temp = temp_normal.copy()
temp[fail_idx] += np.random.uniform(20, 60, len(fail_idx))

vib = vib_normal.copy()
vib[fail_idx] += np.random.uniform(0.3, 1.2, len(fail_idx))

batt = batt_normal.copy()
batt[fail_idx] -= np.random.uniform(1.0, 3.5, len(fail_idx))

load = load_normal.copy()
load[fail_idx] += np.random.uniform(20, 40, len(fail_idx))

rpm = rpm_normal.copy()
rpm[fail_idx] += np.random.choice([-1, 1], len(fail_idx)) * np.random.uniform(500, 1500, len(fail_idx))

oil = oil_normal.copy()
oil[fail_idx] -= np.random.uniform(10, 30, len(fail_idx))

coolant = coolant_normal.copy()
coolant[fail_idx] += np.random.uniform(15, 40, len(fail_idx))

# Clip to realistic ranges
temp = np.clip(temp, 60, 160)
vib = np.clip(vib, 0.1, 2.0)
batt = np.clip(batt, 9.0, 15.0)
load = np.clip(load, 10, 100)
rpm = np.clip(rpm, 500, 6000)
oil = np.clip(oil, 10, 80)
coolant = np.clip(coolant, 70, 140)

# Failure label: based on thresholds with some noise
failure = np.zeros(N, dtype=int)
at_risk = (
    (temp > 110) | (vib > 0.9) | (batt < 11.5) |
    (load > 85) | (oil < 25) | (coolant > 110)
)
failure[at_risk] = 1
# Add 5% noise
noise = np.random.choice(N, int(N * 0.05), replace=False)
failure[noise] = 1 - failure[noise]

# Vehicle age and mileage
age_years = np.random.uniform(0.5, 15, N)
mileage = age_years * np.random.uniform(8000, 20000, N)

df = pd.DataFrame({
    "engine_temp_C": temp.round(2),
    "vibration_ms2": vib.round(3),
    "battery_voltage_V": batt.round(2),
    "engine_load_pct": load.round(1),
    "rpm": rpm.round(0).astype(int),
    "oil_pressure_psi": oil.round(1),
    "coolant_temp_C": coolant.round(2),
    "vehicle_age_years": age_years.round(1),
    "mileage_km": mileage.round(0).astype(int),
    "failure": failure
})

# Inject 2% missing values
for col in ["engine_temp_C", "vibration_ms2", "oil_pressure_psi"]:
    miss_idx = np.random.choice(N, int(N * 0.02), replace=False)
    df.loc[miss_idx, col] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/vehicle_sensor_data.csv", index=False)
print(f"Dataset saved: {len(df)} rows, {df['failure'].mean()*100:.1f}% failure rate")
print(df.describe())
