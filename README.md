#  AutoGuard AI — Vehicle Health & Predictive Maintenance

> A hackathon-ready ML project that predicts vehicle failures, estimates Remaining Useful Life (RUL), scores vehicle health, and provides actionable maintenance recommendations — all via a modern interactive dashboard.

---

##  Dashboard Features

| Feature | Description |
|---|---|
| **Health Score** | 0–100 composite score per vehicle |
| **Failure Risk %** | ML-predicted probability of imminent failure |
| **RUL Estimate** | Remaining Useful Life in days |
| **Recommendations** | Specific, actionable maintenance actions |
| **Explainable AI** | Top features driving each prediction |
| **Radar Chart** | Visual sensor health at a glance |
| **Batch Upload** | Analyse entire fleets via CSV |
| **Download Results** | Export predictions to CSV |

---

##  Project Structure

```
autoguard/
├── app.py                  # Streamlit dashboard (main entry point)
├── run_pipeline.py         # One-shot setup script
├── requirements.txt
├── README.md
├── data/
│   └── vehicle_sensor_data.csv   # Generated dataset (5,000 vehicles)
├── models/
│   ├── best_model.pkl            # Saved best ML model
│   ├── scaler.pkl                # Feature scaler
│   ├── feature_names.pkl         # Feature list
│   └── results.json              # Model comparison metrics
├── assets/
│   ├── eda_distributions.png
│   ├── eda_correlation.png
│   ├── eda_temp_failure.png
│   ├── eda_boxplots.png
│   ├── model_comparison.png
│   ├── cm_random_forest.png
│   └── fi_random_forest.png
└── src/
    ├── generate_data.py    # Synthetic dataset generator
    ├── preprocess.py       # Cleaning + feature engineering
    ├── eda.py              # EDA charts
    ├── train.py            # Model training & evaluation
    └── predict_utils.py    # Health score, RUL, recommendations, XAI
```

---

##  Quick Start

### 1. Clone / download the project

```bash
cd autoguard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline (once)

```bash
python run_pipeline.py
```

This will:
- Generate 5,000 synthetic vehicle sensor records
- Run EDA and save charts to `assets/`
- Train Random Forest, Decision Tree, and XGBoost
- Save the best model to `models/`

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

##  ML Models

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| **Random Forest** ✅ | ~94.8% | ~92.3% | ~0.92 |
| XGBoost | ~94.4% | ~91.7% | ~0.93 |
| Decision Tree | ~92.9% | ~89.2% | ~0.90 |

Best model selected automatically by F1 score.

---

##  Features Used

**Raw sensors:** Engine Temp, Vibration, Battery Voltage, Engine Load, RPM, Oil Pressure, Coolant Temp, Vehicle Age, Mileage

**Engineered:**
- `temp_load_ratio` — thermal efficiency indicator
- `vib_rpm_ratio` — mechanical stress at speed
- `batt_age_interaction` — battery health adjusted for age
- `thermal_stress` — combined thermal load
- `is_overheating / is_low_battery / is_high_vib` — binary alert flags

---

##  Dataset

Synthetic dataset of 5,000 vehicle records with ~35% failure rate.
Includes realistic operating ranges, failure injection patterns, and 2% missing values for preprocessing practice.

CSV columns: `engine_temp_C, vibration_ms2, battery_voltage_V, engine_load_pct, rpm, oil_pressure_psi, coolant_temp_C, vehicle_age_years, mileage_km, failure`

---

##  Hackathon Slide Mockup Descriptions

**Slide 1 — Problem:** Fleet operators lose millions to unplanned breakdowns. Predictive maintenance can cut downtime by 30–50%.

**Slide 2 — Solution:** AutoGuard AI monitors 9 sensor streams in real-time, predicts failures before they occur, and tells mechanics exactly what to fix.

**Slide 3 — Demo:** Live dashboard showing Health Score gauge (84 → green), Failure Risk (3.7% → safe), RUL (235 days), and radar chart.

**Slide 4 — Technology:** Random Forest classifier (94.8% accuracy), feature engineering pipeline, SHAP-style feature importance, Streamlit UI.

**Slide 5 — Impact:** Early warning → scheduled repair → 10× cheaper than emergency breakdown. ROI calculable per fleet size.

---

##  License

MIT — free for personal, educational, and hackathon use.
"# Autoguard_AI" 
