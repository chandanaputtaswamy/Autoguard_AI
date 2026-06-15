import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

from predict_utils import predict, predict_batch, FRIENDLY
from preprocess import load_and_clean

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoGuard AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.metric-label { color: #a0aec0; font-size: 13px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { font-size: 36px; font-weight: 700; }
.good  { color: #48bb78; }
.warn  { color: #ed8936; }
.danger{ color: #fc5c65; }
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.badge-good   { background:#c6f6d5; color:#276749; }
.badge-warn   { background:#fefcbf; color:#744210; }
.badge-danger { background:#fed7d7; color:#9b2335; }
.rec-card {
    background: #1e293b;
    border-left: 4px solid #4f8ef7;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #e2e8f0;
    font-size: 14px;
}
.header-band {
    background: linear-gradient(90deg, #0f3460, #533483);
    padding: 18px 24px;
    border-radius: 12px;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def health_color(score):
    if score >= 70: return "good"
    if score >= 40: return "warn"
    return "danger"

def risk_color(prob):
    if prob < 30: return "good"
    if prob < 60: return "warn"
    return "danger"

def gauge(value, title, min_val=0, max_val=100, threshold_warn=40, threshold_danger=70, reverse=False):
    if reverse:
        color = "#fc5c65" if value < threshold_warn else "#ed8936" if value < threshold_danger else "#48bb78"
    else:
        color = "#48bb78" if value < threshold_warn else "#ed8936" if value < threshold_danger else "#fc5c65"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14, "color": "#a0aec0"}},
        number={"font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#4a5568"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1a202c",
            "bordercolor": "#2d3748",
            "steps": [
                {"range": [min_val, max_val * 0.4], "color": "#2d3748"},
                {"range": [max_val * 0.4, max_val * 0.7], "color": "#2a3548"},
                {"range": [max_val * 0.7, max_val], "color": "#2d2040"},
            ],
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 AutoGuard AI")
    st.markdown("*Vehicle Health & Predictive Maintenance*")
    st.divider()
    mode = st.radio("Input Mode", ["🎛️ Manual Entry", "📂 Upload CSV"], label_visibility="collapsed")
    st.divider()
    # Model info
    try:
        with open("models/results.json") as f:
            results = json.load(f)
        best = max(results, key=lambda k: results[k]["f1"])
        st.markdown("**🏆 Active Model**")
        st.markdown(f"`{best}`")
        r = results[best]
        st.metric("Accuracy", f"{r['accuracy']*100:.1f}%")
        st.metric("F1 Score", f"{r['f1']*100:.1f}%")
        st.metric("ROC-AUC", f"{r['roc_auc']:.4f}")
    except Exception:
        pass

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
    <h1 style="color:white;margin:0;font-size:28px;">🚗 AutoGuard AI — Vehicle Health Dashboard</h1>
    <p style="color:#a0c4ff;margin:4px 0 0;">Predictive Maintenance & Explainable Failure Detection</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Manual Entry
# ══════════════════════════════════════════════════════════════════════════════
if "Manual" in mode:
    st.markdown("### 🎛️ Enter Sensor Readings")

    col1, col2, col3 = st.columns(3)
    with col1:
        engine_temp   = st.slider("Engine Temp (°C)", 60.0, 160.0, 88.0, 0.5)
        vibration     = st.slider("Vibration (m/s²)", 0.1, 2.0, 0.45, 0.01)
        battery       = st.slider("Battery Voltage (V)", 9.0, 15.0, 13.4, 0.1)
    with col2:
        engine_load   = st.slider("Engine Load (%)", 10.0, 100.0, 55.0, 1.0)
        rpm           = st.slider("RPM", 500, 6000, 2400, 50)
        oil_pressure  = st.slider("Oil Pressure (psi)", 10.0, 80.0, 45.0, 0.5)
    with col3:
        coolant_temp  = st.slider("Coolant Temp (°C)", 70.0, 140.0, 88.0, 0.5)
        vehicle_age   = st.slider("Vehicle Age (years)", 0.5, 15.0, 4.0, 0.5)
        mileage       = st.number_input("Mileage (km)", 1000, 400000, 65000, 1000)

    if st.button("🔍 Run Diagnosis", use_container_width=True, type="primary"):
        row = {
            "engine_temp_C": engine_temp, "vibration_ms2": vibration,
            "battery_voltage_V": battery, "engine_load_pct": engine_load,
            "rpm": rpm, "oil_pressure_psi": oil_pressure,
            "coolant_temp_C": coolant_temp, "vehicle_age_years": vehicle_age,
            "mileage_km": mileage,
        }
        with st.spinner("Analysing vehicle health..."):
            result = predict(row)

        st.divider()
        st.markdown("### 📊 Diagnosis Results")

        # KPI cards
        hs = result["health_score"]
        fp = result["failure_prob"]
        rul = result["rul_days"]
        hc = health_color(hs)
        rc = risk_color(fp)
        rul_c = "good" if rul > 180 else "warn" if rul > 60 else "danger"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Health Score</div>
                <div class="metric-value {hc}">{hs}</div>
                <div style="color:#718096;font-size:12px;">/ 100</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            label = "HIGH RISK" if fp >= 60 else "MODERATE" if fp >= 30 else "LOW RISK"
            badge_cls = "badge-danger" if fp >= 60 else "badge-warn" if fp >= 30 else "badge-good"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Failure Risk</div>
                <div class="metric-value {rc}">{fp}%</div>
                <span class="badge {badge_cls}">{label}</span>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Remaining Useful Life</div>
                <div class="metric-value {rul_c}">{rul}</div>
                <div style="color:#718096;font-size:12px;">days estimated</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            status = "CRITICAL" if result["prediction"] else "NORMAL"
            sc = "danger" if result["prediction"] else "good"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">System Status</div>
                <div class="metric-value {sc}" style="font-size:26px;">{status}</div>
                <div style="color:#718096;font-size:12px;">ML prediction</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauges
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(gauge(hs, "Health Score", reverse=True,
                                  threshold_warn=40, threshold_danger=70),
                            use_container_width=True, key="g_health")
        with g2:
            st.plotly_chart(gauge(fp, "Failure Risk %",
                                  threshold_warn=30, threshold_danger=60),
                            use_container_width=True, key="g_risk")
        with g3:
            st.plotly_chart(gauge(min(rul, 365), "RUL (days)", max_val=365, reverse=True,
                                  threshold_warn=60, threshold_danger=180),
                            use_container_width=True, key="g_rul")

        # Recommendations + Explainability
        r_col, e_col = st.columns([1, 1])
        with r_col:
            st.markdown("#### 🔧 Maintenance Recommendations")
            for rec in result["recommendations"]:
                st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)

        with e_col:
            st.markdown("#### 🔬 Why is the vehicle at risk?")
            if result["risk_factors"]:
                names  = [r[0] for r in result["risk_factors"]]
                values = [r[1] for r in result["risk_factors"]]
                fig = go.Figure(go.Bar(
                    x=values, y=names, orientation="h",
                    marker=dict(color=values, colorscale="RdYlGn_r", showscale=False),
                    text=[f"{v:.1f}%" for v in values], textposition="outside",
                ))
                fig.update_layout(
                    height=280, margin=dict(l=0, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Importance (%)", color="#a0aec0", gridcolor="#2d3748"),
                    yaxis=dict(color="#e2e8f0"),
                    font=dict(color="#e2e8f0"),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Radar chart of sensor health
        st.markdown("#### 📡 Sensor Health Radar")
        sensors = {
            "Engine Temp": min(100, max(0, (160 - engine_temp) / (160 - 60) * 100)),
            "Vibration":   min(100, max(0, (2.0 - vibration) / (2.0 - 0.1) * 100)),
            "Battery":     min(100, max(0, (battery - 9.0) / (15.0 - 9.0) * 100)),
            "Engine Load": min(100, max(0, (100 - engine_load) / 90 * 100)),
            "Oil Press":   min(100, max(0, (oil_pressure - 10) / 70 * 100)),
            "Coolant":     min(100, max(0, (140 - coolant_temp) / 70 * 100)),
        }
        labels = list(sensors.keys())
        values_r = list(sensors.values())
        fig_r = go.Figure(go.Scatterpolar(
            r=values_r + [values_r[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(79,142,247,0.2)",
            line=dict(color="#4F8EF7", width=2),
            marker=dict(color="#4F8EF7", size=7),
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], color="#4a5568", gridcolor="#2d3748"),
                angularaxis=dict(color="#a0aec0"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=340, margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_r, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: Upload CSV
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### 📂 Upload Vehicle Sensor CSV")
    st.info("Expected columns: engine_temp_C, vibration_ms2, battery_voltage_V, engine_load_pct, rpm, oil_pressure_psi, coolant_temp_C, vehicle_age_years, mileage_km")

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])
    use_sample = st.checkbox("Use built-in sample dataset instead")

    df_input = None
    if uploaded:
        df_input = pd.read_csv(uploaded)
    elif use_sample:
        df_input = load_and_clean("data/vehicle_sensor_data.csv").sample(200, random_state=1).reset_index(drop=True)
        st.success("✅ Sample dataset loaded (200 rows)")

    if df_input is not None:
        with st.spinner("Running batch predictions..."):
            df_result = predict_batch(df_input)

        st.divider()

        # Summary KPIs
        avg_hs  = df_result["health_score"].mean()
        avg_fp  = df_result["failure_prob_pct"].mean()
        n_fail  = df_result["prediction"].sum()
        pct_fail = n_fail / len(df_result) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Vehicles", len(df_result))
        c2.metric("Avg Health Score", f"{avg_hs:.1f}")
        c3.metric("Avg Failure Risk", f"{avg_fp:.1f}%")
        c4.metric("At-Risk Vehicles", f"{n_fail} ({pct_fail:.0f}%)")

        # Charts row
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("**Health Score Distribution**")
            fig_hs = px.histogram(df_result, x="health_score", nbins=20,
                                  color_discrete_sequence=["#4F8EF7"],
                                  labels={"health_score": "Health Score"})
            fig_hs.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"),
            )
            st.plotly_chart(fig_hs, use_container_width=True)

        with ch2:
            st.markdown("**Failure Risk Distribution**")
            fig_fp = px.histogram(df_result, x="failure_prob_pct", nbins=20,
                                  color_discrete_sequence=["#fc5c65"],
                                  labels={"failure_prob_pct": "Failure Risk (%)"})
            fig_fp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"),
            )
            st.plotly_chart(fig_fp, use_container_width=True)

        # Scatter
        st.markdown("**Health Score vs Failure Risk (all vehicles)**")
        fig_sc = px.scatter(df_result, x="health_score", y="failure_prob_pct",
                            color="prediction",
                            color_discrete_map={0: "#48bb78", 1: "#fc5c65"},
                            labels={"health_score": "Health Score",
                                    "failure_prob_pct": "Failure Risk (%)",
                                    "prediction": "Failure"},
                            opacity=0.7, height=380)
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"),
            margin=dict(t=10),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Table — at-risk only
        at_risk_df = df_result[df_result["prediction"] == 1].sort_values(
            "failure_prob_pct", ascending=False).head(20)
        if not at_risk_df.empty:
            st.markdown(f"**🚨 Top At-Risk Vehicles (showing up to 20 of {n_fail})**")
            disp_cols = [c for c in ["engine_temp_C","vibration_ms2","battery_voltage_V",
                                      "engine_load_pct","health_score",
                                      "failure_prob_pct","rul_days"] if c in at_risk_df.columns]
            st.dataframe(
                at_risk_df[disp_cols].rename(columns={
                    "engine_temp_C": "Temp °C",
                    "vibration_ms2": "Vibration",
                    "battery_voltage_V": "Battery V",
                    "engine_load_pct": "Load %",
                    "health_score": "Health",
                    "failure_prob_pct": "Risk %",
                    "rul_days": "RUL (days)"
                }).style.background_gradient(subset=["Risk %"], cmap="RdYlGn_r"),
                use_container_width=True
            )

        # Download
        csv_out = df_result.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results CSV", csv_out,
                           "autoguard_predictions.csv", "text/csv",
                           use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:12px;padding:8px 0;">
    AutoGuard AI &nbsp;|&nbsp; Predictive Maintenance System &nbsp;|&nbsp;
    Powered by Random Forest + XGBoost &nbsp;|&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
