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
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Main Theme Variables */
:root {
    --bg-dark: #0a0e1a;
    --card-bg: rgba(17, 24, 39, 0.65);
    --card-border: rgba(255, 255, 255, 0.08);
    --primary: #3b82f6;
    --primary-glow: rgba(59, 130, 246, 0.4);
    --success: #10b981;
    --success-glow: rgba(16, 185, 129, 0.3);
    --warning: #f59e0b;
    --warning-glow: rgba(245, 158, 11, 0.3);
    --danger: #ef4444;
    --danger-glow: rgba(239, 68, 68, 0.3);
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
}

/* Apply fonts */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', 'Montserrat', sans-serif;
    font-weight: 700;
}

/* Glassmorphism Cards */
.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 20px;
}

.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.15);
}

/* Telemetry Card Status Accents */
.card-success { border-left: 4px solid var(--success); }
.card-warning { border-left: 4px solid var(--warning); }
.card-danger { border-left: 4px solid var(--danger); }

/* Metric Styling */
.metric-header {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.metric-val {
    font-size: 38px;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    line-height: 1;
    margin-bottom: 6px;
}

.metric-desc {
    font-size: 12px;
    color: var(--text-muted);
}

/* Alert Beacons */
.beacon-container {
    display: flex;
    align-items: center;
    gap: 8px;
}

.beacon {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

.beacon-pulse-green {
    background: var(--success);
    box-shadow: 0 0 0 0 var(--success-glow);
    animation: pulse-green 1.8s infinite;
}

.beacon-pulse-red {
    background: var(--danger);
    box-shadow: 0 0 0 0 var(--danger-glow);
    animation: pulse-red 1.8s infinite;
}

.beacon-pulse-warning {
    background: var(--warning);
    box-shadow: 0 0 0 0 var(--warning-glow);
    animation: pulse-warning 1.8s infinite;
}

@keyframes pulse-green {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

@keyframes pulse-red {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

@keyframes pulse-warning {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(245, 158, 11, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
}

/* Custom Badges */
.badge-custom {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 1px solid transparent;
}

.badge-custom-success {
    background: rgba(16, 185, 129, 0.1);
    color: var(--success);
    border-color: rgba(16, 185, 129, 0.2);
}

.badge-custom-warning {
    background: rgba(245, 158, 11, 0.1);
    color: var(--warning);
    border-color: rgba(245, 158, 11, 0.2);
}

.badge-custom-danger {
    background: rgba(239, 68, 68, 0.1);
    color: var(--danger);
    border-color: rgba(239, 68, 68, 0.2);
}

.badge-custom-info {
    background: rgba(59, 130, 246, 0.1);
    color: var(--primary);
    border-color: rgba(59, 130, 246, 0.2);
}

/* Mechanic Work Orders / Recommendations */
.work-order-card {
    background: rgba(22, 28, 45, 0.85);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    transition: border-color 0.2s ease;
}

.work-order-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
}

.work-order-meta {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 90px;
}

.work-order-content {
    flex-grow: 1;
}

.dtc-badge {
    background: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1;
    font-family: monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
    text-align: center;
}

/* Styled sidebar active model card */
.sidebar-model-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 14px;
    margin-top: 10px;
}

/* User Manual Elements */
.sensor-deep-dive {
    border-left: 3px solid var(--primary);
    padding-left: 14px;
    margin: 15px 0;
}

.step-card {
    background: rgba(17, 24, 39, 0.4);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

.step-number {
    font-size: 24px;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    color: var(--primary);
    opacity: 0.7;
    margin-bottom: 6px;
}

/* Print/Ticket Header */
.ticket-header {
    border-bottom: 2px dashed rgba(255, 255, 255, 0.15);
    padding-bottom: 16px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def health_color(score):
    if score >= 70: return "#10b981"
    if score >= 40: return "#f59e0b"
    return "#ef4444"

def risk_color(prob):
    if prob < 30: return "#10b981"
    if prob < 60: return "#f59e0b"
    return "#ef4444"

def gauge(value, title, min_val=0, max_val=100, threshold_warn=40, threshold_danger=70, reverse=False):
    green_c = "#10b981"
    orange_c = "#f59e0b"
    red_c = "#ef4444"
    
    if reverse:
        color = red_c if value < threshold_warn else orange_c if value < threshold_danger else green_c
    else:
        color = green_c if value < threshold_warn else orange_c if value < threshold_danger else red_c
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 15, "color": "#9ca3af", "family": "Space Grotesk"}},
        number={"font": {"size": 32, "color": color, "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#4b5563", "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(31, 41, 55, 0.4)",
            "bordercolor": "rgba(255, 255, 255, 0.08)",
            "borderwidth": 1,
            "steps": [
                {"range": [min_val, max_val * 0.4], "color": "rgba(31, 41, 55, 0.2)"},
                {"range": [max_val * 0.4, max_val * 0.7], "color": "rgba(31, 41, 55, 0.3)"},
                {"range": [max_val * 0.7, max_val], "color": "rgba(31, 41, 55, 0.5)"},
            ],
        }
    ))
    fig.update_layout(
        height=180, margin=dict(l=15, r=15, t=35, b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f3f4f6"},
    )
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <span style="font-size: 32px;">🚗</span>
        <div>
            <h2 style="margin: 0; font-size: 20px; font-weight: 800; font-family: 'Space Grotesk', sans-serif;">AUTOGUARD AI</h2>
            <span style="font-size: 11px; color: #9ca3af; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">Predictive Telematics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div class="beacon-container" style="background: rgba(31, 41, 55, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); padding: 10px 14px; border-radius: 10px;">
        <span class="beacon beacon-pulse-green"></span>
        <span style="font-size: 12px; font-weight: 600; color: #f3f4f6;">TELEMETRY FEED ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Model info
    try:
        with open("models/results.json") as f:
            results = json.load(f)
        best = max(results, key=lambda k: results[k]["f1"])
        r = results[best]
        
        st.markdown(f"""
        <div class="sidebar-model-card">
            <span style="font-size: 10px; font-weight: 800; color: #9ca3af; text-transform: uppercase;">Active Classifier</span>
            <div style="font-size: 15px; font-weight: 700; color: #3b82f6; margin-bottom: 8px;">{best}</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                <div>
                    <div style="font-size: 9px; color: #9ca3af; text-transform: uppercase;">Accuracy</div>
                    <div style="font-size: 14px; font-weight: 700; color: #f3f4f6;">{r['accuracy']*100:.1f}%</div>
                </div>
                <div>
                    <div style="font-size: 9px; color: #9ca3af; text-transform: uppercase;">F1-Score</div>
                    <div style="font-size: 14px; font-weight: 700; color: #f3f4f6;">{r['f1']*100:.1f}%</div>
                </div>
            </div>
            <div style="font-size: 9px; color: #9ca3af; text-transform: uppercase; margin-top: 6px;">ROC-AUC</div>
            <div style="font-size: 13px; font-weight: 700; color: #10b981;">{r['roc_auc']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass
        
    st.divider()
    
    st.markdown("""
    <div style="font-size: 10px; color: #4b5563; text-align: center;">
        AutoGuard Telematics Console v2.4.0
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(90deg, rgba(31, 41, 55, 0.8), rgba(17, 24, 39, 0.9)); border: 1px solid rgba(255, 255, 255, 0.05); padding: 24px; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
    <h1 style="color:white;margin:0;font-size:32px;font-family: 'Space Grotesk', sans-serif;font-weight: 800;letter-spacing:-0.02em;">🚗 AutoGuard AI Telematics Portal</h1>
    <p style="color:#60a5fa;margin:4px 0 0;font-size:14px;font-weight:500;">Predictive Maintenance Pipeline & Explainable Failure Detection</p>
</div>
""", unsafe_allow_html=True)

# ── Navigation Tabs ───────────────────────────────────────────────────────────
tab_live, tab_fleet, tab_manual, tab_insights = st.tabs([
    "🎛️ Live Diagnostics Console", 
    "📂 Fleet Analytics Hub", 
    "📖 Interactive User Manual", 
    "🧠 ML Engine Insights"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Live Diagnostics Console
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("### 🎛️ Single Vehicle Real-Time Diagnostics")
    st.markdown("Adjust the sliders below to simulate live vehicle sensor telematics and run the diagnostics pipeline.")
    
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

    if "manual_diagnosis" not in st.session_state:
        st.session_state.manual_diagnosis = None

    if st.button("🔍 Run Telematics Diagnosis", use_container_width=True, type="primary"):
        row = {
            "engine_temp_C": engine_temp, "vibration_ms2": vibration,
            "battery_voltage_V": battery, "engine_load_pct": engine_load,
            "rpm": rpm, "oil_pressure_psi": oil_pressure,
            "coolant_temp_C": coolant_temp, "vehicle_age_years": vehicle_age,
            "mileage_km": mileage,
        }
        with st.spinner("Executing machine learning diagnostics pipeline..."):
            st.session_state.manual_diagnosis = {
                "inputs": row,
                "results": predict(row)
            }

    if st.session_state.manual_diagnosis is not None:
        inputs = st.session_state.manual_diagnosis["inputs"]
        result = st.session_state.manual_diagnosis["results"]
        
        st.divider()
        st.markdown("### 📊 Diagnostic Results")

        hs = result["health_score"]
        fp = result["failure_prob"]
        rul = result["rul_days"]
        prediction = result["prediction"]
        
        h_color = health_color(hs)
        r_color = risk_color(fp)
        
        if hs >= 70:
            h_status = "HEALTHY"
            h_card_cls = "card-success"
        elif hs >= 40:
            h_status = "WARNING"
            h_card_cls = "card-warning"
        else:
            h_status = "CRITICAL"
            h_card_cls = "card-danger"
            
        if fp >= 60:
            f_status = "HIGH RISK"
            f_badge = "badge-custom-danger"
            f_card_cls = "card-danger"
        elif fp >= 30:
            f_status = "MODERATE"
            f_badge = "badge-custom-warning"
            f_card_cls = "card-warning"
        else:
            f_status = "LOW RISK"
            f_badge = "badge-custom-success"
            f_card_cls = "card-success"
            
        rul_color = "#10b981" if rul > 180 else "#f59e0b" if rul > 60 else "#ef4444"
        rul_card_cls = "card-success" if rul > 180 else "card-warning" if rul > 60 else "card-danger"
        
        if prediction == 1:
            status_text = "CRITICAL BREAKDOWN RISK"
            status_color = "#ef4444"
            status_card_cls = "card-danger"
            status_beacon = "beacon-pulse-red"
        else:
            status_text = "SYSTEMS NORMAL"
            status_color = "#10b981"
            status_card_cls = "card-success"
            status_beacon = "beacon-pulse-green"
            
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="glass-card {h_card_cls}">
                <div class="metric-header">Health Score</div>
                <div class="metric-val" style="color: {h_color};">{hs:.1f}</div>
                <div class="metric-desc">
                    <span style="font-weight: 700; color: {h_color};">{h_status}</span> &nbsp;|&nbsp; 100.0 max
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="glass-card {f_card_cls}">
                <div class="metric-header">Failure Probability</div>
                <div class="metric-val" style="color: {r_color};">{fp:.1f}%</div>
                <div class="metric-desc">
                    <span class="badge-custom {f_badge}">{f_status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="glass-card {rul_card_cls}">
                <div class="metric-header">Estimated RUL</div>
                <div class="metric-val" style="color: {rul_color};">{rul}</div>
                <div class="metric-desc">Remaining Useful Life (days)</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="glass-card {status_card_cls}">
                <div class="metric-header">ML Pipeline Prediction</div>
                <div class="metric-val" style="color: {status_color}; font-size: 18px; padding: 11px 0 12px; font-weight:700;">{status_text}</div>
                <div class="metric-desc" style="display: flex; align-items: center; gap: 6px;">
                    <span class="beacon {status_beacon}"></span>
                    <span>Active Telematics Feed</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(gauge(hs, "Composite Health Score", reverse=True,
                                  threshold_warn=40, threshold_danger=70),
                            use_container_width=True, key="live_g_health")
        with g2:
            st.plotly_chart(gauge(fp, "Imminent Failure Risk %",
                                  threshold_warn=30, threshold_danger=60),
                            use_container_width=True, key="live_g_risk")
        with g3:
            st.plotly_chart(gauge(min(rul, 365), "Remaining Useful Life (RUL)", max_val=365, reverse=True,
                                  threshold_warn=60, threshold_danger=180),
                            use_container_width=True, key="live_g_rul")

        st.markdown("<br>", unsafe_allow_html=True)

        left_col, right_col = st.columns([1.1, 0.9])
        with left_col:
            st.markdown("#### 🔧 Active Mechanic Work Orders & Diagnostic Trouble Codes (DTC)")
            recs = result["recommendations"]
            for rec in recs:
                # Defensive: handle both dict (new format) and plain string (legacy)
                if isinstance(rec, str):
                    rec = {"text": rec, "priority": "ROUTINE", "code": "INFO", "icon": "ℹ️"}
                badge_class = "badge-custom-danger" if rec["priority"] == "CRITICAL" else "badge-custom-warning" if rec["priority"] == "WARNING" else "badge-custom-info"
                st.markdown(f"""
                <div class="work-order-card">
                    <div class="work-order-meta">
                        <span class="dtc-badge">{rec['code']}</span>
                        <span class="badge-custom {badge_class}" style="font-size: 8px; padding: 2px 6px; margin-top:4px; text-align:center;">{rec['priority']}</span>
                    </div>
                    <div class="work-order-content">
                        <div style="font-size: 14px; font-weight: 500; line-height: 1.4; color: #f3f4f6;">
                            {rec['icon']} {rec['text']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🖨️ Compile Official Maintenance Ticket", use_container_width=True):
                ticket_id = f"AG-{np.random.randint(100000, 999999)}"
                st.markdown(f"""
                <div class="glass-card" style="font-family: 'Courier New', Courier, monospace; background: rgba(0, 0, 0, 0.4); border: 2px dashed rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 24px;">
                    <div class="ticket-header" style="text-align: center;">
                        <h4 style="margin: 0; color: #fff; letter-spacing: 0.1em; font-family: monospace;">AUTOGUARD SECURE WORK ORDER</h4>
                        <div style="font-size: 12px; color: #9ca3af; margin-top: 4px;">SYSTEM PORTAL v2.4.0 &nbsp;|&nbsp; REF: {ticket_id}</div>
                        <div style="font-size: 11px; color: #9ca3af;">TIMESTAMP: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                    </div>
                    
                    <table style="width: 100%; font-size: 12px; border-collapse: collapse; margin-bottom: 16px;">
                        <tr>
                            <td style="color: #9ca3af; padding: 4px 0;">VEHICLE AGE:</td>
                            <td style="text-align: right; color: #fff; font-weight: bold;">{inputs['vehicle_age_years']} Years</td>
                        </tr>
                        <tr>
                            <td style="color: #9ca3af; padding: 4px 0;">FLEET MILEAGE:</td>
                            <td style="text-align: right; color: #fff; font-weight: bold;">{inputs['mileage_km']:,} KM</td>
                        </tr>
                        <tr>
                            <td style="color: #9ca3af; padding: 4px 0;">DIAGNOSTIC HEALTH SCORE:</td>
                            <td style="text-align: right; color: {h_color}; font-weight: bold;">{hs}/100.0</td>
                        </tr>
                        <tr>
                            <td style="color: #9ca3af; padding: 4px 0;">PROBABILITY OF FAILURE:</td>
                            <td style="text-align: right; color: {r_color}; font-weight: bold;">{fp}%</td>
                        </tr>
                        <tr>
                            <td style="color: #9ca3af; padding: 4px 0;">ESTIMATED LIFE EXPIRY (RUL):</td>
                            <td style="text-align: right; color: {rul_color}; font-weight: bold;">{rul} Days</td>
                        </tr>
                    </table>
                    
                    <div style="font-size: 12px; font-weight: bold; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 4px; margin-bottom: 8px; color: #fff;">ACTIVE DIAGNOSTIC CODES & REPAIR ACTIONS:</div>
                    {"".join([f'''
                    <div style="display: flex; gap: 8px; font-size: 12px; margin-bottom: 6px; align-items: flex-start;">
                        <span style="background: rgba(255,255,255,0.08); padding: 1px 4px; border-radius: 3px; font-weight: bold; min-width: 65px; text-align: center;">[{(item if isinstance(item, dict) else {"code":"INFO","priority":"ROUTINE","text":item,"icon":"ℹ️"})['code']}]</span>
                        <span style="color: #9ca3af;"><strong style="color: {h_color if (item if isinstance(item, dict) else {}).get('priority','') == 'ROUTINE' else r_color}; font-weight: 700;">({(item if isinstance(item, dict) else {"code":"INFO","priority":"ROUTINE","text":item,"icon":"ℹ️"})['priority']})</strong> {(item if isinstance(item, dict) else {"code":"INFO","priority":"ROUTINE","text":item,"icon":"ℹ️"})['text']}</span>
                    </div>
                    ''' for item in recs])}
                    
                    <div style="border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 12px; margin-top: 16px; text-align: center; font-size: 11px; color: #60a5fa;">
                        AUTHENTICATED SECURE LOG &nbsp;|&nbsp; DIGITAL SERVICE RECORD PRINT
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with right_col:
            st.markdown("#### 🔬 Explainable AI: Feature Risk Factors")
            if result["risk_factors"]:
                names  = [r[0] for r in result["risk_factors"]]
                values = [r[1] for r in result["risk_factors"]]
                
                fig = go.Figure(go.Bar(
                    x=values, y=names, orientation="h",
                    marker=dict(
                        color=values, 
                        colorscale=[
                            [0.0, "rgba(59, 130, 246, 0.85)"], 
                            [0.5, "rgba(245, 158, 11, 0.85)"], 
                            [1.0, "rgba(239, 68, 68, 0.9)"]
                        ], 
                        showscale=False
                    ),
                    text=[f"{v:.1f}%" for v in values], 
                    textposition="outside",
                    textfont=dict(color="#f3f4f6", family="Space Grotesk")
                ))
                fig.update_layout(
                    height=280, margin=dict(l=0, r=40, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        title="Risk Contribution Importance (%)", 
                        color="#9ca3af", 
                        gridcolor="rgba(255,255,255,0.05)",
                        zerolinecolor="rgba(255,255,255,0.05)"
                    ),
                    yaxis=dict(color="#f3f4f6", tickfont=dict(family="Inter")),
                    font=dict(color="#f3f4f6"),
                )
                st.plotly_chart(fig, use_container_width=True, key="live_explain_bar")

            st.markdown("#### 📡 Sensor Health Analysis Radar")
            sensors = {
                "Engine Temp": min(100, max(0, (160 - inputs["engine_temp_C"]) / (160 - 60) * 100)),
                "Vibration":   min(100, max(0, (2.0 - inputs["vibration_ms2"]) / (2.0 - 0.1) * 100)),
                "Battery":     min(100, max(0, (inputs["battery_voltage_V"] - 9.0) / (15.0 - 9.0) * 100)),
                "Engine Load": min(100, max(0, (100 - inputs["engine_load_pct"]) / 90 * 100)),
                "Oil Press":   min(100, max(0, (inputs["oil_pressure_psi"] - 10) / 70 * 100)),
                "Coolant":     min(100, max(0, (140 - inputs["coolant_temp_C"]) / 70 * 100)),
            }
            labels = list(sensors.keys())
            values_r = list(sensors.values())
            fig_r = go.Figure(go.Scatterpolar(
                r=values_r + [values_r[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(59, 130, 246, 0.15)",
                line=dict(color="#3b82f6", width=2.5),
                marker=dict(color="#60a5fa", size=8),
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor="rgba(17, 24, 39, 0.4)",
                    radialaxis=dict(visible=True, range=[0, 100], color="#9ca3af", gridcolor="rgba(255,255,255,0.05)"),
                    angularaxis=dict(color="#f3f4f6", gridcolor="rgba(255,255,255,0.05)"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f3f4f6", family="Space Grotesk"),
                height=340, margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig_r, use_container_width=True, key="live_radar")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Fleet Analytics Hub
# ══════════════════════════════════════════════════════════════════════════════
with tab_fleet:
    st.markdown("### 📂 Fleet-Wide Telematics Batch Upload")
    st.markdown("Upload vehicle sensor records (CSV format) to perform batch diagnoses, inspect fleet failure curves, and download actionable insights.")
    st.info("Expected columns: `engine_temp_C`, `vibration_ms2`, `battery_voltage_V`, `engine_load_pct`, `rpm`, `oil_pressure_psi`, `coolant_temp_C`, `vehicle_age_years`, `mileage_km`")

    uploaded = st.file_uploader("Drag and drop your Fleet Telematics CSV here", type=["csv"])
    use_sample = st.checkbox("Load Sample Telematics Dataset (200 records)")

    df_input = None
    if uploaded:
        df_input = pd.read_csv(uploaded)
    elif use_sample:
        df_input = load_and_clean("data/vehicle_sensor_data.csv").sample(200, random_state=1).reset_index(drop=True)
        st.success("✅ Fleet telematics sample dataset loaded (200 records)")

    if df_input is not None:
        with st.spinner("Processing telematics datasets and running batch classifications..."):
            df_result = predict_batch(df_input)

        st.divider()
        st.markdown("### 📊 Fleet Diagnostics Summary")

        # Summary KPIs
        avg_hs  = df_result["health_score"].mean()
        avg_fp  = df_result["failure_prob_pct"].mean()
        n_fail  = df_result["prediction"].sum()
        pct_fail = n_fail / len(df_result) * 100

        hc = health_color(avg_hs)
        rc = risk_color(avg_fp)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-header">Total Vehicles</div>
                <div class="metric-val" style="color: #3b82f6;">{len(df_result)}</div>
                <div class="metric-desc">Registered Fleet Assets</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="glass-card card-success">
                <div class="metric-header">Average Health</div>
                <div class="metric-val" style="color: {hc};">{avg_hs:.1f}</div>
                <div class="metric-desc">Fleet health score index</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="glass-card card-warning">
                <div class="metric-header">Average Failure Risk</div>
                <div class="metric-val" style="color: {rc};">{avg_fp:.1f}%</div>
                <div class="metric-desc">Mean risk probability</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            fc = "#ef4444" if pct_fail > 15 else "#f59e0b" if pct_fail > 5 else "#10b981"
            st.markdown(f"""
            <div class="glass-card card-danger">
                <div class="metric-header">At-Risk Assets</div>
                <div class="metric-val" style="color: {fc};">{n_fail}</div>
                <div class="metric-desc">
                    <span style="font-weight: 700; color: {fc};">{pct_fail:.1f}%</span> of fleet
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Charts row
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("**Health Score Distribution**")
            fig_hs = px.histogram(df_result, x="health_score", nbins=20,
                                  color_discrete_sequence=["#3b82f6"],
                                  labels={"health_score": "Health Score"})
            fig_hs.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f3f4f6", family="Space Grotesk"), height=260,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#9ca3af"), 
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#9ca3af"),
            )
            st.plotly_chart(fig_hs, use_container_width=True, key="fleet_hs_hist")

        with ch2:
            st.markdown("**Failure Risk Distribution**")
            fig_fp = px.histogram(df_result, x="failure_prob_pct", nbins=20,
                                  color_discrete_sequence=["#ef4444"],
                                  labels={"failure_prob_pct": "Failure Risk (%)"})
            fig_fp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f3f4f6", family="Space Grotesk"), height=260,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#9ca3af"), 
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#9ca3af"),
            )
            st.plotly_chart(fig_fp, use_container_width=True, key="fleet_fp_hist")

        # Scatter
        st.markdown("**Health Score vs Failure Risk (All Fleet Vehicles)**")
        fig_sc = px.scatter(df_result, x="health_score", y="failure_prob_pct",
                            color="prediction",
                            color_discrete_map={0: "#10b981", 1: "#ef4444"},
                            labels={"health_score": "Health Score",
                                    "failure_prob_pct": "Failure Risk (%)",
                                    "prediction": "Failure Imminent"},
                            opacity=0.75, height=360)
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f3f4f6", family="Space Grotesk"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#9ca3af"), 
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#9ca3af"),
            margin=dict(t=10),
        )
        st.plotly_chart(fig_sc, use_container_width=True, key="fleet_sc")

        # Table — at-risk only
        at_risk_df = df_result[df_result["prediction"] == 1].sort_values(
            "failure_prob_pct", ascending=False).head(20)
        if not at_risk_df.empty:
            st.markdown(f"**🚨 Top At-Risk Fleet Assets (showing top 20 of {n_fail} at risk)**")
            disp_cols = [c for c in ["engine_temp_C","vibration_ms2","battery_voltage_V",
                                      "engine_load_pct","health_score",
                                      "failure_prob_pct","rul_days"] if c in at_risk_df.columns]
            st.dataframe(
                at_risk_df[disp_cols].rename(columns={
                    "engine_temp_C": "Engine Temp (°C)",
                    "vibration_ms2": "Vibration (m/s²)",
                    "battery_voltage_V": "Battery (V)",
                    "engine_load_pct": "Load %",
                    "health_score": "Health Score",
                    "failure_prob_pct": "Failure Risk %",
                    "rul_days": "RUL (Days)"
                }).style.background_gradient(subset=["Failure Risk %"], cmap="RdYlGn_r"),
                use_container_width=True
            )

        # Download
        csv_out = df_result.to_csv(index=False).encode()
        st.download_button("⬇️ Download Full Fleet Predictions CSV", csv_out,
                           "autoguard_fleet_predictions.csv", "text/csv",
                           use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Interactive User Manual
# ══════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.markdown("### 📖 AutoGuard AI Technical & Operating Manual")
    st.markdown("Welcome to the AutoGuard AI knowledge base. This guide outlines how the predictive maintenance framework interprets vehicle telematics.")
    
    manual_sub = st.selectbox("Select Manual Section", [
        "🗺️ Telematics Processing Pipeline", 
        "📡 Sensor & OBD-II Directory", 
        "🎮 Telemetry Simulator (Sandbox Training)",
        "❓ Frequently Asked Questions"
    ])
    
    if manual_sub == "🗺️ Telematics Processing Pipeline":
        st.markdown("#### 🗺️ How AutoGuard AI Processes Telemetry")
        st.markdown("The pipeline transforms raw, noisy telematics into actionable maintenance work orders through four primary phases:")
        
        st.markdown("""
        <div class="step-card">
            <div class="step-number">01</div>
            <div style="font-weight: 700; font-size: 15px; color: #fff; margin-bottom: 6px;">Continuous Sensor Gathering</div>
            <div style="font-size: 13px; color: #9ca3af; line-height: 1.5;">
                Vehicle Telematics Gateways collect 9 physical sensor streams from the Controller Area Network (CAN bus) at high frequency. Sensor data includes temperatures, mechanical vibrations, electrical voltages, and rotational dynamics.
            </div>
        </div>
        
        <div class="step-card">
            <div class="step-number">02</div>
            <div style="font-weight: 700; font-size: 15px; color: #fff; margin-bottom: 6px;">Real-Time Feature Engineering</div>
            <div style="font-size: 13px; color: #9ca3af; line-height: 1.5;">
                Raw readings are transformed into stress interaction features. For example, high engine load isn't dangerous on its own, but combined with elevated engine temperatures, it raises the <strong>Thermal Stress Index</strong>, which indicates cooling efficiency breakdown.
            </div>
        </div>
        
        <div class="step-card">
            <div class="step-number">03</div>
            <div style="font-weight: 700; font-size: 15px; color: #fff; margin-bottom: 6px;">AI Machine Learning Classifiers</div>
            <div style="font-size: 13px; color: #9ca3af; line-height: 1.5;">
                An ensemble of <strong>Random Forest</strong> and <strong>XGBoost</strong> classifiers evaluates the engineered feature array. It estimates the statistical probability of breakdown within the next 30 days and calculates the vehicle's Remaining Useful Life (RUL).
            </div>
        </div>
        
        <div class="step-card">
            <div class="step-number">04</div>
            <div style="font-weight: 700; font-size: 15px; color: #fff; margin-bottom: 6px;">Actionable Work Tickets & OBD Mapping</div>
            <div style="font-size: 13px; color: #9ca3af; line-height: 1.5;">
                Identified faults are mapped directly to standard OBD-II Diagnostic Trouble Codes (DTCs) and compiled into priority-coded mechanic work orders, helping mechanics start repairs immediately without manual inspection.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    elif manual_sub == "📡 Sensor & OBD-II Directory":
        st.markdown("#### 📡 Sensor & OBD-II DTC Reference Directory")
        st.markdown("Select a sensor below to explore its target operating parameters, warning thresholds, and associated OBD-II codes.")
        
        selected_sensor = st.selectbox("Select Sensor Stream", [
            "Engine Temperature",
            "Vibration Level",
            "Battery Voltage",
            "Engine Load",
            "Oil Pressure",
            "Coolant Temperature"
        ])
        
        if selected_sensor == "Engine Temperature":
            st.markdown("""
            <div class="glass-card card-danger">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; font-size: 18px; color: #fff;">🌡️ Engine Temperature</h3>
                    <span class="dtc-badge">OBD CODE: P0217</span>
                </div>
                <div class="sensor-deep-dive">
                    <p style="font-size: 13px; line-height: 1.5; color: #cbd5e1;">
                        Monitors the internal operating temperature of the engine block. Heat management is critical to prevent metal warping and component seizure.
                    </p>
                    <table style="width: 100%; font-size: 12px; margin-top: 10px; color: #9ca3af;">
                        <tr><td style="padding: 4px 0;"><strong>Normal Range:</strong></td><td style="color: #10b981;">80.0°C – 100.0°C</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Warning Level:</strong></td><td style="color: #f59e0b;">> 102.0°C</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Critical Failure Threshold:</strong></td><td style="color: #ef4444;">> 110.0°C (Triggers P0217)</td></tr>
                    </table>
                    <p style="font-size: 12px; margin-top: 12px; color: #9ca3af;">
                        <strong>Common Causes of Anomalies:</strong> Low radiator coolant, leaking coolant hoses, stuck thermostat, radiator fan electrical failure, or oil pump failure.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif selected_sensor == "Vibration Level":
            st.markdown("""
            <div class="glass-card card-danger">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; font-size: 18px; color: #fff;">🔧 Vibration Level</h3>
                    <span class="dtc-badge">OBD CODE: P0300</span>
                </div>
                <div class="sensor-deep-dive">
                    <p style="font-size: 13px; line-height: 1.5; color: #cbd5e1;">
                        Measures tri-axial mechanical vibration intensity on the engine mount. High frequency oscillations indicate mechanical instability.
                    </p>
                    <table style="width: 100%; font-size: 12px; margin-top: 10px; color: #9ca3af;">
                        <tr><td style="padding: 4px 0;"><strong>Normal Range:</strong></td><td style="color: #10b981;">0.1 – 0.6 m/s²</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Warning Level:</strong></td><td style="color: #f59e0b;">> 0.75 m/s²</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Critical Failure Threshold:</strong></td><td style="color: #ef4444;">> 0.90 m/s² (Triggers P0300)</td></tr>
                    </table>
                    <p style="font-size: 12px; margin-top: 12px; color: #9ca3af;">
                        <strong>Common Causes of Anomalies:</strong> Engine cylinder misfires, deteriorated motor mounts, bent driveshaft, loose pulley, or dry suspension linkages.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif selected_sensor == "Battery Voltage":
            st.markdown("""
            <div class="glass-card card-warning">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; font-size: 18px; color: #fff;">🔋 Battery Voltage</h3>
                    <span class="dtc-badge">OBD CODE: P0562</span>
                </div>
                <div class="sensor-deep-dive">
                    <p style="font-size: 13px; line-height: 1.5; color: #cbd5e1;">
                        Monitors the electrical potential across the vehicle charging system. Under alternator operation, voltage should hover above nominal battery charge.
                    </p>
                    <table style="width: 100%; font-size: 12px; margin-top: 10px; color: #9ca3af;">
                        <tr><td style="padding: 4px 0;"><strong>Normal Range:</strong></td><td style="color: #10b981;">12.6V – 14.4V (running)</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Warning Level:</strong></td><td style="color: #f59e0b;">< 12.0V</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Critical Failure Threshold:</strong></td><td style="color: #ef4444;">< 11.5V (Triggers P0562)</td></tr>
                    </table>
                    <p style="font-size: 12px; margin-top: 12px; color: #9ca3af;">
                        <strong>Common Causes of Anomalies:</strong> Worn alternator brushes, failed voltage regulator, internal battery cell short-circuit, corroded battery terminals, or excessive parasitic ignition-off load.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif selected_sensor == "Engine Load":
            st.markdown("""
            <div class="glass-card card-warning">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; font-size: 18px; color: #fff;">⚙️ Engine Load</h3>
                    <span class="dtc-badge">OBD CODE: P115D</span>
                </div>
                <div class="sensor-deep-dive">
                    <p style="font-size: 13px; line-height: 1.5; color: #cbd5e1;">
                        Indicates the percentage of the engine's peak torque capacity currently being utilized. High load for prolonged durations accelerates heat build-up.
                    </p>
                    <table style="width: 100%; font-size: 12px; margin-top: 10px; color: #9ca3af;">
                        <tr><td style="padding: 4px 0;"><strong>Normal Range:</strong></td><td style="color: #10b981;">10% – 70% (typical driving)</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Warning Level:</strong></td><td style="color: #f59e0b;">> 80%</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Critical Failure Threshold:</strong></td><td style="color: #ef4444;">> 85% (Triggers P115D)</td></tr>
                    </table>
                    <p style="font-size: 12px; margin-top: 12px; color: #9ca3af;">
                        <strong>Common Causes of Anomalies:</strong> Vehicle overload, driving up steep inclines under high load, transmission slip, clogged catalytic converter, or dirty mass air flow (MAF) sensor.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif selected_sensor == "Oil Pressure":
            st.markdown("""
            <div class="glass-card card-danger">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; font-size: 18px; color: #fff;">🛢️ Oil Pressure</h3>
                    <span class="dtc-badge">OBD CODE: P0522</span>
                </div>
                <div class="sensor-deep-dive">
                    <p style="font-size: 13px; line-height: 1.5; color: #cbd5e1;">
                        Measures the pressure of the motor oil circulating through the engine oil galleys. Low oil pressure starvation will destroy bearings rapidly.
                    </p>
                    <table style="width: 100%; font-size: 12px; margin-top: 10px; color: #9ca3af;">
                        <tr><td style="padding: 4px 0;"><strong>Normal Range:</strong></td><td style="color: #10b981;">30 – 60 psi</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Warning Level:</strong></td><td style="color: #f59e0b;">< 28 psi</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Critical Failure Threshold:</strong></td><td style="color: #ef4444;">< 25 psi (Triggers P0522)</td></tr>
                    </table>
                    <p style="font-size: 12px; margin-top: 12px; color: #9ca3af;">
                        <strong>Common Causes of Anomalies:</strong> Extremely low engine oil level, oil pump wear, clogged oil filter, oil dilution from fuel leak, or failed oil pressure sending switch.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif selected_sensor == "Coolant Temperature":
            st.markdown("""
            <div class="glass-card card-danger">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; font-size: 18px; color: #fff;">💧 Coolant Temperature</h3>
                    <span class="dtc-badge">OBD CODE: P0117</span>
                </div>
                <div class="sensor-deep-dive">
                    <p style="font-size: 13px; line-height: 1.5; color: #cbd5e1;">
                        Measures the temperature of the liquid coolant mixture flowing through radiator channels. Coolant issues map to high block temperatures.
                    </p>
                    <table style="width: 100%; font-size: 12px; margin-top: 10px; color: #9ca3af;">
                        <tr><td style="padding: 4px 0;"><strong>Normal Range:</strong></td><td style="color: #10b981;">80.0°C – 95.0°C</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Warning Level:</strong></td><td style="color: #f59e0b;">> 100.0°C</td></tr>
                        <tr><td style="padding: 4px 0;"><strong>Critical Failure Threshold:</strong></td><td style="color: #ef4444;">> 110.0°C (Triggers P0117)</td></tr>
                    </table>
                    <p style="font-size: 12px; margin-top: 12px; color: #9ca3af;">
                        <strong>Common Causes of Anomalies:</strong> Failed thermostat, radiator blockages, cracked water pump impeller, air pockets in cooling channels, or cooling fan relay failure.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif manual_sub == "🎮 Telemetry Simulator (Sandbox Training)":
        st.markdown("#### 🎮 Telemetry Simulator (Sandbox)")
        st.markdown("Use this micro-sandbox to play with telemetry sensor values and observe the model prediction, RUL, and DTC alerts in real time without clicking 'Run Telematics Diagnosis'.")
        
        sim_col1, sim_col2 = st.columns([1, 1])
        
        with sim_col1:
            st.markdown("##### ⚙️ Adjust Telemetry Inputs")
            s_temp = st.slider("Engine Temp (°C)", 60.0, 160.0, 90.0, 1.0, key="sim_temp")
            s_vib = st.slider("Vibration (m/s²)", 0.1, 2.0, 0.4, 0.05, key="sim_vib")
            s_volt = st.slider("Battery Voltage (V)", 9.0, 15.0, 13.5, 0.1, key="sim_volt")
            s_load = st.slider("Engine Load (%)", 10.0, 100.0, 50.0, 2.0, key="sim_load")
            s_rpm = st.slider("RPM", 500, 6000, 2000, 100, key="sim_rpm")
            s_oil = st.slider("Oil Pressure (psi)", 10.0, 80.0, 45.0, 1.0, key="sim_oil")
            s_cool = st.slider("Coolant Temp (°C)", 70.0, 140.0, 90.0, 1.0, key="sim_cool")
            s_age = st.slider("Vehicle Age (years)", 0.5, 15.0, 3.0, 0.5, key="sim_age")
            s_mileage = st.number_input("Mileage (km)", 1000, 400000, 50000, 2000, key="sim_mileage")
            
        with sim_col2:
            st.markdown("##### ⚡ Real-Time Pipeline Response")
            
            sim_row = {
                "engine_temp_C": s_temp, "vibration_ms2": s_vib,
                "battery_voltage_V": s_volt, "engine_load_pct": s_load,
                "rpm": s_rpm, "oil_pressure_psi": s_oil,
                "coolant_temp_C": s_cool, "vehicle_age_years": s_age,
                "mileage_km": s_mileage,
            }
            
            sim_res = predict(sim_row)
            
            sim_hs = sim_res["health_score"]
            sim_fp = sim_res["failure_prob"]
            sim_rul = sim_res["rul_days"]
            sim_pred = sim_res["prediction"]
            
            sh_color = health_color(sim_hs)
            sr_color = risk_color(sim_fp)
            srul_color = "#10b981" if sim_rul > 180 else "#f59e0b" if sim_rul > 60 else "#ef4444"
            
            status_box_bg = "rgba(16, 185, 129, 0.15)" if sim_pred == 0 else "rgba(239, 68, 68, 0.15)"
            status_box_border = "#10b981" if sim_pred == 0 else "#ef4444"
            status_text = "SYSTEM HEALTHY" if sim_pred == 0 else "CRITICAL BREAKDOWN RISK"
            
            st.markdown(f"""
            <div style="background: {status_box_bg}; border: 1px solid {status_box_border}; border-radius: 12px; padding: 15px; text-align: center; margin-bottom: 20px;">
                <span style="font-size: 11px; font-weight: 800; color: #9ca3af; text-transform: uppercase;">Real-Time Status Alert</span>
                <h4 style="margin: 5px 0 0; color: #fff; font-family: 'Space Grotesk', sans-serif;">{status_text}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            c_hs, c_fp, c_rul = st.columns(3)
            with c_hs:
                st.markdown(f"""
                <div style="background: rgba(31, 41, 55, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 10px; text-transform: uppercase; color: #9ca3af;">Health Score</div>
                    <div style="font-size: 24px; font-weight: 800; color: {sh_color}; margin-top: 4px;">{sim_hs}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_fp:
                st.markdown(f"""
                <div style="background: rgba(31, 41, 55, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 10px; text-transform: uppercase; color: #9ca3af;">Failure Risk</div>
                    <div style="font-size: 24px; font-weight: 800; color: {sr_color}; margin-top: 4px;">{sim_fp}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c_rul:
                st.markdown(f"""
                <div style="background: rgba(31, 41, 55, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 10px; text-transform: uppercase; color: #9ca3af;">RUL Days</div>
                    <div style="font-size: 24px; font-weight: 800; color: {srul_color}; margin-top: 4px;">{sim_rul}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br><strong>Active Diagnostic Trouble Codes:</strong>", unsafe_allow_html=True)
            sim_recs = sim_res["recommendations"]
            if sim_recs:
                for item in sim_recs:
                    # Defensive: handle both dict (new format) and plain string (legacy)
                    if isinstance(item, str):
                        item = {"text": item, "priority": "ROUTINE", "code": "INFO", "icon": "ℹ️"}
                    s_badge = "badge-custom-danger" if item["priority"] == "CRITICAL" else "badge-custom-warning" if item["priority"] == "WARNING" else "badge-custom-info"
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(31, 41, 55, 0.3); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 8px 12px; margin-bottom: 6px;">
                        <span style="font-size: 12px; color: #f3f4f6;">{item['icon']} {item['code']}</span>
                        <span class="badge-custom {s_badge}" style="font-size: 8px; padding: 2px 6px;">{item['priority']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:12px; color:#9ca3af;'>No fault codes detected. System normal.</div>", unsafe_allow_html=True)

    elif manual_sub == "❓ Frequently Asked Questions":
        st.markdown("#### ❓ Telematics System FAQ")
        
        with st.expander("Q: How does AutoGuard AI predict Remaining Useful Life (RUL)?"):
            st.markdown("""
            AutoGuard AI applies an exponential decay algorithm anchored to physical baseline lifespans (365 days) and scaled dynamically by two telemetry vectors:
            1. **Accumulated wear factors** (such as vehicle age and mileage).
            2. **Imminent failure probabilities** generated in real-time by our classifier ensemble.
            As battery potential declines or temperatures cross warning thresholds, the probability of failure surges, and the RUL adjusts exponentially.
            """)
            
        with st.expander("Q: What classification models are loaded by the active backend?"):
            st.markdown("""
            AutoGuard AI trains three classifiers:
            - **Random Forest Classifier**: Best suited for robust, stable predictions and identifying features correlations.
            - **XGBoost (Extreme Gradient Boosting)**: Excels at capturing high-dimensional non-linear interactions between engineered metrics.
            - **Decision Tree Classifier**: Included for benchmark profiling.
            
            The system evaluates F1-scores on test sets automatically and loads the top performer to make real-time predictions.
            """)
            
        with st.expander("Q: Can AutoGuard be integrated on-edge in telematics hardware?"):
            st.markdown("""
            Yes. The backend uses `joblib` serialized scikit-learn and XGBoost pipelines, which can compile into lightweight representations or C/C++ runtimes. This allows deployment directly on-board vehicle gateways or telematics control units (TCUs) to process CAN bus telemetry locally.
            """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ML Engine Performance Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("### 🧠 ML Telematics Engine Performance Insights")
    st.markdown("Detailed breakdown of model evaluations, features engineering, and training characteristics.")
    
    try:
        with open("models/results.json") as f:
            results = json.load(f)
        
        st.markdown("#### 🏆 Model Performance Comparison")
        res_data = []
        for model_name, metrics in results.items():
            res_data.append({
                "Model Classifier": model_name,
                "Accuracy (%)": f"{metrics['accuracy']*100:.2f}%",
                "Precision (%)": f"{metrics['precision']*100:.2f}%",
                "Recall (%)": f"{metrics['recall']*100:.2f}%",
                "F1 Score (%)": f"{metrics['f1']*100:.2f}%",
                "ROC-AUC": f"{metrics['roc_auc']:.4f}"
            })
        st.dataframe(pd.DataFrame(res_data), use_container_width=True)
    except Exception:
        st.warning("⚠️ Model comparison results file (models/results.json) not found.")

    st.markdown("---")
    
    st.markdown("#### 🔬 AI-Engineered Interaction Features")
    st.markdown("Rather than relying solely on raw sensors, AutoGuard constructs interaction indices to improve predictive accuracy:")
    st.markdown("""
    - **Temp/Load Ratio (`temp_load_ratio`)**: Computes temperature relative to engine load. Elevated temperatures during low loads indicate significant heat dispersion inefficiencies.
    - **Vibration/RPM Ratio (`vib_rpm_ratio`)**: Captures mechanical friction stress. Distinguishes structural mounts wear from normal engine speed harmonics.
    - **Battery × Age Interaction (`batt_age_interaction`)**: Adjusts battery voltage thresholds. Accounts for starter resistance increases in older vehicles.
    - **Thermal Stress Index (`thermal_stress`)**: Computes the average of engine block temperature and liquid coolant temperature to assess cooling loop performance.
    """)

    st.markdown("---")
    
    st.markdown("#### 📊 Evaluation Visualizations")
    try:
        with open("models/results.json") as f:
            results = json.load(f)
        best = max(results, key=lambda k: results[k]["f1"])
        
        model_slug = "random_forest"
        if "xgboost" in best.lower():
            model_slug = "xgboost"
        elif "decision" in best.lower():
            model_slug = "decision_tree"
            
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(f"assets/fi_{model_slug}.png", caption=f"Feature Importances ({best})", use_container_width=True)
        with col_img2:
            st.image(f"assets/cm_{model_slug}.png", caption=f"Confusion Matrix ({best})", use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("assets/model_comparison.png", caption="Model Algorithm Comparison", use_container_width=True)
    except Exception as e:
        st.info(f"Could not load evaluation plots: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#4b5563;font-size:12px;padding:8px 0; font-family: 'Space Grotesk', sans-serif;">
    AutoGuard AI &nbsp;|&nbsp; Predictive Maintenance Telematics System &nbsp;|&nbsp;
    Powered by Random Forest + XGBoost &nbsp;|&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)

