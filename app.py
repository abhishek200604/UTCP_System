"""
app.py — Streamlit Web Application
Urban Traffic Congestion Prediction Under Sparse and Noisy Sensor Data

Three-page dashboard: Prediction Interface · Pune Traffic Map · Spatio-Temporal Analysis
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from utils import (
    load_data, engineer_features, smooth_noise, get_feature_columns,
    EVENT_IMPACT_MAP, WEATHER_MAP, CONGESTION_MAP,
)

# ──────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────
st.set_page_config(
    page_title="UTCP System — Traffic Congestion Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────
# TERMINAL STARTUP BANNER (prints once)
# ──────────────────────────────────────
_C = "\033[96m"; _G = "\033[92m"; _M = "\033[95m"; _B = "\033[1m"; _D = "\033[2m"; _R = "\033[0m"
if "banner_shown" not in st.session_state:
    st.session_state.banner_shown = True
    _W = 52

    def _cpad(txt, w):
        import unicodedata
        ex = sum(1 for c in txt if unicodedata.east_asian_width(c) in ('W','F') or ord(c) > 0xFFFF)
        p = w - len(txt) - ex; l = p // 2; r = p - l
        return ' ' * l + txt + ' ' * r

    print()
    print(f"  {_C}{_B}╔{'═' * _W}╗{_R}")
    print(f"  {_C}{_B}║{' ' * _W}║{_R}")
    print(f"  {_C}{_B}║{_cpad('UTCP System  —  Dashboard', _W)}║{_R}")
    print(f"  {_C}{_B}║{_cpad('Urban Traffic Congestion Prediction', _W)}║{_R}")
    print(f"  {_C}{_B}║{' ' * _W}║{_R}")
    print(f"  {_C}{_B}╚{'═' * _W}╝{_R}")
    print()
    print(f"  {_D}▸ Pages:   3 (Predict · Map · Analysis){_R}")
    print(f"  {_D}▸ Dataset:  Pune · 20 Areas · January 2025{_R}")
    print(f"  {_D}▸ Model:    Linear Regression  (R² = 1.0){_R}")
    print()
    print(f"  {_G}{_B}✅ Dashboard ready — open the URL above in your browser{_R}")
    print()

# ──────────────────────────────────────
# HELPER: convert 12h + AM/PM → 24h
# ──────────────────────────────────────
def hour12_to_24(h12: int, ampm: str) -> int:
    """Convert 12-hour value (1-12) + 'AM'/'PM' string to 0-23."""
    is_pm = ampm == "PM"
    if not is_pm:
        return 0 if h12 == 12 else h12
    else:
        return 12 if h12 == 12 else h12 + 12

st.markdown("""
<style>
    /* ====== THEME VARIABLES ====== */
    :root {
        --header-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --header-text: #ffffff;
        --card-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --card-text: #ffffff;
        --card-shadow: rgba(102, 126, 234, 0.25);
        --pred-low-bg: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --pred-low-text: #fff;
        --pred-medium-bg: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        --pred-medium-text: #1a3a0a;
        --pred-high-bg: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        --pred-high-text: #4a2f00;
        --pred-severe-bg: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        --pred-severe-text: #fff;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --header-bg: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            --card-bg: linear-gradient(135deg, #4a3f8a 0%, #6c3bb5 100%);
            --card-shadow: rgba(74, 63, 138, 0.45);
        }
    }
    [data-testid="stAppViewContainer"][data-theme="dark"],
    .stApp[data-theme="dark"] {
        --header-bg: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        --card-bg: linear-gradient(135deg, #4a3f8a 0%, #6c3bb5 100%);
        --card-shadow: rgba(74, 63, 138, 0.45);
    }

    /* ====== HEADER ====== */
    .main-header {
        background: var(--header-bg);
        padding: 1.8rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: var(--header-text);
        text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; margin-bottom: 0.3rem; }
    .main-header p  { opacity: 0.85; font-size: 1.05rem; }

    /* ====== METRIC CARDS ====== */
    .metric-card {
        background: var(--card-bg);
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        color: var(--card-text);
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px var(--card-shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px var(--card-shadow);
    }
    .metric-card h3 { font-size: 1.8rem; margin: 0; }
    .metric-card p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }

    /* ====== SIDEBAR ====== */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem;
    }
    .sidebar-brand {
        text-align: center;
        padding: 1.2rem 1rem 0.8rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-brand h2 {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .sidebar-brand .brand-sub {
        font-size: 0.78rem;
        opacity: 0.6;
        margin-top: 0.2rem;
    }
    .nav-divider {
        border: none;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        margin: 0.8rem 0;
    }

    /* ====== PREDICTION BOX ====== */
    .pred-box {
        background: var(--pred-low-bg);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        color: var(--pred-low-text);
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(17, 153, 142, 0.35);
    }
    .pred-box h2, .pred-box h3 { color: inherit; }
    .pred-box.severe {
        background: var(--pred-severe-bg);
        color: var(--pred-severe-text);
        box-shadow: 0 4px 20px rgba(235, 51, 73, 0.35);
    }
    .pred-box.high {
        background: var(--pred-high-bg);
        color: var(--pred-high-text);
        box-shadow: 0 4px 20px rgba(247, 151, 30, 0.35);
    }
    .pred-box.medium {
        background: var(--pred-medium-bg);
        color: var(--pred-medium-text);
        box-shadow: 0 4px 20px rgba(86, 171, 47, 0.35);
    }

    /* ====== RESPONSIVE ====== */
    @media (max-width: 768px) {
        .main-header { padding: 1.2rem 1rem; border-radius: 12px; }
        .main-header h1 { font-size: 1.4rem; }
        .metric-card { padding: 0.8rem 1rem; border-radius: 10px; }
        .metric-card h3 { font-size: 1.3rem; }
        .metric-card p  { font-size: 0.78rem; }
        .pred-box { padding: 1rem 1.2rem; border-radius: 12px; }
        .pred-box h2 { font-size: 1.2rem !important; }
        .pred-box h3 { font-size: 1rem !important; }
        /* Stack columns on small screens */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-header h1 { font-size: 1.8rem; }
        .metric-card h3 { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────
# DATA & MODEL LOADING (cached)
# ──────────────────────────────────────
DATA_PATH = "data/pune_january_20areas.csv"
MODEL_PATH = "models/congestion_model.pkl"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "models/model_metrics.json"


@st.cache_data(show_spinner="Loading dataset …")
def cached_load():
    df = load_data(DATA_PATH)
    df = smooth_noise(df)
    df, le_area = engineer_features(df)
    return df, le_area


@st.cache_resource(show_spinner="Loading model …")
def cached_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        return model, scaler
    return None, None




def score_to_level(score):
    if score >= 75:
        return "Severe"
    elif score >= 55:
        return "High"
    elif score >= 35:
        return "Medium"
    else:
        return "Low"


# ──────────────────────────────────────
# SIDEBAR NAVIGATION  (styled buttons)
# ──────────────────────────────────────
NAV_PAGES = [
    ("🔮", "Prediction Interface"),
    ("🗺️", "Pune Traffic Map"),
    ("📈", "Spatio-Temporal Analysis"),
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "Prediction Interface"

# Brand header
st.sidebar.markdown(
    '<div class="sidebar-brand">'
    '<h2>🚦 UTCP System</h2>'
    '<div class="brand-sub">Urban Traffic Congestion Prediction</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown('<hr class="nav-divider">', unsafe_allow_html=True)

# Render navigation buttons
for icon, label in NAV_PAGES:
    is_active = st.session_state.active_page == label
    if st.sidebar.button(
        f"{icon}  {label}",
        key=f"nav_{label}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.active_page = label
        st.rerun()

page = st.session_state.active_page

# ──────────────────────────────────────
# HEADER  (clean banner)
# ──────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    '<h1>🚦 Urban Traffic Congestion Prediction</h1>'
    '</div>',
    unsafe_allow_html=True,
)

# Load data once
df, le_area = cached_load()

# ══════════════════════════════════════
# PAGE 1: PUNE TRAFFIC MAP
# ══════════════════════════════════════
if page == "Pune Traffic Map":
    st.subheader("🗺️ Pune Traffic Congestion Map")

    mc1, mc2 = st.columns([3, 1])
    with mc1:
        map_h = st.number_input("🕐 Hour", min_value=1, max_value=12, value=9, step=1, key="map_h")
    with mc2:
        map_ampm = st.selectbox("AM / PM", ["AM", "PM"], key="map_ampm")
    hour_sel = hour12_to_24(map_h, map_ampm)

    map_df = df[df["Hour"] == hour_sel].copy()

    # Aggregate per area for the selected hour
    agg = (
        map_df.groupby("Area_Name")
        .agg(
            lat=("Latitude", "first"),
            lon=("Longitude", "first"),
            score=("Final_Congestion_Score", "mean"),
            vol=("Traffic_Volume", "mean"),
            speed=("Avg_Speed_kmph", "mean"),
        )
        .reset_index()
    )

    color_map = {
        "Low": "#2ecc71",
        "Medium": "#f1c40f",
        "High": "#e67e22",
        "Severe": "#e74c3c",
    }
    level_order = ["Low", "Medium", "High", "Severe"]
    agg["level"] = pd.Categorical(
        agg["score"].apply(score_to_level),
        categories=level_order,
        ordered=True,
    )

    map_style = "carto-darkmatter"
    fig = px.scatter_map(
        agg,
        lat="lat",
        lon="lon",
        size="score",
        color="level",
        color_discrete_map=color_map,
        category_orders={"level": level_order},
        hover_name="Area_Name",
        hover_data={"score": ":.1f", "vol": ":.0f", "speed": ":.1f", "lat": False, "lon": False},
        size_max=25,
        zoom=11,
        map_style=map_style,
        title=f"Congestion at {map_h} {map_ampm}",
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Legend / info
    st.markdown("**Color legend:** 🟢 Low  🟡 Medium  🟠 High  🔴 Severe")

# ══════════════════════════════════════
# PAGE 2: SPATIO-TEMPORAL ANALYSIS
# ══════════════════════════════════════
elif page == "Spatio-Temporal Analysis":
    st.subheader("📈 Spatio-Temporal Analysis")

    tab1, tab2, tab3 = st.tabs(["🔥 Heatmap", "📊 Area Comparison", "📈 Hourly Trend"])

    with tab1:
        pivot = df.pivot_table(
            values="Final_Congestion_Score",
            index="Area_Name",
            columns="Hour",
            aggfunc="mean",
        )
        fig = px.imshow(
            pivot,
            color_continuous_scale="YlOrRd",
            labels=dict(x="Hour", y="Area", color="Score"),
            aspect="auto",
            title="Avg Congestion by Area & Hour",
        )
        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        area_avg = df.groupby("Area_Name")["Final_Congestion_Score"].mean().sort_values(ascending=True)
        fig2 = px.bar(
            x=area_avg.values,
            y=area_avg.index,
            orientation="h",
            color=area_avg.values,
            color_continuous_scale="RdYlGn_r",
            labels={"x": "Avg Score", "y": ""},
            title="Average Congestion by Area",
        )
        fig2.update_layout(
            height=max(400, len(area_avg) * 28),
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(automargin=True),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        hourly = df.groupby("Hour")["Final_Congestion_Score"].mean()
        fig3 = px.line(
            x=hourly.index,
            y=hourly.values,
            markers=True,
            labels={"x": "Hour", "y": "Avg Score"},
            title="City-wide Hourly Congestion Trend",
        )
        fig3.update_traces(line=dict(color="#764ba2", width=3))
        fig3.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(dtick=2),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════
# PAGE 3: PREDICTION INTERFACE
# ══════════════════════════════════════
elif page == "Prediction Interface":
    st.subheader("🔮 Congestion Prediction")

    model, scaler = cached_model()
    if model is None:
        st.warning("⚠️ No trained model found. Run `python train_model.py` first.")
        st.stop()

    # ── Location & Time ──
    with st.container(border=True):
        st.markdown("📍 **Location & Time**")
        pred_area = st.selectbox("Area", sorted(df["Area_Name"].unique()), key="pi_area")
        col_h, col_ap, col_dom = st.columns([3, 1, 3])
        with col_h:
            pred_h12 = st.number_input("🕐 Hour", min_value=1, max_value=12, value=9, step=1, key="pi_h12")
        with col_ap:
            pred_ampm = st.selectbox("AM / PM", ["AM", "PM"], key="pi_ampm")
        with col_dom:
            pred_dom = st.number_input("📅 Day of Month", min_value=1, max_value=31, value=15, step=1, key="pi_dom")
        pred_hour = hour12_to_24(pred_h12, pred_ampm)
        DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pred_dow = DAY_NAMES.index(
            st.radio("📆 Day of Week", DAY_NAMES, index=0, horizontal=True, key="pi_dow")
        )

    # ── Conditions ──
    with st.container(border=True):
        st.markdown("🌦️ **Conditions**")
        pred_weather = st.radio("Weather", list(WEATHER_MAP.keys()), horizontal=True, key="pi_weather")
        pred_event = st.radio("Event Impact", list(EVENT_IMPACT_MAP.keys()), horizontal=True, key="pi_event")

    # ── Situational Flags ──
    with st.container(border=True):
        st.markdown("⚡ **Situational Flags**")
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            pred_holiday = "Yes" if st.toggle("🏖️ Holiday", key="pi_hol") else "No"
        with tc2:
            pred_weekend = "Yes" if st.toggle("📅 Weekend", key="pi_wkd") else "No"
        with tc3:
            pred_accident = "Yes" if st.toggle("🚨 Accident", key="pi_acc") else "No"
        with tc4:
            pred_roadwork = "Yes" if st.toggle("🚧 Road Work", key="pi_rw") else "No"

    if st.button("🔮 **Predict Congestion**", type="primary", width="stretch"):
        # Build feature vector — estimate sensor values from condition-matched history
        area_enc = le_area.transform([pred_area])[0]
        area_mask = df["Area_Name"] == pred_area
        lat = df.loc[area_mask, "Latitude"].iloc[0]
        lon = df.loc[area_mask, "Longitude"].iloc[0]
        area_cong_idx = df.loc[area_mask, "Area_Congestion_Index"].mean()
        rolling_cong = df.loc[area_mask, "Rolling_Congestion_3h"].mean()

        # Progressive condition-aware filtering:
        # Try exact match first, then relax conditions until enough rows found
        conditions = [
            ("Hour", pred_hour),
            ("Accident", pred_accident),
            ("Road_Work", pred_roadwork),
            ("Weather", pred_weather),
            ("Event_Impact_Level", pred_event),
        ]

        match_mask = area_mask.copy()
        for col, val in conditions:
            candidate = match_mask & (df[col] == val)
            if candidate.sum() >= 1:  # keep filter if at least 1 row matches
                match_mask = candidate

        # Fall back to area-only if nothing matched at all
        if match_mask.sum() == 0:
            match_mask = area_mask

        matched = df.loc[match_mask]
        vol_avg = matched["Traffic_Volume"].mean()
        spd_avg = matched["Avg_Speed_kmph"].mean()
        tscore_avg = matched["Traffic_Score"].mean()
        hts_avg = matched["Holiday_Traffic_Score"].mean()
        cong_num_avg = matched["Congestion_Num"].mean()
        hboost_avg = matched["Holiday_Boost"].mean()

        feature_cols = get_feature_columns()
        row = pd.DataFrame([{
            "Hour": pred_hour,
            "DayOfWeek": pred_dow,
            "DayOfMonth": pred_dom,
            "Is_Weekend_Num": 1 if pred_weekend == "Yes" else 0,
            "Is_Holiday_Num": 1 if pred_holiday == "Yes" else 0,
            "Traffic_Volume": vol_avg,
            "Avg_Speed_kmph": spd_avg,
            "Traffic_Score": tscore_avg,
            "Holiday_Traffic_Score": hts_avg,
            "Event_Impact_Num": EVENT_IMPACT_MAP[pred_event],
            "Weather_Num": WEATHER_MAP[pred_weather],
            "Congestion_Num": cong_num_avg,
            "Accident_Num": 1 if pred_accident == "Yes" else 0,
            "Road_Work_Num": 1 if pred_roadwork == "Yes" else 0,
            "Area_Encoded": area_enc,
            "Area_Congestion_Index": area_cong_idx,
            "Rolling_Congestion_3h": rolling_cong,
            "Area_Hour": area_enc * 100 + pred_hour,
            "Holiday_Boost": hboost_avg,
            "Latitude": lat,
            "Longitude": lon,
        }])

        if scaler is not None:
            row_scaled = pd.DataFrame(scaler.transform(row), columns=feature_cols)
        else:
            row_scaled = row

        prediction = model.predict(row_scaled)[0]
        prediction = float(np.clip(prediction, 0, 100))
        level = score_to_level(prediction)

        css_class = level.lower()
        st.markdown(
            f"""<div class="pred-box {css_class}">
                <h2 style="margin:0">Predicted Congestion Score: {prediction:.1f}</h2>
                <h3 style="margin:0.3rem 0 0 0">Level: {level}</h3>
            </div>""",
            unsafe_allow_html=True,
        )

        # Visual gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text": f"{pred_area} — {pred_hour}:00"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#764ba2"},
                "steps": [
                    {"range": [0, 35], "color": "#2ecc71"},
                    {"range": [35, 55], "color": "#f1c40f"},
                    {"range": [55, 75], "color": "#e67e22"},
                    {"range": [75, 100], "color": "#e74c3c"},
                ],
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")



# ──────────────────────────────────────
# FOOTER
# ──────────────────────────────────────
st.sidebar.markdown('<hr class="nav-divider">', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="text-align:center; opacity:0.5; font-size:0.78rem; padding:0.5rem;">'
    '© UTCP System v1.0'
    '</div>',
    unsafe_allow_html=True,
)
