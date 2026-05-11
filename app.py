import os

import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="FTSE 100 Macro Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Professional styling: works in Light + Dark mode
# -----------------------------
st.markdown("""
<style>
:root {
    --primary:#2563eb; --primary-dark:#1d4ed8; --accent:#ef4444;
    --page-bg:#f8fafc; --text-main:#0f172a; --text-muted:#475569;
    --card-bg:#ffffff; --card-bg-soft:#f1f5f9; --card-border:#cbd5e1;
    --shadow:rgba(15,23,42,0.10);
    --hero-bg-1:#0f172a; --hero-bg-2:#1e293b; --hero-text:#ffffff; --hero-subtext:#dbeafe;
    --success-bg:rgba(22,163,74,0.12); --success-border:rgba(22,163,74,0.35); --success-text:#166534;
    --warn-bg:rgba(217,119,6,0.12); --warn-border:rgba(217,119,6,0.35); --warn-text:#92400e;
}
@media (prefers-color-scheme: dark) {
    :root {
        --page-bg:#0b0f19; --text-main:#f8fafc; --text-muted:#cbd5e1;
        --card-bg:#111827; --card-bg-soft:#1e293b; --card-border:#334155;
        --shadow:rgba(0,0,0,0.35);
        --success-bg:rgba(16,185,129,0.15); --success-border:rgba(16,185,129,0.35); --success-text:#d1fae5;
        --warn-bg:rgba(245,158,11,0.15); --warn-border:rgba(245,158,11,0.35); --warn-text:#fef3c7;
    }
}
.stApp { background:var(--page-bg); color:var(--text-main); }
html, body, [class*="css"] { font-family:"Inter","Segoe UI",Arial,sans-serif; }
.main > div { padding-top:0.5rem; }
.block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1320px; }
h1,h2,h3 { color:var(--text-main)!important; letter-spacing:-0.02em; font-weight:800!important; }

.hero-box {
    padding:1.45rem 1.6rem 1.15rem 1.6rem; border-radius:22px;
    background:linear-gradient(135deg,var(--hero-bg-2),var(--hero-bg-1));
    border:1px solid rgba(148,163,184,0.28); box-shadow:0 18px 40px var(--shadow);
    margin-bottom:1rem;
}
.hero-title { color:var(--hero-text)!important; font-size:clamp(2rem,4vw,2.6rem); font-weight:850; line-height:1.05; margin-bottom:0.45rem; }
.hero-subtitle { color:var(--hero-subtext)!important; font-size:1rem; margin-bottom:0.1rem; }
.section-title { color:var(--text-main)!important; font-size:1.15rem; font-weight:750; margin-top:0.2rem; margin-bottom:0.65rem; }

.glass-card {
    padding:1rem 1rem 0.9rem 1rem; border-radius:20px; background:var(--card-bg);
    border:1px solid var(--card-border); box-shadow:0 14px 30px var(--shadow);
    margin-bottom:1rem; color:var(--text-main)!important;
}
.status-ok,.status-warn {
    padding:0.8rem 0.9rem; border-radius:16px; font-size:0.96rem;
    box-shadow:0 8px 22px var(--shadow); margin-bottom:0.75rem; font-weight:650;
}
.status-ok { background:var(--success-bg); border:1px solid var(--success-border); color:var(--success-text)!important; }
.status-warn { background:var(--warn-bg); border:1px solid var(--warn-border); color:var(--warn-text)!important; }
.small-note { color:var(--text-muted)!important; font-size:0.92rem; line-height:1.5; }

div[data-testid="stMetric"] {
    background:linear-gradient(135deg,var(--card-bg),var(--card-bg-soft));
    border:1px solid var(--card-border); padding:16px 18px; border-radius:18px;
    box-shadow:0 12px 26px var(--shadow); min-height:120px; color:var(--text-main)!important;
}
div[data-testid="stMetricLabel"] { color:var(--text-muted)!important; font-weight:700!important; }
div[data-testid="stMetricValue"] { color:var(--text-main)!important; font-weight:750!important; }
div[data-testid="stMetricDelta"] { font-weight:700!important; }

.stSlider label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    color:var(--text-main)!important; font-weight:650!important;
}
div.stButton > button {
    border-radius:14px; border:1px solid rgba(37,99,235,0.25);
    background:linear-gradient(180deg,var(--primary),var(--primary-dark));
    color:white!important; font-weight:700; box-shadow:0 12px 24px rgba(37,99,235,0.25);
}
div.stButton > button:hover {
    border:1px solid rgba(37,99,235,0.45);
    background:linear-gradient(180deg,#3b82f6,var(--primary)); color:white!important;
}
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid var(--card-border); }
.stTabs [data-baseweb="tab"] { color:var(--text-main)!important; border-radius:12px 12px 0 0; padding-left:14px; padding-right:14px; font-weight:700; }

div[data-testid="stDataFrame"] { border-radius:18px; overflow:hidden; border:1px solid var(--card-border); }
div[data-testid="stExpander"] { border-radius:18px; border:1px solid var(--card-border); overflow:hidden; background:var(--card-bg); }
div[data-testid="stAlert"] { border-radius:16px; }
section[data-testid="stSidebar"] { background:var(--card-bg-soft); border-right:1px solid var(--card-border); }
section[data-testid="stSidebar"] * { color:var(--text-main); }
hr { border-color:var(--card-border); }
@media (max-width:900px) {
    .hero-title { font-size:1.9rem; }
    .hero-subtitle { font-size:0.95rem; }
    .glass-card { padding:0.85rem; }
    div[data-testid="stMetric"] { min-height:105px; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Paths and artifacts
# -----------------------------
MODEL_PATH = "models/best_ridge_tuned.pkl"
COLS_PATH = "models/feature_cols.pkl"
BASE_ROW_PATH = "models/x_base_latest.csv"

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(COLS_PATH)
x_base = pd.read_csv(BASE_ROW_PATH, index_col=0)

# -----------------------------
# Utility functions
# -----------------------------
def predict(x_row: pd.DataFrame) -> float:
    return float(model.predict(x_row[feature_cols])[0])

def show_image(path: str, caption: str) -> None:
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing file: {path}")

def make_line_chart(df: pd.DataFrame, title: str, y_col: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[y_col], mode="lines", name=y_col, line=dict(width=2.5)))
    fig.update_layout(
        title=title, template="plotly_white", height=340,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_title="Date", yaxis_title="Value", showlegend=False
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    return fig

def format_return(val: float) -> str:
    return f"{val:.4%}"

def delta_color_name(val: float) -> str:
    if val > 0:
        return "normal"
    if val < 0:
        return "inverse"
    return "off"

def interpret_return(val: float) -> str:
    if val >= 0.02:
        return "strongly positive"
    if val >= 0.005:
        return "moderately positive"
    if val > -0.005:
        return "broadly neutral"
    if val > -0.02:
        return "moderately negative"
    return "strongly negative"

def change_message(delta: float) -> str:
    if delta > 0.005:
        return "The scenario improves the forecast noticeably relative to the baseline."
    if delta > 0.0:
        return "The scenario slightly improves the forecast relative to the baseline."
    if delta == 0.0:
        return "The scenario does not materially change the forecast."
    if delta > -0.005:
        return "The scenario slightly weakens the forecast relative to the baseline."
    return "The scenario weakens the forecast noticeably relative to the baseline."

@st.cache_data(ttl=60 * 60)
def fetch_ftse_prices_and_returns(start="2020-01-01") -> pd.DataFrame:
    df = yf.download("^FTSE", start=start, progress=False, auto_adjust=True)

    if df.empty:
        raise RuntimeError("yfinance returned no FTSE data.")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", "^FTSE") in df.columns:
            close_series = df[("Close", "^FTSE")]
        else:
            close_series = df["Close"].iloc[:, 0]
    else:
        close_series = df["Close"]

    close = close_series.resample("ME").last()
    ret = close.pct_change()

    out = pd.DataFrame({"ftse_close": close, "ftse_return": ret}).dropna()

    if out.empty:
        raise RuntimeError("FTSE resampling returned no usable monthly data.")

    return out

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Controls")
st.sidebar.caption("Stable deployed version")

refresh = st.sidebar.button("Refresh Live Market Data")
if refresh:
    st.cache_data.clear()

ftse_live_df = None
latest_market_month = None

try:
    ftse_live_df = fetch_ftse_prices_and_returns()
    latest_ftse_return = float(ftse_live_df["ftse_return"].iloc[-1])
    latest_market_month = ftse_live_df.index.max()
    live_status = f"Live FTSE data OK. Latest market month: {latest_market_month.date()}"
except Exception as e:
    latest_ftse_return = float(x_base.iloc[0].get("ftse_return", 0))
    live_status = f"Live FTSE fetch failed; using saved market baseline. Reason: {type(e).__name__}: {e}"

if "failed" in live_status.lower():
    st.sidebar.markdown(f'<div class="status-warn">{live_status}</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<div class="status-ok">{live_status}</div>', unsafe_allow_html=True)

st.sidebar.markdown(
    '<div class="small-note">Macroeconomic inputs are taken from the latest validated saved baseline. '
    'Live market data is refreshed from Yahoo Finance.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">FTSE 100 Macro Intelligence Dashboard</div>
    <div class="hero-subtitle">AI-driven macro-financial forecasting and scenario analysis for academic research.</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Dashboard", "Results & Evidence", "About"])

# ============================================================
# TAB 1: Dashboard
# ============================================================
with tabs[0]:
    st.markdown('<div class="section-title">Forecast Overview</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Scenario shocks")
    s1, s2, s3 = st.columns(3)
    with s1:
        delta_cpi = st.slider("Δ CPI inflation (percentage points)", -2.0, 2.0, 0.0, 0.1)
    with s2:
        delta_bank = st.slider("Δ Bank Rate (percentage points)", -2.0, 2.0, 0.0, 0.1)
    with s3:
        delta_unemp = st.slider("Δ Unemployment rate (percentage points)", -2.0, 2.0, 0.0, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    base_pred = predict(x_base)

    x_scn = x_base.copy()
    if "cpi_inflation_yoy" in x_scn.columns:
        x_scn["cpi_inflation_yoy"] = x_scn["cpi_inflation_yoy"] + delta_cpi
    if "bank_rate" in x_scn.columns:
        x_scn["bank_rate"] = x_scn["bank_rate"] + delta_bank
    if "unemployment_rate" in x_scn.columns:
        x_scn["unemployment_rate"] = x_scn["unemployment_rate"] + delta_unemp

    scn_pred = predict(x_scn)
    change = scn_pred - base_pred

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Forecast metrics")
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Baseline predicted return", format_return(base_pred))
    with k2:
        st.metric("Scenario predicted return", format_return(scn_pred), delta=format_return(change), delta_color=delta_color_name(change))
    with k3:
        st.metric("Prediction change", format_return(change), delta=format_return(change), delta_color=delta_color_name(change))
    with k4:
        st.metric("Latest market month", str(latest_market_month.date()) if latest_market_month is not None else "Saved baseline")
    st.markdown('</div>', unsafe_allow_html=True)

    baseline_text = interpret_return(base_pred)
    scenario_text = interpret_return(scn_pred)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Forecast interpretation")
    st.write(
        f"The baseline model output suggests a **{baseline_text}** next-month FTSE 100 outlook, "
        f"with a predicted return of **{base_pred:.4%}**."
    )
    st.write(
        f"Under the selected macroeconomic scenario, the outlook becomes **{scenario_text}**, "
        f"with a predicted return of **{scn_pred:.4%}**."
    )
    st.write(change_message(change))

    st.subheader("Model summary")
    st.write("""
- Model: Tuned Ridge Regression  
- Target: Next-month FTSE 100 return  
- Features: CPI inflation, unemployment rate, Bank Rate, and lagged macro-financial variables  
- Validation: Time-series-aware train/test split  
- Purpose: Scenario-based forecasting for academic decision support  
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Model-based market outlook")
    if scn_pred > 0.01:
        st.success("Bullish outlook detected")
    elif scn_pred > 0:
        st.info("Slightly positive / stable outlook")
    elif scn_pred > -0.01:
        st.warning("Neutral to slightly negative outlook")
    else:
        st.error("Bearish outlook detected")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Economic interpretation of scenario")
    scenario_points = []

    if delta_cpi > 0:
        scenario_points.append("Higher inflation may reduce market sentiment by increasing cost pressure and uncertainty.")
    elif delta_cpi < 0:
        scenario_points.append("Lower inflation may support sentiment by easing price pressure and improving stability.")

    if delta_bank > 0:
        scenario_points.append("Higher Bank Rate may weigh on returns through tighter financial conditions and a higher discount rate.")
    elif delta_bank < 0:
        scenario_points.append("Lower Bank Rate may support returns through easier financial conditions.")

    if delta_unemp > 0:
        scenario_points.append("Higher unemployment may indicate weaker economic activity and softer earnings expectations.")
    elif delta_unemp < 0:
        scenario_points.append("Lower unemployment may signal stronger economic activity and improved business conditions.")

    if not scenario_points:
        scenario_points.append("No additional macro shock has been applied, so the scenario remains equal to the baseline input.")

    for point in scenario_points:
        st.write(f"- {point}")

    st.subheader("Model limitations")
    st.write("""
- Macroeconomic indicators are slower-moving and may not capture sudden market shocks  
- Model performance depends on feature selection and historical relationships  
- External factors such as geopolitical shocks and firm-specific events are not explicitly included  
- Forecasts are probabilistic estimates and should not be interpreted as guaranteed outcomes  
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Latest market and macro snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPI (YoY %)", f"{float(x_base.iloc[0].get('cpi_inflation_yoy', 0)):.2f}")
    c2.metric("Unemployment (%)", f"{float(x_base.iloc[0].get('unemployment_rate', 0)):.2f}")
    c3.metric("Bank Rate (%)", f"{float(x_base.iloc[0].get('bank_rate', 0)):.2f}")
    c4.metric("FTSE monthly return", f"{latest_ftse_return:.4f}")

    st.subheader("Recent market trend summary")
    if ftse_live_df is not None and len(ftse_live_df) >= 3:
        last_3 = ftse_live_df["ftse_return"].tail(3)
        avg_3 = last_3.mean()
        if avg_3 > 0.01:
            trend_msg = "The FTSE 100 has shown a positive short-term trend over the last three monthly observations."
        elif avg_3 < -0.01:
            trend_msg = "The FTSE 100 has shown a negative short-term trend over the last three monthly observations."
        else:
            trend_msg = "The FTSE 100 has been relatively mixed or stable over the last three monthly observations."
        st.write(trend_msg)
        st.write(f"Average monthly return over the last 3 observations: **{avg_3:.4%}**.")
    else:
        st.write("Recent trend summary is unavailable because live market history could not be loaded.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    with st.expander("Show baseline feature row"):
        st.dataframe(x_base[feature_cols], use_container_width=True)

    st.subheader("Recent market and macro data")
    ch1, ch2 = st.columns(2)

    with ch1:
        if ftse_live_df is not None:
            st.plotly_chart(
                make_line_chart(ftse_live_df.tail(24), "FTSE 100 Monthly Returns (last 24 months)", "ftse_return"),
                use_container_width=True
            )
        else:
            st.warning("Live FTSE chart unavailable; app is using saved baseline.")

    with ch2:
        macro_summary = pd.DataFrame({
            "Metric": ["CPI (YoY %)", "Unemployment (%)", "Bank Rate (%)"],
            "Saved baseline value": [
                float(x_base.iloc[0].get("cpi_inflation_yoy", 0)),
                float(x_base.iloc[0].get("unemployment_rate", 0)),
                float(x_base.iloc[0].get("bank_rate", 0)),
            ]
        })
        st.dataframe(macro_summary, use_container_width=True, hide_index=True)

    st.info(
        "This dashboard provides AI-driven macro-financial forecasting and scenario analysis for academic research purposes. "
        "Live market data is refreshed from Yahoo Finance, while macroeconomic inputs are taken from the latest validated saved baseline. "
        "Forecasts are not guaranteed market predictions and are not financial advice."
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2: Results & Evidence
# ============================================================
with tabs[1]:
    st.subheader("Model evidence and dissertation plots")

    perf_df = pd.DataFrame({
        "Model": ["Ridge (tuned, TSCV)", "Ridge Regression", "Gradient Boosting", "Baseline (return_t)"],
        "MAE": [0.028996, 0.029812, 0.039645, 0.041435],
        "RMSE": [0.037194, 0.037969, 0.050094, 0.052102],
        "R²": [-0.035601, -0.079228, -0.878534, -1.032138]
    })

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Model performance summary")
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        show_image("figures/actual_vs_predicted_tuned_ridge.png", "Actual vs Predicted (Tuned Ridge, test period)")
        show_image("figures/residuals_tuned_ridge.png", "Residuals over time (Tuned Ridge)")

    with colB:
        show_image("figures/scenario_impacts.png", "Scenario impacts (macro shocks)")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 3: About
# ============================================================
with tabs[2]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("About this dashboard")
    st.write(
        """
        This dashboard was developed as part of a final-year dissertation project focused on
        AI-driven macroeconomic and market forecasting for the UK market.

        **Core idea**
        - Use UK macroeconomic indicators such as CPI inflation, unemployment rate, and Bank Rate
          to help predict next-month FTSE 100 return.
        - Support decision-making through scenario analysis rather than direct trading signals.

        **Why this deployed version is stable**
        - Live market data (FTSE 100) is refreshed from Yahoo Finance.
        - Macroeconomic inputs are loaded from the latest validated saved baseline used by the trained model.
        - This avoids unreliable public macro endpoints breaking the application.

        **Model design**
        - Supervised regression problem
        - Time-series-aware validation
        - Regularised linear model (Ridge Regression) as the final deployed model
        - Scenario analysis for macro shocks

        **Data Sources**
        - FTSE 100 prices: Yahoo Finance
        - Macroeconomic variables: ONS / Bank of England (processed offline)
        - Model: Trained Ridge Regression (scikit-learn)

        **Important note**
        This dashboard is designed for academic research and demonstration purposes.
        It should not be interpreted as financial advice or a guaranteed prediction system.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
