# ===== START: app.py =====
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
# Simple styling
# -----------------------------
st.markdown("""
<style>
.main > div {
    padding-top: 1rem;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
.status-ok {
    padding: 0.6rem 0.9rem;
    border-radius: 10px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #d1fae5;
    font-size: 0.95rem;
    margin-bottom: 0.6rem;
}
.status-warn {
    padding: 0.6rem 0.9rem;
    border-radius: 10px;
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #fef3c7;
    font-size: 0.95rem;
    margin-bottom: 0.6rem;
}
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 0.4rem;
    margin-bottom: 0.2rem;
}
.small-note {
    font-size: 0.9rem;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/best_ridge_tuned.pkl"
COLS_PATH = "models/feature_cols.pkl"
BASE_ROW_PATH = "models/x_base_latest.csv"

# -----------------------------
# Load model artifacts
# -----------------------------
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
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[y_col],
            mode="lines",
            name=y_col,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_title="Date",
        yaxis_title="Value",
        showlegend=False
    )
    return fig

@st.cache_data(ttl=60 * 60)
def fetch_ftse_prices_and_returns(start="2020-01-01") -> pd.DataFrame:
    """
    Stable live data source: Yahoo Finance FTSE 100.
    We only use this for recent market display, not to rebuild macro features live.
    """
    df = yf.download("^FTSE", start=start, progress=False, auto_adjust=True)

    if df.empty:
        raise RuntimeError("yfinance returned no FTSE data.")

    # Sometimes yfinance can return multi-index columns depending on version/setup.
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", "^FTSE") in df.columns:
            close_series = df[("Close", "^FTSE")]
        else:
            close_series = df["Close"].iloc[:, 0]
    else:
        close_series = df["Close"]

    # IMPORTANT FIX: use ME instead of M
    close = close_series.resample("ME").last()
    ret = close.pct_change()

    out = pd.DataFrame({
        "ftse_close": close,
        "ftse_return": ret
    }).dropna()

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

# -----------------------------
# Stable live market fetch only
# -----------------------------
ftse_live_df = None
live_status = None
latest_market_month = None

try:
    ftse_live_df = fetch_ftse_prices_and_returns()
    latest_ftse_return = float(ftse_live_df["ftse_return"].iloc[-1])
    latest_market_month = ftse_live_df.index.max()
    live_status = f"Live FTSE data OK. Latest market month: {latest_market_month.date()}"
except Exception as e:
    latest_ftse_return = float(x_base.iloc[0].get("ftse_return", 0))
    latest_market_month = None
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
st.title("FTSE 100 Macro Intelligence Dashboard")
st.caption("AI-driven market forecasting and scenario analysis for academic decision support.")

tabs = st.tabs(["Dashboard", "Results & Evidence", "About"])

# ============================================================
# TAB 1: Dashboard
# ============================================================
with tabs[0]:
    st.markdown('<div class="section-title">Model Forecast Overview</div>', unsafe_allow_html=True)

    # -----------------------------
    # Scenario controls
    # -----------------------------
    st.subheader("Scenario shocks")
    s1, s2, s3 = st.columns(3)
    with s1:
        delta_cpi = st.slider("Δ CPI inflation (percentage points)", -2.0, 2.0, 0.0, 0.1)
    with s2:
        delta_bank = st.slider("Δ Bank Rate (percentage points)", -2.0, 2.0, 0.0, 0.1)
    with s3:
        delta_unemp = st.slider("Δ Unemployment rate (percentage points)", -2.0, 2.0, 0.0, 0.1)

    # -----------------------------
    # Predictions
    # -----------------------------
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

    # -----------------------------
    # Key metrics
    # -----------------------------
    st.subheader("Key metrics")
    def format_return(val):
    return f"{val:.4%}"

def delta_color(val):
    if val > 0:
        return "normal"
    elif val < 0:
        return "inverse"
    return "off"

k1.metric(
    "Baseline predicted return",
    format_return(base_pred)
)

k2.metric(
    "Scenario predicted return",
    format_return(scn_pred),
    delta=format_return(scn_pred - base_pred),
    delta_color=delta_color(scn_pred - base_pred)
)

k3.metric(
    "Prediction change",
    format_return(change),
    delta=format_return(change),
    delta_color=delta_color(change)
)
    k4.metric(
        "Latest market month",
        str(latest_market_month.date()) if latest_market_month is not None else "Saved baseline"
    )

    # -----------------------------
    # Interpretation helpers
    # -----------------------------
    def interpret_return(val: float) -> str:
        if val >= 0.02:
            return "strongly positive"
        elif val >= 0.005:
            return "moderately positive"
        elif val > -0.005:
            return "broadly neutral"
        elif val > -0.02:
            return "moderately negative"
        else:
            return "strongly negative"

    def change_message(delta: float) -> str:
        if delta > 0.005:
            return "The scenario improves the forecast noticeably relative to the baseline."
        elif delta > 0.0:
            return "The scenario slightly improves the forecast relative to the baseline."
        elif delta == 0.0:
            return "The scenario does not materially change the forecast."
        elif delta > -0.005:
            return "The scenario slightly weakens the forecast relative to the baseline."
        else:
            return "The scenario weakens the forecast noticeably relative to the baseline."

    baseline_text = interpret_return(base_pred)
    scenario_text = interpret_return(scn_pred)

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

    # -----------------------------
    # Economic interpretation
    # -----------------------------
    st.subheader("Scenario insight")
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

    # -----------------------------
    # Latest snapshot
    # -----------------------------
    st.subheader("Latest market and macro snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPI (YoY %)", f"{float(x_base.iloc[0].get('cpi_inflation_yoy', 0)):.2f}")
    c2.metric("Unemployment (%)", f"{float(x_base.iloc[0].get('unemployment_rate', 0)):.2f}")
    c3.metric("Bank Rate (%)", f"{float(x_base.iloc[0].get('bank_rate', 0)):.2f}")
    c4.metric("FTSE monthly return", f"{latest_ftse_return:.4f}")

    # -----------------------------
    # FTSE recent trend summary
    # -----------------------------
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
        st.write(
            f"Average monthly return over the last 3 observations: **{avg_3:.4%}**."
        )
    else:
        st.write("Recent trend summary is unavailable because live market history could not be loaded.")

    with st.expander("Show baseline feature row"):
        st.dataframe(x_base[feature_cols], use_container_width=True)

    # -----------------------------
    # Recent market and macro data
    # -----------------------------
    st.subheader("Recent market and macro data")
    ch1, ch2 = st.columns(2)

    with ch1:
        if ftse_live_df is not None:
            st.plotly_chart(
                make_line_chart(
                    ftse_live_df.tail(24),
                    "FTSE 100 Monthly Returns (last 24 months)",
                    "ftse_return"
                ),
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
        "This is a stable academic decision-support dashboard. "
        "Live market data is refreshed from Yahoo Finance, while macroeconomic inputs are taken from the latest validated saved baseline. "
        "Forecasts are not guaranteed market predictions and are not financial advice."
    )
# ============================================================
# TAB 2: Results & Evidence
# ============================================================
with tabs[1]:
    st.subheader("Model evidence and dissertation plots")
    colA, colB = st.columns(2)

    with colA:
        show_image(
            "figures/actual_vs_predicted_tuned_ridge.png",
            "Actual vs Predicted (Tuned Ridge, test period)"
        )
        show_image(
            "figures/residuals_tuned_ridge.png",
            "Residuals over time (Tuned Ridge)"
        )

    with colB:
        show_image(
            "figures/scenario_impacts.png",
            "Scenario impacts (macro shocks)"
        )

# ============================================================
# TAB 3: About
# ============================================================
with tabs[2]:
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

        **Important note**
        This dashboard is designed for academic research and demonstration purposes.
        It should not be interpreted as financial advice or a guaranteed prediction system.
        """
    )

# ===== END: app.py =====
