# ===== START: app.py =====
import os
from io import StringIO

import streamlit as st
import pandas as pd
import joblib
import requests
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

# -----------------------------
# Utility
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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# ============================================================
# LIVE DATA FETCH
# ============================================================

@st.cache_data(ttl=60 * 60)
def fetch_ftse_prices_and_returns(start="1990-01-01") -> pd.DataFrame:
    df = yf.download("^FTSE", start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("yfinance returned no FTSE data.")
    close = df["Close"].resample("M").last()
    ret = close.pct_change()
    out = pd.DataFrame({
        "ftse_close": close,
        "ftse_return": ret
    }).dropna()
    return out

@st.cache_data(ttl=60 * 60)
def fetch_gbpusd_monthly_return(start="1990-01-01") -> pd.DataFrame:
    df = yf.download("GBPUSD=X", start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("yfinance returned no GBP/USD data.")
    close = df["Close"].resample("M").last()
    ret = close.pct_change()
    return pd.DataFrame({"gbpusd_return": ret}).dropna()

@st.cache_data(ttl=60 * 60)
def fetch_ons_timeseries_csv(uri_path: str, value_name: str) -> pd.DataFrame:
    """
    Try ONS generator CSV first, then fall back to ONS JSON API.
    """
    encoded_uri = requests.utils.quote(uri_path, safe="")
    url = f"https://www.ons.gov.uk/generator?format=csv&uri={encoded_uri}"

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        raw = pd.read_csv(StringIO(r.text))

        if raw.shape[1] >= 2:
            tmp = raw.iloc[:, :2].copy()
            tmp.columns = ["date_raw", "value"]
            tmp["date"] = pd.to_datetime(tmp["date_raw"], errors="coerce")
            tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
            tmp = tmp.dropna(subset=["date", "value"]).set_index("date").sort_index()
            if not tmp.empty:
                return tmp.rename(columns={"value": value_name})[[value_name]]
    except Exception:
        pass

    return fetch_ons_timeseries_json(uri_path, value_name)

@st.cache_data(ttl=60 * 60)
def fetch_ons_timeseries_json(uri_path: str, value_name: str) -> pd.DataFrame:
    parts = uri_path.strip("/").split("/")
    series_id = parts[-2]
    dataset_id = parts[-1]
    url = f"https://api.ons.gov.uk/timeseries/{series_id}/dataset/{dataset_id}/data"

    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()

    candidates = j.get("months") or j.get("quarters") or j.get("years") or []
    rows = [(item.get("date"), item.get("value")) for item in candidates]

    df = pd.DataFrame(rows, columns=["date_raw", "value"])
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).set_index("date").sort_index()
    if df.empty:
        raise RuntimeError(f"ONS JSON returned no usable data for {uri_path}")
    return df.rename(columns={"value": value_name})[[value_name]]

@st.cache_data(ttl=60 * 60)
def fetch_boe_bank_rate_iumabedr() -> pd.DataFrame:
    """
    Scrape the public Bank Rate history page instead of the blocked CSV endpoint.
    The page contains 'Date Changed' and 'Rate' rows publicly.
    """
    url = "https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))
    rate_table = None

    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        joined = " | ".join(cols)
        if "date changed" in joined and "rate" in joined:
            rate_table = t.copy()
            break

    if rate_table is None:
        raise RuntimeError("Could not find Bank Rate table on BoE page.")

    rate_table.columns = [str(c).strip().lower() for c in rate_table.columns]
    date_col = rate_table.columns[0]
    rate_col = rate_table.columns[-1]

    tmp = rate_table[[date_col, rate_col]].copy()
    tmp.columns = ["date", "bank_rate"]

    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce", dayfirst=True)
    tmp["bank_rate"] = pd.to_numeric(tmp["bank_rate"], errors="coerce")
    tmp = tmp.dropna(subset=["date", "bank_rate"]).set_index("date").sort_index()

    if tmp.empty:
        raise RuntimeError("BoE Bank Rate table parsed but no usable rows were found.")

    daily_index = pd.date_range(tmp.index.min(), pd.Timestamp.today(), freq="D")
    tmp = tmp.reindex(daily_index).ffill()
    tmp.index.name = "date"

    bank_me = tmp["bank_rate"].resample("M").last().to_frame("bank_rate")
    return bank_me

def build_features_live(
    cpi: pd.DataFrame,
    unemp: pd.DataFrame,
    bank: pd.DataFrame,
    ftse_df: pd.DataFrame,
    gbpusd_ret: pd.DataFrame | None = None
):
    cpi_me = cpi.resample("M").last()
    unemp_me = unemp.resample("M").last()
    bank_me = bank.resample("M").last()
    ftse_me = ftse_df.resample("M").last()

    data = (
        cpi_me.join(unemp_me, how="inner")
        .join(bank_me, how="inner")
        .join(ftse_me[["ftse_return"]], how="inner")
    )

    if gbpusd_ret is not None:
        gbp_me = gbpusd_ret.resample("M").last()
        data = data.join(gbp_me, how="inner")

    for col in ["cpi_inflation_yoy", "unemployment_rate", "bank_rate", "ftse_return", "gbpusd_return"]:
        if col in data.columns:
            data[f"{col}_lag1"] = data[col].shift(1)
            data[f"{col}_lag3"] = data[col].shift(3)

    if "ftse_return" in data.columns:
        data["ftse_return_roll3_mean"] = data["ftse_return"].rolling(3).mean()
        data["ftse_return_roll3_std"] = data["ftse_return"].rolling(3).std()

    data = data.dropna()

    usable = [c for c in feature_cols if c in data.columns]
    X_full = data[usable].copy()
    return X_full, data

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Controls")
st.sidebar.caption("Live refresh + scenario analysis")

refresh = st.sidebar.button("Refresh Live Data (APIs)")
if refresh:
    st.cache_data.clear()

# -----------------------------
# Live data with fallback
# -----------------------------
x_base = None
live_status = None
latest_month = None
latest_data = {}
ftse_live_df = None
macro_chart_df = None

try:
    cpi = fetch_ons_timeseries_csv(
        "/economy/inflationandpriceindices/timeseries/d7g7/mm23",
        "cpi_inflation_yoy"
    )
    unemp = fetch_ons_timeseries_csv(
        "/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms",
        "unemployment_rate"
    )
    bank = fetch_boe_bank_rate_iumabedr()
    ftse_live_df = fetch_ftse_prices_and_returns()

    gbp_needed = any(col.startswith("gbpusd_return") for col in feature_cols)
    gbp = fetch_gbpusd_monthly_return() if gbp_needed else None

    X_live, merged_live = build_features_live(cpi, unemp, bank, ftse_live_df, gbpusd_ret=gbp)
    x_base = X_live.loc[[X_live.index.max()]].copy()

    latest_month = X_live.index.max()
    latest_data["cpi_inflation_yoy"] = float(cpi.resample("M").last().iloc[-1, 0])
    latest_data["unemployment_rate"] = float(unemp.resample("M").last().iloc[-1, 0])
    latest_data["bank_rate"] = float(bank.resample("M").last().iloc[-1, 0])
    latest_data["ftse_return"] = float(ftse_live_df["ftse_return"].iloc[-1])

    if gbp_needed and gbp is not None:
        latest_data["gbpusd_return"] = float(gbp["gbpusd_return"].iloc[-1])

    macro_chart_df = pd.concat(
        [
            cpi.resample("M").last().tail(24),
            unemp.resample("M").last().tail(24),
            bank.resample("M").last().tail(24)
        ],
        axis=1
    )

    live_status = f"Live data OK. Latest usable month: {latest_month.date()}"

except Exception as e:
    x_base = pd.read_csv(BASE_ROW_PATH, index_col=0)
    live_status = f"Live data failed; using saved baseline file. Reason: {type(e).__name__}: {e}"

if "failed" in live_status.lower():
    st.sidebar.markdown(f'<div class="status-warn">{live_status}</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<div class="status-ok">{live_status}</div>', unsafe_allow_html=True)

st.sidebar.markdown(
    '<div class="small-note">This dashboard is for academic forecasting and scenario analysis, not financial advice.</div>',
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

    st.subheader("Scenario shocks")
    s1, s2, s3 = st.columns(3)
    with s1:
        delta_cpi = st.slider("Δ CPI inflation (percentage points)", -2.0, 2.0, 0.0, 0.1)
    with s2:
        delta_bank = st.slider("Δ Bank Rate (percentage points)", -2.0, 2.0, 0.0, 0.1)
    with s3:
        delta_unemp = st.slider("Δ Unemployment rate (percentage points)", -2.0, 2.0, 0.0, 0.1)

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

    st.subheader("Key metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline predicted return", f"{base_pred:.6f}")
    k2.metric("Scenario predicted return", f"{scn_pred:.6f}")
    k3.metric("Prediction change", f"{change:.6f}")
    k4.metric("Latest usable month", str(latest_month.date()) if latest_month is not None else "Saved baseline")

    st.subheader("Latest market and macro snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPI (YoY %)", f"{latest_data.get('cpi_inflation_yoy', x_base.iloc[0].get('cpi_inflation_yoy', 0)):.2f}")
    c2.metric("Unemployment (%)", f"{latest_data.get('unemployment_rate', x_base.iloc[0].get('unemployment_rate', 0)):.2f}")
    c3.metric("Bank Rate (%)", f"{latest_data.get('bank_rate', x_base.iloc[0].get('bank_rate', 0)):.2f}")
    c4.metric("FTSE monthly return", f"{latest_data.get('ftse_return', x_base.iloc[0].get('ftse_return', 0)):.4f}")

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
        if macro_chart_df is not None:
            fig = go.Figure()
            for col in macro_chart_df.columns:
                fig.add_trace(go.Scatter(x=macro_chart_df.index, y=macro_chart_df[col], mode="lines", name=col))
            fig.update_layout(
                title="Macro Indicators (last 24 months)",
                template="plotly_dark",
                height=340,
                margin=dict(l=20, r=20, t=45, b=20),
                xaxis_title="Date",
                yaxis_title="Value"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Live macro charts unavailable; app is using saved baseline.")

    st.info(
        "This is an academic decision-support tool. Macroeconomic indicators may be released with lags and revised. "
        "Forecasts are not guaranteed market predictions and are not financial advice."
    )

# ============================================================
# TAB 2: Results & Evidence
# ============================================================
with tabs[1]:
    st.subheader("Model evidence and dissertation plots")
    colA, colB = st.columns(2)

    with colA:
        show_image("figures/actual_vs_predicted_tuned_ridge.png",
                   "Actual vs Predicted (Tuned Ridge, test period)")
        show_image("figures/residuals_tuned_ridge.png",
                   "Residuals over time (Tuned Ridge)")

    with colB:
        show_image("figures/scenario_impacts.png",
                   "Scenario impacts (macro shocks)")

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
