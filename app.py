# ===== START: app.py =====
import os
from io import StringIO

import streamlit as st
import pandas as pd
import joblib
import requests
import yfinance as yf

st.set_page_config(page_title="FYP Econ-Market AI Dashboard", layout="wide")

# -----------------------------
# Paths (repo structure)
# -----------------------------
MODEL_PATH = "models/best_ridge_tuned.pkl"
COLS_PATH = "models/feature_cols.pkl"
BASE_ROW_PATH = "models/x_base_latest.csv"

# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(COLS_PATH)

st.title("AI-Driven Macroeconomic & Market Forecasting (FTSE 100)")
st.caption("Decision-support dashboard (not financial advice). Macroeconomic series can be released with lags and revised.")

tabs = st.tabs(["Forecast + Scenario Tool", "Results & Evidence"])

def predict(x_row: pd.DataFrame) -> float:
    """Predict next-month FTSE return using the trained pipeline."""
    return float(model.predict(x_row[feature_cols])[0])

# ============================================================
# LIVE DATA FETCH (Option 1B) + safe fallback
# ============================================================

@st.cache_data(ttl=60*60)  # cache for 1 hour
def fetch_ftse_monthly_returns(start="1990-01-01") -> pd.DataFrame:
    df = yf.download("^FTSE", start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("yfinance returned no FTSE data.")
    me = df["Close"].resample("M").last()
    ret = me.pct_change()
    return pd.DataFrame({"ftse_return": ret}).dropna()

@st.cache_data(ttl=60*60)
def fetch_gbpusd_monthly_return(start="1990-01-01") -> pd.DataFrame:
    """
    GBP/USD monthly % change (return). Only used if your model expects gbpusd_return features.
    """
    df = yf.download("GBPUSD=X", start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("yfinance returned no GBPUSD data.")
    me = df["Close"].resample("M").last()
    ret = me.pct_change()
    return pd.DataFrame({"gbpusd_return": ret}).dropna()

@st.cache_data(ttl=60*60)
def fetch_ons_timeseries_csv(uri_path: str, value_name: str) -> pd.DataFrame:
    """
    ONS timeseries CSV downloader.
    If CSV format is unexpected, falls back to ONS JSON API.
    """
    url = f"https://www.ons.gov.uk{uri_path}/download?format=csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text))

    # Try to find a date column
    date_col = None
    for c in raw.columns:
        if str(c).strip().lower() in ["date", "time period", "time_period", "timeperiod"]:
            date_col = c
            break

    # If we have date/value format
    value_col = None
    for c in raw.columns:
        if str(c).strip().lower() == "value":
            value_col = c
            break

    if date_col is not None and value_col is not None:
        tmp = raw[[date_col, value_col]].copy()
        tmp.columns = ["date_raw", "value"]
        tmp["date"] = pd.to_datetime(tmp["date_raw"], errors="coerce")
        tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
        tmp = tmp.dropna(subset=["date", "value"]).set_index("date").sort_index()
        return tmp.rename(columns={"value": value_name})[[value_name]]

    # Fallback: JSON API (more stable)
    return fetch_ons_timeseries_json(uri_path, value_name)

@st.cache_data(ttl=60*60)
def fetch_ons_timeseries_json(uri_path: str, value_name: str) -> pd.DataFrame:
    parts = uri_path.strip("/").split("/")
    series_id = parts[-2]
    dataset_id = parts[-1]
    url = f"https://api.ons.gov.uk/timeseries/{series_id}/dataset/{dataset_id}/data"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()

    # months is most common; fall back to other keys if needed
    candidates = j.get("months") or j.get("quarters") or j.get("years") or []
    rows = []
    for item in candidates:
        rows.append((item.get("date"), item.get("value")))

    df = pd.DataFrame(rows, columns=["date_raw", "value"])
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).set_index("date").sort_index()
    return df.rename(columns={"value": value_name})[[value_name]]

@st.cache_data(ttl=60*60)
def fetch_boe_bank_rate_iumabedr() -> pd.DataFrame:
    """
    Bank of England IUMABEDR (monthly average Bank Rate).
    If BoE changes the endpoint format in the future, we may need to adjust parsing.
    """
    url = (
        "https://www.bankofengland.co.uk/boeapps/database/Rates.asp?"
        "Travel=NIxIRx&into=GBP&Rateview=D&RateType=Plain&Rate=IUMABEDR&"
        "From=01/Jan/1990&To=Now&CSV=true"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))

    # Normalize date/value columns defensively
    date_col = df.columns[0]
    val_col = df.columns[-1]

    tmp = df[[date_col, val_col]].copy()
    tmp.columns = ["date", "bank_rate"]
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp["bank_rate"] = pd.to_numeric(tmp["bank_rate"], errors="coerce")
    tmp = tmp.dropna(subset=["date", "bank_rate"]).set_index("date").sort_index()

    # Monthly average, month-end index
    bank_me = tmp["bank_rate"].resample("M").mean().to_frame("bank_rate")
    return bank_me

def build_features_live(cpi, unemp, bank, ftse_ret, gbpusd_ret=None) -> pd.DataFrame:
    """
    Build a feature frame aligned to the columns in feature_cols.
    - Creates required lags and rolling stats.
    - Adds GBPUSD return features only if gbpusd_ret is provided and needed.
    """
    # align to month-end
    cpi_me = cpi.resample("M").last()
    unemp_me = unemp.resample("M").last()
    bank_me = bank.resample("M").mean()
    ftse_me = ftse_ret.resample("M").last()

    data = cpi_me.join(unemp_me, how="inner").join(bank_me, how="inner").join(ftse_me, how="inner")

    # Optional GBPUSD
    if gbpusd_ret is not None:
        gbp_me = gbpusd_ret.resample("M").last()
        data = data.join(gbp_me, how="inner")

    # Create lags for columns that exist
    for col in ["cpi_inflation_yoy", "unemployment_rate", "bank_rate", "ftse_return", "gbpusd_return"]:
        if col in data.columns:
            data[f"{col}_lag1"] = data[col].shift(1)
            data[f"{col}_lag3"] = data[col].shift(3)

    # Rolling features for FTSE return (if present)
    if "ftse_return" in data.columns:
        data["ftse_return_roll3_mean"] = data["ftse_return"].rolling(3).mean()
        data["ftse_return_roll3_std"] = data["ftse_return"].rolling(3).std()

    data = data.dropna()

    # Keep only the features your trained model expects
    usable = [c for c in feature_cols if c in data.columns]
    X_full = data[usable].copy()
    return X_full

# Sidebar controls
st.sidebar.header("Live Data")
refresh = st.sidebar.button("Refresh Live Data (APIs)")

if refresh:
    st.cache_data.clear()

# Try live; fallback to saved baseline
x_base = None
live_status = None

try:
    # Fetch ONS CPI and Unemployment (use your exact URIs)
    cpi = fetch_ons_timeseries_csv(
        "/economy/inflationandpriceindices/timeseries/d7g7/mm23",
        "cpi_inflation_yoy"
    )
    unemp = fetch_ons_timeseries_csv(
        "/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms",
        "unemployment_rate"
    )
    bank = fetch_boe_bank_rate_iumabedr()
    ftse = fetch_ftse_monthly_returns()

    # Fetch GBPUSD only if your model expects it
    gbp_needed = any(col.startswith("gbpusd_return") for col in feature_cols)
    gbp = fetch_gbpusd_monthly_return() if gbp_needed else None

    X_live = build_features_live(cpi, unemp, bank, ftse, gbpusd_ret=gbp)
    x_base = X_live.loc[[X_live.index.max()]].copy()
    live_status = f"Live data OK. Latest usable month: {X_live.index.max().date()}"
except Exception as e:
    # Fallback to saved baseline row
    x_base = pd.read_csv(BASE_ROW_PATH, index_col=0)
    live_status = f"Live data failed; using saved baseline file. Reason: {type(e).__name__}"

st.sidebar.success(live_status)

# ============================================================
# TAB 1: Forecast + Scenario
# ============================================================
with tabs[0]:
    st.subheader("Baseline input (latest row)")
    st.dataframe(x_base[feature_cols], use_container_width=True)

    st.subheader("Scenario shocks (adjust current macro values)")
    col1, col2, col3 = st.columns(3)
    with col1:
        delta_cpi = st.slider("Δ CPI inflation (percentage points)", -2.0, 2.0, 0.0, 0.1)
    with col2:
        delta_bank = st.slider("Δ Bank Rate (percentage points)", -2.0, 2.0, 0.0, 0.1)
    with col3:
        delta_unemp = st.slider("Δ Unemployment rate (percentage points)", -2.0, 2.0, 0.0, 0.1)

    base_pred = predict(x_base)

    x_scn = x_base.copy()

    # Apply shocks only if the column exists in your feature set (safe)
    if "cpi_inflation_yoy" in x_scn.columns:
        x_scn["cpi_inflation_yoy"] = x_scn["cpi_inflation_yoy"] + delta_cpi
    if "bank_rate" in x_scn.columns:
        x_scn["bank_rate"] = x_scn["bank_rate"] + delta_bank
    if "unemployment_rate" in x_scn.columns:
        x_scn["unemployment_rate"] = x_scn["unemployment_rate"] + delta_unemp

    scn_pred = predict(x_scn)
    change = scn_pred - base_pred

    st.markdown("### Predictions")
    a, b, c = st.columns(3)
    a.metric("Baseline predicted next-month return", f"{base_pred:.6f}")
    b.metric("Scenario predicted next-month return", f"{scn_pred:.6f}")
    c.metric("Change (Scenario - Baseline)", f"{change:.6f}")

    st.info(
        "This is an academic decision-support tool. Macroeconomic indicators can be released with lags and may be revised. "
        "Forecasts are not guaranteed market predictions and are not financial advice."
    )

# ============================================================
# TAB 2: Evidence (plots) - safe loading
# ============================================================
def show_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing file: {path}")

with tabs[1]:
    st.subheader("Evidence and key plots")
    colA, colB = st.columns(2)

    with colA:
        show_image("figures/actual_vs_predicted_tuned_ridge.png",
                   "Actual vs Predicted (Tuned Ridge, test period)")
        show_image("figures/residuals_tuned_ridge.png",
                   "Residuals over time (Tuned Ridge)")

    with colB:
        show_image("figures/scenario_impacts.png",
                   "Scenario impacts (macro shocks)")

# ===== END: app.py =====
