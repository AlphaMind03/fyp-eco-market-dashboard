# ===== START: app.py =====
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="FYP Econ-Market AI Dashboard", layout="wide")

MODEL_PATH = "models/best_ridge_tuned.pkl"
COLS_PATH = "models/feature_cols.pkl"
BASE_ROW_PATH = "models/x_base_latest.csv"

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(COLS_PATH)
x_base = pd.read_csv(BASE_ROW_PATH, index_col=0)

st.title("AI-Driven Macroeconomic & Market Forecasting (FTSE 100)")
st.caption("Decision-support dashboard (not financial advice).")

tabs = st.tabs(["Forecast + Scenario Tool", "Results & Evidence"])

def predict(x_row: pd.DataFrame) -> float:
    return float(model.predict(x_row[feature_cols])[0])

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
    x_scn["cpi_inflation_yoy"] = x_scn["cpi_inflation_yoy"] + delta_cpi
    x_scn["bank_rate"] = x_scn["bank_rate"] + delta_bank
    x_scn["unemployment_rate"] = x_scn["unemployment_rate"] + delta_unemp

    scn_pred = predict(x_scn)
    change = scn_pred - base_pred

    st.markdown("### Predictions")
    a, b, c = st.columns(3)
    a.metric("Baseline predicted next-month return", f"{base_pred:.6f}")
    b.metric("Scenario predicted next-month return", f"{scn_pred:.6f}")
    c.metric("Change (Scenario - Baseline)", f"{change:.6f}")

    st.info("These outputs are model-based sensitivities for academic decision support, not guaranteed market predictions.")

import os

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