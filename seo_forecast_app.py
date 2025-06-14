import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

# Step 1: Upload data
st.title("SEO Forecasting Tool - Evergreen Projects")
st.markdown("""
This tool helps forecast organic search traffic for evergreen SEO content such as Hotels, Money, and Telegraph Puzzles.
You can apply scenario-based uplifts or declines (e.g. publishing more articles, Google AIO, newsletters) starting at different times.
""")

uploaded_file = st.file_uploader("Upload historical data (CSV with 'ds' and 'y' columns)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['ds'] = pd.to_datetime(df['ds'])

    st.subheader("Historical Data Preview")
    st.dataframe(df.tail())

    # Step 2: Scenario Modifiers
    st.subheader("Scenario Modifiers (Add and Set Start Month)")

    if "modifiers" not in st.session_state:
        st.session_state.modifiers = [{"label": "", "value": 0, "start_month": 0}]

    if st.button("➕ Add new modifier"):
        st.session_state.modifiers.append({"label": "", "value": 0, "start_month": 0})

    if st.button("♻️ Reset modifiers"):
        st.session_state.modifiers = [{"label": "", "value": 0, "start_month": 0}]

    # Input for each modifier
    updated_modifiers = []
    for i, mod in enumerate(st.session_state.modifiers):
        cols = st.columns([2, 1, 1])
        label = cols[0].text_input(f"Modifier #{i+1} Label", value=mod["label"], key=f"label_{i}")
        value = cols[1].slider("Change (%)", min_value=-50, max_value=100, value=mod["value"], key=f"value_{i}")
        start_month = cols[2].number_input("Start month", min_value=0, max_value=24, value=mod.get("start_month", 0), key=f"start_{i}")
        updated_modifiers.append({"label": label, "value": value, "start_month": start_month})

    st.session_state.modifiers = updated_modifiers

    # Step 3: Forecasting period
    forecast_periods = st.number_input("Months to Forecast", min_value=1, max_value=24, value=6)

    # Step 4: Fit Prophet model
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_periods * 30)
    forecast = model.predict(future)

    # Step 5: Apply modifiers
    forecast['yhat_adjusted'] = forecast['yhat']
    forecast_start_date = forecast['ds'].min()

    for mod in st.session_state.modifiers:
        if mod['value'] == 0:
            continue
        mod_multiplier = 1 + (mod['value'] / 100)
        mod_start_date = forecast_start_date + relativedelta(months=mod['start_month'])
        forecast.loc[forecast['ds'] >= mod_start_date, 'yhat_adjusted'] *= mod_multiplier

    # Step 6: Plot
    st.subheader("Forecast")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast['ds'], forecast['yhat'], label='Baseline Forecast')
    ax.plot(forecast['ds'], forecast['yhat_adjusted'], label='Scenario Forecast', linestyle='--')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("SEO Sessions")
    ax.legend()
    st.pyplot(fig)

    # Step 7: Export option
    st.subheader("Download Forecast")
    output = forecast[['ds', 'yhat', 'yhat_adjusted']].copy()
    output.columns = ['Date', 'Baseline Forecast', 'Scenario Forecast']
    csv = output.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "forecast_output.csv", "text/csv")
