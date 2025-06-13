import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Step 1: Upload data
st.title("SEO Forecasting Tool - Evergreen Projects")
st.markdown("""
This tool helps forecast organic search traffic for evergreen SEO content such as Hotels, Money, and Telegraph Puzzles.
You can apply scenario-based uplifts (e.g. publishing more articles, adding newsletters).
""")

uploaded_file = st.file_uploader("Upload historical data (CSV with 'ds' and 'y' columns)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['ds'] = pd.to_datetime(df['ds'])

    st.subheader("Historical Data Preview")
    st.dataframe(df.tail())

    # Step 2: Optional uplift input
    st.subheader("Scenario Modifiers")
    uplift_pct = st.slider("Expected % Uplift (e.g. content investment, newsletter)", min_value=-50, max_value=100, value=0)

    forecast_periods = st.number_input("Months to Forecast", min_value=1, max_value=24, value=6)

    # Step 3: Fit Prophet model
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_periods * 30)
    forecast = model.predict(future)

    # Step 4: Apply uplift
    if uplift_pct != 0:
        uplift_multiplier = 1 + (uplift_pct / 100)
        forecast['yhat_uplift'] = forecast['yhat'] * uplift_multiplier
    else:
        forecast['yhat_uplift'] = forecast['yhat']

    # Step 5: Plot
    st.subheader("Forecast")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast['ds'], forecast['yhat'], label='Baseline Forecast')
    ax.plot(forecast['ds'], forecast['yhat_uplift'], label='With Scenario Uplift', linestyle='--')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("SEO Sessions")
    ax.legend()
    st.pyplot(fig)

    # Step 6: Export option
    st.subheader("Download Forecast")
    output = forecast[['ds', 'yhat', 'yhat_uplift']].copy()
    output.columns = ['Date', 'Baseline Forecast', 'With Uplift']
    csv = output.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "forecast_output.csv", "text/csv")
