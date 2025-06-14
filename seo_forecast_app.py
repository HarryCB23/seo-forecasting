import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import ExponentialSmoothing as HoltWinters
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import timedelta

# Step 1: Upload data
st.title("SEO Forecasting Tool - Evergreen Projects")
st.markdown("""
This tool helps forecast organic search traffic for evergreen SEO content such as Hotels, Money, and Telegraph Puzzles.
You can apply scenario-based uplifts (e.g. publishing more articles, adding newsletters).
""")

# Step 2: Upload CSV with 'ds' and 'y'
uploaded_file = st.file_uploader("Upload historical data (CSV with 'ds' and 'y' columns)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')

    st.subheader("Historical Data Preview")
    st.dataframe(df.tail())

    # Step 3: Select model
    model_choice = st.selectbox("Choose Forecasting Model", [
        "Prophet",
        "Exponential Smoothing",
        "Holt-Winters (Multiplicative)",
        "ARIMA",
        "Decay Model (Logarithmic)",
        "Custom Growth/Decay Combo",
        "Gradient Boosting (placeholder)",
        "Fourier Series Model (placeholder)",
        "Bayesian Structural Time Series (placeholder)"
    ])

    model_descriptions = {
        "Prophet": "Best for general-purpose forecasting with seasonality and trend. Good for evergreen content.",
        "Exponential Smoothing": "Gives more weight to recent data. Best for stable or slowly changing traffic.",
        "Holt-Winters (Multiplicative)": "Adds trend and seasonality. Best for seasonal content like travel or events.",
        "ARIMA": "Best for consistent, stationary traffic patterns. Requires longer history.",
        "Decay Model (Logarithmic)": "Models traffic that drops off over time (e.g. news articles or product releases).",
        "Custom Growth/Decay Combo": "User-defined growth and decay for campaigns or expected declines.",
        "Gradient Boosting (placeholder)": "Advanced machine learning model. Requires structured feature input.",
        "Fourier Series Model (placeholder)": "Captures complex seasonality (e.g. weekly patterns).",
        "Bayesian Structural Time Series (placeholder)": "Probabilistic model with multiple trend components and uncertainty."
    }

    st.caption(model_descriptions[model_choice])

    # Step 4: Number of forecast months
    forecast_periods = st.number_input("Months to Forecast", min_value=1, max_value=24, value=6)
    forecast_days = forecast_periods * 30

    # Step 5: Scenario Modifiers
    st.subheader("Scenario Modifiers")
    if "modifiers" not in st.session_state:
        st.session_state.modifiers = [{"label": "", "value": 0, "start_month": 1}]

    if st.button("Add Another Scenario Modifier"):
        st.session_state.modifiers.append({"label": "", "value": 0, "start_month": 1})

    for i, mod in enumerate(st.session_state.modifiers):
        col1, col2, col3 = st.columns([3, 2, 2])
        st.session_state.modifiers[i]["label"] = col1.text_input(f"Modifier {i+1} Label", mod["label"], key=f"label_{i}")
        st.session_state.modifiers[i]["value"] = col2.slider(f"% Change", -50, 100, mod["value"], key=f"value_{i}")
        st.session_state.modifiers[i]["start_month"] = col3.number_input("Start Month", 1, forecast_periods, mod["start_month"], key=f"start_{i}")

    # Step 6: Forecast based on selected model
    if model_choice == "Prophet":
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast['yhat_uplift'] = forecast['yhat']

    elif model_choice == "Exponential Smoothing":
        model = ExponentialSmoothing(df['y'], trend=None, seasonal=None)
        fitted = model.fit()
        forecast_values = fitted.forecast(forecast_days)
        forecast = pd.DataFrame({
            'ds': pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days),
            'yhat': forecast_values
        })
        forecast['yhat_uplift'] = forecast['yhat']

    elif model_choice == "Holt-Winters (Multiplicative)":
        model = HoltWinters(df['y'], trend='add', seasonal='mul', seasonal_periods=30)
        fitted = model.fit()
        forecast_values = fitted.forecast(forecast_days)
        forecast = pd.DataFrame({
            'ds': pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days),
            'yhat': forecast_values
        })
        forecast['yhat_uplift'] = forecast['yhat']

    elif model_choice == "ARIMA":
        model = ARIMA(df['y'], order=(1, 1, 1))
        fitted = model.fit()
        forecast_values = fitted.forecast(steps=forecast_days)
        forecast = pd.DataFrame({
            'ds': pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days),
            'yhat': forecast_values
        })
        forecast['yhat_uplift'] = forecast['yhat']

    elif model_choice == "Decay Model (Logarithmic)":
        last_value = df['y'].iloc[-1]
        decay_days = np.arange(1, forecast_days + 1)
        decay_values = last_value / np.log(decay_days + 1)
        forecast = pd.DataFrame({
            'ds': pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days),
            'yhat': decay_values,
            'yhat_uplift': decay_values
        })

    else:
        st.warning("This model is a placeholder and will be available in a future version.")
        st.stop()

    # Step 7: Apply Modifiers
    forecast['month'] = ((forecast['ds'] - forecast['ds'].min()) / np.timedelta64(1, 'M')).astype(int) + 1

    for mod in st.session_state.modifiers:
        uplift_factor = 1 + (mod['value'] / 100)
        forecast.loc[forecast['month'] >= mod['start_month'], 'yhat_uplift'] *= uplift_factor

    # Step 8: Plot Forecast
    st.subheader("Forecast")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast['ds'], forecast['yhat'], label='Baseline Forecast')
    ax.plot(forecast['ds'], forecast['yhat_uplift'], label='With Scenario Modifiers', linestyle='--')
    if 'yhat_lower' in forecast and 'yhat_upper' in forecast:
        ax.fill_between(forecast['ds'], forecast.get('yhat_lower', forecast['yhat']), forecast.get('yhat_upper', forecast['yhat']), alpha=0.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("SEO Sessions")
    ax.legend()
    st.pyplot(fig)

    # Step 9: Export
    st.subheader("Download Forecast")
    output = forecast[['ds', 'yhat', 'yhat_uplift']].copy()
    output.columns = ['Date', 'Baseline Forecast', 'With Uplift']
    csv = output.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "forecast_output.csv", "text/csv")
