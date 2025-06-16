import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import ExponentialSmoothing as HoltWinters
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import timedelta

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="SEO Forecasting Tool",
    page_icon="📈", # You can use emojis or a path to an image file
    layout="wide", # Use the full width of the browser
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

st.title("📈 SEO Forecasting Tool - Evergreen Projects")
st.markdown("""
This tool helps forecast organic search traffic for evergreen SEO content such as Hotels, Money, and Telegraph Puzzles.
You can apply scenario-based uplifts (e.g. publishing more articles, adding newsletters).
""")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")

# Step 1: Upload data moved to sidebar
st.sidebar.subheader("1. Upload Historical Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'ds' and 'y' columns", type=["csv"])

df = None # Initialize df outside the if block
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        st.sidebar.success("Data uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}. Please ensure it's a CSV with 'ds' (date) and 'y' (value) columns.")

# Only proceed with the rest of the app if data is uploaded
if df is not None:
    st.sidebar.divider()

    st.sidebar.subheader("2. Model Selection")
    # Step 3: Select model moved to sidebar
    model_choice = st.sidebar.selectbox("Choose Forecasting Model", [
        "Prophet",
        "Exponential Smoothing",
        "Holt-Winters (Multiplicative)",
        "ARIMA",
        "Decay Model (Logarithmic)",
        "Custom Growth/Decay Combo",
        "Gradient Boosting (placeholder)",
        "Fourier Series Model (placeholder)",
        "Bayesian Structural Time Series (placeholder)"
    ], help="Select the statistical model best suited for your data's characteristics.")

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
    st.sidebar.caption(model_descriptions[model_choice])

    st.sidebar.divider()

    st.sidebar.subheader("3. Forecast Horizon")
    # Step 4: Number of forecast months moved to sidebar
    forecast_periods = st.sidebar.number_input("Months to Forecast", min_value=1, max_value=24, value=6, help="Enter the number of months you wish to forecast into the future (max 24).")
    forecast_days = forecast_periods * 30 # Approximation for simplicity

    st.sidebar.divider()

    # Step 5: Scenario Modifiers moved to sidebar and wrapped in a container
    st.sidebar.subheader("4. Scenario Modifiers")
    st.sidebar.info("Add percentage changes to your forecast based on planned initiatives.")

    # Using st.container for grouping the modifiers section in the sidebar
    with st.sidebar.container():
        if "modifiers" not in st.session_state:
            st.session_state.modifiers = [{"label": "", "value": 0, "start_month": 1}]

        if st.button("➕ Add Another Scenario Modifier", type="secondary", use_container_width=True):
            st.session_state.modifiers.append({"label": "", "value": 0, "start_month": 1})

        for i, mod in enumerate(st.session_state.modifiers):
            # Using columns within the sidebar to align modifier inputs
            col1, col2, col3, col4 = st.columns([3, 2, 2, 0.5])
            with col1:
                st.session_state.modifiers[i]["label"] = st.text_input(f"Modifier {i+1} Label", mod["label"], key=f"label_{i}", placeholder="e.g., New Content Series")
            with col2:
                st.session_state.modifiers[i]["value"] = st.slider(f"% Change", -50, 100, mod["value"], key=f"value_{i}")
            with col3:
                st.session_state.modifiers[i]["start_month"] = st.number_input("Start Month", 1, forecast_periods, mod["start_month"], key=f"start_{i}")
            with col4:
                st.write("") # Spacer for alignment
                if st.button("🗑️", key=f"delete_{i}", help="Remove this modifier"):
                    st.session_state.modifiers.pop(i)
                    st.rerun() # Rerun to update the list instantly

    st.divider() # Visual separator in the main content

    # --- Main Content Area ---
    st.subheader("📊 Historical Data Preview")
    st.dataframe(df.tail(10), use_container_width=True) # Show more rows and use full width

    st.divider()

    st.subheader("5. Generate & View Forecast")
    if st.button("🚀 Run Forecast", type="primary", use_container_width=True): # Primary button for main action
        if model_choice in ["Gradient Boosting (placeholder)", "Fourier Series Model (placeholder)", "Bayesian Structural Time Series (placeholder)", "Custom Growth/Decay Combo"]:
            st.warning("This model is a placeholder and will be available in a future version. Please select another model.")
        else:
            with st.spinner("Generating forecast..."):
                # Step 6: Forecast based on selected model
                if model_choice == "Prophet":
                    # Using st.expander for Prophet-specific settings
                    with st.expander("Prophet Model Settings", expanded=False):
                        prophet_seasonality = st.selectbox("Select seasonality modes to include", ["None", "Daily", "Weekly", "Monthly", "All"], help="Choose the seasonality components Prophet should account for.")

                    model = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=False,
                        yearly_seasonality=False # Prophet includes yearly by default if enough data
                    )

                    if prophet_seasonality == "Daily":
                        model.add_seasonality(name='daily', period=1, fourier_order=5)
                    elif prophet_seasonality == "Weekly":
                        model.add_seasonality(name='weekly', period=7, fourier_order=3)
                    elif prophet_seasonality == "Monthly":
                        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                    elif prophet_seasonality == "All":
                        model.add_seasonality(name='daily', period=1, fourier_order=5)
                        model.add_seasonality(name='weekly', period=7, fourier_order=3)
                        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

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
                    model = HoltWinters(df['y'], trend='add', seasonal='mul', seasonal_periods=30) # Assuming monthly seasonality
                    fitted = model.fit()
                    forecast_values = fitted.forecast(forecast_days)
                    forecast = pd.DataFrame({
                        'ds': pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days),
                        'yhat': forecast_values
                    })
                    forecast['yhat_uplift'] = forecast['yhat']

                elif model_choice == "ARIMA":
                    try:
                        model = ARIMA(df['y'], order=(1, 1, 1))
                        fitted = model.fit()
                        forecast_values = fitted.forecast(steps=forecast_days)
                        forecast = pd.DataFrame({
                            'ds': pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days),
                            'yhat': forecast_values
                        })
                        forecast['yhat_uplift'] = forecast['yhat']
                    except Exception as e:
                        st.error(f"ARIMA model failed to fit. This might happen with short or non-stationary data. Error: {e}")
                        st.stop()

                elif model_choice == "Decay Model (Logarithmic)":
                    last_value = df['y'].iloc[-1]
                    decay_days = np.arange(1, forecast_days + 1)
                    # Adjusted decay for more realistic values
                    decay_values = last_value * np.exp(-0.01 * decay_days) # Simple exponential decay for example
                    forecast = pd.DataFrame({
                        'ds': pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days),
                        'yhat': decay_values,
                        'yhat_uplift': decay_values
                    })


                # Step 7: Apply Modifiers
                forecast['month'] = ((forecast['ds'].dt.to_period("M") - forecast['ds'].min().to_period("M")).apply(lambda x: x.n)) + 1

                for mod in st.session_state.modifiers:
                    if mod['label'] and mod['value'] != 0: # Only apply if a label is given and value is not zero
                        uplift_factor = 1 + (mod['value'] / 100)
                        forecast.loc[forecast['month'] >= mod['start_month'], 'yhat_uplift'] *= uplift_factor

                st.success("Forecast generated successfully!")
                st.divider()

                # Step 8: Plot Forecast using Matplotlib (as per original code, can be improved later)
                st.subheader("Forecast Plot")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(forecast['ds'], forecast['yhat'], label='Baseline Forecast')
                ax.plot(forecast['ds'], forecast['yhat_uplift'], label='With Scenario Modifiers', linestyle='--')
                if 'yhat_lower' in forecast and 'yhat_upper' in forecast:
                    ax.fill_between(forecast['ds'], forecast.get('yhat_lower', forecast['yhat']), forecast.get('yhat_upper', forecast['yhat']), alpha=0.2)
                ax.set_xlabel("Date")
                ax.set_ylabel("SEO Sessions")
                ax.legend()
                st.pyplot(fig)

                # Step 8.5: Show Monthly Forecast Summary
                st.divider()
                st.subheader("Monthly Forecast Summary")
                forecast_monthly = forecast.set_index('ds').resample('M').sum(numeric_only=True)
                forecast_monthly.reset_index(inplace=True)
                forecast_monthly = forecast_monthly[['ds', 'yhat', 'yhat_uplift']]
                forecast_monthly.columns = ['Month', 'Baseline Forecast', 'With Uplift']
                st.dataframe(forecast_monthly, use_container_width=True)

                # Step 9: Export Option Selection
                st.divider()
                st.subheader("Download Forecast")
                export_choice = st.radio("Select Forecast Output Format", ["Weekly", "Monthly"], horizontal=True)

                if export_choice == "Weekly":
                    forecast_weekly = forecast.set_index('ds').resample('W').sum(numeric_only=True)
                    forecast_weekly.reset_index(inplace=True)
                    output = forecast_weekly[['ds', 'yhat', 'yhat_uplift']].copy()
                    output.columns = ['Date', 'Baseline Forecast', 'With Uplift']
                else:
                    output = forecast_monthly.copy()
                    output.columns = ['Date', 'Baseline Forecast', 'With Uplift'] # Ensure consistent column names

                csv = output.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Forecast CSV",
                    data=csv,
                    file_name=f"seo_forecast_{export_choice.lower()}_output.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
else:
    st.info("Please upload your historical data CSV file in the sidebar to begin forecasting.")
    # Optional: Add an image or instructions here to guide the user
    # st.image("https://example.com/your-upload-image.png", width=300)

