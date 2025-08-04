import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import ExponentialSmoothing as HoltWinters
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import timedelta
import calendar

st.set_page_config(
    page_title="SEO Forecasting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 SEO & Conversions Forecasting Tool")
st.markdown("""
Predict your organic search traffic and estimate potential conversions with this easy-to-use tool.

**Key Features:**
* **Traffic Forecasts:** Project future SEO sessions.
* **Scenario Planning:** Model impact of initiatives (uplifts/decays).
* **Conversions Estimates:** Calculate conversions based on your sessions and conversion rate.
* **Visual Insights:** See traffic & conversions trends on interactive graphs.
* **Exportable Data:** Download detailed forecasts for analysis.

Gain actionable insights to strategize and understand the financial impact of your SEO efforts.
""")

with st.expander("❓ How This App Works", expanded=False):
    st.markdown("""
    This application allows you to forecast future SEO traffic and estimate potential conversions based on your historical data.

    **Step-by-Step Guide:**

    1.  **Upload Data:** Upload a CSV file containing your historical `ds` (date in DD/MM/YYYY format) and `y` (SEO sessions) data using the sidebar.
    2.  **Select Model:** Choose a suitable forecasting model from the sidebar. Each model has a brief explanation to guide your selection.
    3.  **Set Horizon:** Specify the number of months you wish to forecast into the future.
    4.  **Add Scenarios:** Use scenario modifiers to model anticipated changes (e.g., content launches, algorithm updates).
    5.  **Set Conversion Rate:** Enter your average Conversion Rate (%) in the sidebar to enable conversions forecasting.
    6.  **Set Content Decay:** Choose an annual decay rate if you want to model a decline in traffic without active intervention.
    7.  **Run Forecast:** Click "🚀 Run Forecast" in the main section. The app will then display interactive traffic and conversions forecast plots and summary tables.
    8.  **Download Data:** Select your desired output format and download the comprehensive forecast CSV.
    """)
st.divider()

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")

if 'df_historical' not in st.session_state:
    st.session_state.df_historical = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'df_historical_conversions' not in st.session_state:
    st.session_state.df_historical_conversions = None
if 'current_conversion_rate' not in st.session_state:
    st.session_state.current_conversion_rate = 1.0

def process_historical_data(input_df, conversion_rate_value):
    df_temp = input_df.copy()
    if 'ds' not in df_temp.columns or 'y' not in df_temp.columns:
        st.error("Uploaded CSV must contain 'ds' (date) and 'y' (value) columns.")
        st.session_state.df_historical = None
        st.session_state.df_historical_conversions = None
        st.session_state.forecast_data = None
        return

    try:
        df_temp['ds'] = pd.to_datetime(df_temp['ds'], dayfirst=True, errors='coerce')
        df_temp.dropna(subset=['ds'], inplace=True)
    except Exception as e:
        st.error(f"Error parsing date column 'ds': {e}.")
        st.session_state.df_historical = None
        st.session_state.df_historical_conversions = None
        st.session_state.forecast_data = None
        return

    df_temp['y'] = pd.to_numeric(df_temp['y'], errors='coerce')
    df_temp.dropna(subset=['y'], inplace=True)
    if df_temp.empty:
        st.error("No valid data remains after processing. Please check your CSV file.")
        st.session_state.df_historical = None
        st.session_state.df_historical_conversions = None
        st.session_state.forecast_data = None
        return

    df_temp.drop_duplicates(subset=['ds'], inplace=True)
    processed_df = df_temp.sort_values('ds').reset_index(drop=True)
    st.session_state.df_historical = processed_df

    df_historical_with_conversions = processed_df.copy()
    df_historical_with_conversions['y_conversions'] = df_historical_with_conversions['y'] * (conversion_rate_value / 100)
    st.session_state.df_historical_conversions = df_historical_with_conversions
    st.sidebar.success("Data loaded successfully!")
    st.session_state.forecast_data = None

st.sidebar.subheader("1. Upload Historical Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'ds' and 'y' columns", type=["csv"])

if uploaded_file:
    try:
        temp_df = pd.read_csv(uploaded_file)
        process_historical_data(temp_df, st.session_state.current_conversion_rate)
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}.")

def load_sample_data_and_process_wrapper():
    sample_file_path = './data/Search PVs Example.csv'
    try:
        temp_df = pd.read_csv(sample_file_path)
        process_historical_data(temp_df, st.session_state.current_conversion_rate)
    except FileNotFoundError:
        st.error(f"Sample data file not found at '{sample_file_path}'.")
    except Exception as e:
        st.error(f"Error loading sample data file: {e}.")

if st.session_state.df_historical is None:
    st.sidebar.button("Load Sample Data", on_click=load_sample_data_and_process_wrapper, use_container_width=True)
    st.sidebar.markdown("---")

df = st.session_state.df_historical

if df is not None:
    st.subheader("📊 Historical Data Preview")
    st.dataframe(df.tail(10), use_container_width=True)

    st.sidebar.divider()
    st.sidebar.subheader("2. Model Selection")
    model_choice = st.sidebar.selectbox("Choose Forecasting Model", [
        "Prophet", "Exponential Smoothing", "ARIMA", "Decay Model (Logarithmic)"
    ], help="Select the core statistical model best suited for your data's characteristics.")

    with st.sidebar.expander("Advanced Model Options", expanded=False):
        advanced_model_choice = st.selectbox("More Models", [
            "None (Use Basic Model)", "Holt-Winters (Multiplicative)", "Custom Growth/Decay Combo",
            "Gradient Boosting (placeholder)", "Fourier Series Model (placeholder)", "Bayesian Structural Time Series (placeholder)"
        ])
        if advanced_model_choice != "None (Use Basic Model)":
            model_choice = advanced_model_choice

    model_descriptions = {
        "Prophet": "Great for general web traffic...",
        "Exponential Smoothing": "Gives more importance to your most recent data...",
        "Holt-Winters (Multiplicative)": "Identifies trends and repeating seasonal patterns...",
        "ARIMA": "A classic statistical model that's good at forecasting consistent traffic patterns...",
        "Decay Model (Logarithmic)": "Designed for traffic that starts high then drops off...",
        "Custom Growth/Decay Combo": "Allows you to define your own growth or decay rates...",
        "Gradient Boosting (placeholder)": "Advanced machine learning model...",
        "Fourier Series Model (placeholder)": "Captures complex repeating patterns...",
        "Bayesian Structural Time Series (placeholder)": "Breaks down your traffic into components..."
    }
    st.sidebar.caption(model_descriptions.get(model_choice, ""))

    st.sidebar.divider()
    st.sidebar.subheader("3. Forecast Horizon")
    forecast_periods = st.sidebar.number_input("Months to Forecast", min_value=1, max_value=24, value=6)
    forecast_days = forecast_periods * 30

    st.sidebar.divider()
    st.sidebar.subheader("4. Scenario Modifiers")
    st.sidebar.info("Add percentage changes to your forecast based on planned initiatives.")

    with st.sidebar.container():
        if "modifiers" not in st.session_state:
            st.session_state.modifiers = [{"label": "", "value": 0, "start_month": 1, "end_month": forecast_periods}]
        if st.button("➕ Add Another Scenario Modifier", type="secondary", use_container_width=True):
            st.session_state.modifiers.append({"label": "", "value": 0, "start_month": 1, "end_month": forecast_periods})

        for i, mod in enumerate(st.session_state.modifiers):
            col1, col2, col3, col4 = st.columns([2.5, 1.5, 3, 0.5])
            with col1:
                st.session_state.modifiers[i]["label"] = st.text_input(f"Modifier {i+1} Label", mod["label"], key=f"label_{i}", placeholder="e.g., New Content Series")
            with col2:
                st.session_state.modifiers[i]["value"] = st.number_input(f"% Change", -100, 1000, mod["value"], key=f"value_{i}", format="%d")
            with col3:
                current_start = mod["start_month"]
                current_end = mod.get("end_month", forecast_periods)
                current_start = max(1, min(current_start, forecast_periods))
                current_end = max(current_start, min(current_end, forecast_periods))
                month_range = st.slider(
                    f"Effect Duration (Months)", 1, forecast_periods, value=(current_start, current_end), key=f"month_range_{i}"
                )
                st.session_state.modifiers[i]["start_month"] = month_range[0]
                st.session_state.modifiers[i]["end_month"] = month_range[1]
            with col4:
                st.write("")
                if st.button("🗑️", key=f"delete_{i}"):
                    st.session_state.modifiers.pop(i)
                    st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("5. Conversions")
    st.sidebar.info("Enter your average Conversion Rate (%) to forecast conversions.")
    new_conversion_rate = st.sidebar.number_input("Conversion Rate (%)", min_value=0.0, value=st.session_state.current_conversion_rate, step=0.01, format="%.2f")
    if new_conversion_rate != st.session_state.current_conversion_rate:
        st.session_state.current_conversion_rate = new_conversion_rate
        if st.session_state.df_historical is not None:
            df_historical_with_conversions_recalc = st.session_state.df_historical.copy()
            df_historical_with_conversions_recalc['y_conversions'] = df_historical_with_conversions_recalc['y'] * (st.session_state.current_conversion_rate / 100)
            st.session_state.df_historical_conversions = df_historical_with_conversions_recalc

    st.sidebar.divider()
    st.sidebar.subheader("6. Content Decay")
    st.sidebar.info("This is an annual decline in organic traffic if no new content or SEO is undertaken.")
    decay_option = st.sidebar.selectbox(
        "Select Annual Content Decay Rate",
        ["None", "5% Annual Decay", "7.5% Annual Decay", "10% Annual Decay"]
    )
    annual_decay_rate = 0.0
    if decay_option == "5% Annual Decay":
        annual_decay_rate = 0.05
    elif decay_option == "7.5% Annual Decay":
        annual_decay_rate = 0.075
    elif decay_option == "10% Annual Decay":
        annual_decay_rate = 0.10

    st.divider()

    @st.cache_data(show_spinner="Generating forecast (this might take a moment)...")
    def generate_forecast(df_input, forecast_days_input, modifiers_input, conversion_rate_input, model_choice_input, prophet_seasonality_input, annual_decay_rate_input):
        df_for_model = df_input.copy()
        forecast_result = None

        forecast_period_start_date = df_for_model['ds'].max() + timedelta(days=1)

        # --- Model Fit and Prediction ---
        try:
            if model_choice_input == "Prophet":
                model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
                if prophet_seasonality_input == "Daily":
                    model.add_seasonality(name='daily', period=1, fourier_order=5)
                elif prophet_seasonality_input == "Weekly":
                    model.add_seasonality(name='weekly', period=7, fourier_order=3)
                elif prophet_seasonality_input == "Monthly":
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                elif prophet_seasonality_input == "All":
                    model.add_seasonality(name='daily', period=1, fourier_order=5)
                    model.add_seasonality(name='weekly', period=7, fourier_order=3)
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                model.fit(df_for_model)
                future = model.make_future_dataframe(periods=forecast_days_input)
                forecast_result = model.predict(future)
                forecast_result = forecast_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                forecast_result['yhat_uplift'] = forecast_result['yhat']
            elif model_choice_input == "Exponential Smoothing":
                model = ExponentialSmoothing(df_for_model['y'], trend=None, seasonal=None)
                fitted = model.fit()
                forecast_values = fitted.forecast(forecast_days_input)
                forecast_result = pd.DataFrame({
                    'ds': pd.date_range(start=df_for_model['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days_input),
                    'yhat': forecast_values
                })
                forecast_result['yhat_uplift'] = forecast_result['yhat']
            elif model_choice_input == "Holt-Winters (Multiplicative)":
                if len(df_for_model['ds']) > 1:
                    freq_days = (df_for_model['ds'].iloc[1] - df_for_model['ds'].iloc[0]).days
                    if freq_days >= 28 and freq_days <= 31:
                        seasonal_p = 12
                    else:
                        seasonal_p = 30
                else:
                    seasonal_p = 30
                if df_for_model['y'].isnull().any() or not pd.api.types.is_numeric_dtype(df_for_model['y']):
                    raise ValueError("Holt-Winters requires numeric 'y' values without NaNs.")
                if (df_for_model['y'] <= 0).any():
                    st.warning("Holt-Winters (Multiplicative) is sensitive to zero or negative values. Falling back to additive trend/seasonal.")
                    model = HoltWinters(df_for_model['y'], trend='add', seasonal='add', seasonal_periods=seasonal_p)
                else:
                    model = HoltWinters(df_for_model['y'], trend='add', seasonal='mul', seasonal_periods=seasonal_p)
                fitted = model.fit()
                forecast_values = fitted.forecast(forecast_days_input)
                forecast_result = pd.DataFrame({
                    'ds': pd.date_range(start=df_for_model['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days_input),
                    'yhat': forecast_values
                })
                forecast_result['yhat_uplift'] = forecast_result['yhat']
            elif model_choice_input == "ARIMA":
                model = ARIMA(df_for_model['y'], order=(1, 1, 1))
                fitted = model.fit()
                forecast_values = fitted.forecast(steps=forecast_days_input)
                forecast_result = pd.DataFrame({
                    'ds': pd.date_range(start=df_for_model['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days_input),
                    'yhat': forecast_values
                })
                forecast_result['yhat_uplift'] = forecast_result['yhat']
            elif model_choice_input == "Decay Model (Logarithmic)":
                last_value = df_for_model['y'].iloc[-1]
                decay_days = np.arange(1, forecast_days_input + 1)
                decay_values = last_value * np.exp(-0.01 * decay_days)
                forecast_result = pd.DataFrame({
                    'ds': pd.date_range(start=df_for_model['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days_input),
                    'yhat': decay_values,
                })
                forecast_result['yhat_uplift'] = forecast_result['yhat']
        except Exception as e:
            st.error(f"Model fitting failed: {e}")
            return None

        # --- Apply Content Decay ---
        if forecast_result is not None and annual_decay_rate_input > 0:
            annual_decay_decimal = annual_decay_rate_input
            daily_decay_factor = (1 - annual_decay_decimal) ** (1/365)
            decay_series_yhat = forecast_result['yhat'].copy()
            if 'yhat_lower' in forecast_result.columns:
                decay_series_yhat_lower = forecast_result['yhat_lower'].copy()
            else:
                decay_series_yhat_lower = forecast_result['yhat'].copy()
                forecast_result['yhat_lower'] = decay_series_yhat_lower
            if 'yhat_upper' in forecast_result.columns:
                decay_series_yhat_upper = forecast_result['yhat_upper'].copy()
            else:
                decay_series_yhat_upper = forecast_result['yhat'].copy()
                forecast_result['yhat_upper'] = decay_series_yhat_upper
            start_forecast_idx = forecast_result[forecast_result['ds'] >= forecast_period_start_date].index
            if len(start_forecast_idx) == 0:
                start_forecast_idx = [0]
            start_forecast_idx = start_forecast_idx[0]
            for i in range(start_forecast_idx, len(forecast_result)):
                days_from_start = (forecast_result.loc[i, 'ds'] - forecast_period_start_date).days
                decay_multiplier = (daily_decay_factor ** days_from_start)
                decay_series_yhat.loc[i] *= decay_multiplier
                decay_series_yhat_lower.loc[i] *= decay_multiplier
                decay_series_yhat_upper.loc[i] *= decay_multiplier
            forecast_result['yhat'] = decay_series_yhat
            forecast_result['yhat_lower'] = decay_series_yhat_lower
            forecast_result['yhat_upper'] = decay_series_yhat_upper
            forecast_result['yhat_uplift'] = forecast_result['yhat'].copy()

        # --- Apply Scenario Modifiers ---
        if forecast_result is not None:
            forecast_result['net_modifier_factor'] = 1.0
            first_forecast_month_period_overall = forecast_result['ds'].min().to_period("M")
            forecast_result['relative_month_num'] = (
                forecast_result['ds'].dt.to_period("M") - first_forecast_month_period_overall
            ).apply(lambda x: x.n) + 1
            forecast_start_date_for_modifiers_final = df_for_model['ds'].max() + timedelta(days=1)
            for mod in modifiers_input:
                if mod['label'] and mod['value'] != 0:
                    change_as_decimal = mod['value'] / 100.0
                    forecast_result.loc[
                        (forecast_result['relative_month_num'] >= mod['start_month']) &
                        (forecast_result['relative_month_num'] <= mod['end_month']) &
                        (forecast_result['ds'] >= forecast_start_date_for_modifiers_final),
                        'net_modifier_factor'
                    ] += change_as_decimal
            forecast_result.loc[
                forecast_result['ds'] >= forecast_start_date_for_modifiers_final,
                'yhat_uplift'
            ] = forecast_result.loc[forecast_result['ds'] >= forecast_start_date_for_modifiers_final, 'yhat'] * \
                forecast_result.loc[forecast_result['ds'] >= forecast_start_date_for_modifiers_final, 'net_modifier_factor']

            forecast_result['yhat_uplift_conversions'] = forecast_result['yhat_uplift'] * (conversion_rate_input / 100)
            forecast_result['yhat_conversions'] = forecast_result['yhat'] * (conversion_rate_input / 100)

            if 'yhat_lower' in forecast_result and 'yhat_upper' in forecast_result and not (forecast_result['yhat_lower'] == forecast_result['yhat']).all():
                forecast_result.loc[
                    forecast_result['ds'] >= forecast_period_start_date,
                    'yhat_lower'
                ] = forecast_result.loc[forecast_result['ds'] >= forecast_period_start_date, 'yhat_lower'] * \
                    (daily_decay_factor ** (forecast_result.loc[forecast_result['ds'] >= forecast_period_start_date, 'ds'] - forecast_period_start_date).dt.days)
                forecast_result.loc[
                    forecast_result['ds'] >= forecast_period_start_date,
                    'yhat_upper'
                ] = forecast_result.loc[forecast_result['ds'] >= forecast_period_start_date, 'yhat_upper'] * \
                    (daily_decay_factor ** (forecast_result.loc[forecast_result['ds'] >= forecast_period_start_date, 'ds'] - forecast_period_start_date).dt.days)
                forecast_result.loc[
                    forecast_result['ds'] >= forecast_start_date_for_modifiers_final,
                    'yhat_lower_conversions'
                ] = forecast_result['yhat_lower'] * (conversion_rate_input / 100) * forecast_result['net_modifier_factor']
                forecast_result.loc[
                    forecast_result['ds'] >= forecast_start_date_for_modifiers_final,
                    'yhat_upper_conversions'
                ] = forecast_result['yhat_upper'] * (conversion_rate_input / 100) * forecast_result['net_modifier_factor']
                forecast_result.loc[
                    forecast_result['ds'] < forecast_start_date_for_modifiers_final,
                    'yhat_lower_conversions'
                ] = forecast_result['yhat_lower'] * (conversion_rate_input / 100)
                forecast_result.loc[
                    forecast_result['ds'] < forecast_start_date_for_modifiers_final,
                    'yhat_upper_conversions'
                ] = forecast_result['yhat_upper'] * (conversion_rate_input / 100)
            else:
                forecast_result['yhat_lower'] = forecast_result['yhat']
                forecast_result['yhat_upper'] = forecast_result['yhat']
                forecast_result['yhat_lower_conversions'] = forecast_result['yhat_conversions']
                forecast_result['yhat_upper_conversions'] = forecast_result['yhat_conversions']

            forecast_result = forecast_result.drop(columns=['net_modifier_factor', 'relative_month_num'], errors='ignore')
        return forecast_result

    st.subheader("6. Generate & View Forecast")
    prophet_seasonality = None
    if model_choice == "Prophet":
        with st.expander("Prophet Model Settings", expanded=False):
            prophet_seasonality = st.selectbox(
                "Select seasonality modes to include",
                ["None", "Daily", "Weekly", "Monthly", "All"],
                key="prophet_seasonality_selector"
            )

    run_forecast_button_clicked = st.button("🚀 Run Forecast", type="primary", use_container_width=True)
    if run_forecast_button_clicked:
        if model_choice in ["Gradient Boosting (placeholder)", "Fourier Series Model (placeholder)", "Bayesian Structural Time Series (placeholder)", "Custom Growth/Decay Combo", "None (Use Basic Model)"]:
            st.warning("This model is a placeholder or not a valid selection.")
            st.session_state.forecast_data = None
        else:
            st.session_state.forecast_data = generate_forecast(
                df_input=df,
                forecast_days_input=forecast_days,
                modifiers_input=st.session_state.modifiers,
                conversion_rate_input=st.session_state.current_conversion_rate,
                model_choice_input=model_choice,
                prophet_seasonality_input=prophet_seasonality,
                annual_decay_rate_input=annual_decay_rate
            )
            if st.session_state.forecast_data is not None:
                st.success("Forecast generated successfully!")
            else:
                st.warning("Forecast generation failed. Please check your data and model parameters.")

    if st.session_state.forecast_data is not None and st.session_state.df_historical_conversions is not None:
        forecast = st.session_state.forecast_data
        df_historical_conversions = st.session_state.df_historical_conversions
        last_historical_date = df_historical_conversions['ds'].max()
        forecast_future = forecast[forecast['ds'] > last_historical_date].copy()
        df_historical_monthly = df_historical_conversions.set_index('ds').resample('M').sum(numeric_only=True).reset_index()
        forecast_future_monthly = forecast_future.set_index('ds').resample('M').sum(numeric_only=True).reset_index()
        st.divider()
        st.subheader("Summary Insights")
        col1, col2, col3, col4 = st.columns(4)
        avg_monthly_baseline_sessions = forecast_future_monthly['yhat'].mean()
        avg_monthly_scenario_sessions = forecast_future_monthly['yhat_uplift'].mean()
        avg_monthly_session_uplift_amount = avg_monthly_scenario_sessions - avg_monthly_baseline_sessions
        avg_monthly_session_uplift_percent = (avg_monthly_session_uplift_amount / avg_monthly_baseline_sessions) if avg_monthly_baseline_sessions > 0 else 0.0
        total_additional_sessions = (forecast_future_monthly['yhat_uplift'] - forecast_future_monthly['yhat']).sum()
        avg_monthly_conversions_scenario = forecast_future_monthly['yhat_uplift_conversions'].mean()
        total_additional_conversions = (forecast_future_monthly['yhat_uplift_conversions'] - forecast_future_monthly['yhat_conversions']).sum()
        with col1:
            delta_color_avg_sessions = "normal"
            if avg_monthly_session_uplift_amount > 0: delta_color_avg_sessions = "inverse"
            elif avg_monthly_session_uplift_amount < 0: delta_color_avg_sessions = "off"
            st.metric(label="Average Monthly Session Uplift", value=f"{avg_monthly_session_uplift_amount:,.0f}", delta=f"{avg_monthly_session_uplift_percent:.1%}", delta_color=delta_color_avg_sessions)
        with col2:
            st.metric(label="Total Additional Sessions", value=f"{total_additional_sessions:,.0f}")
        with col3:
            st.metric(label="Average Monthly Conversions", value=f"{avg_monthly_conversions_scenario:,.0f}")
        with col4:
            st.metric(label="Total Additional Conversions", value=f"{total_additional_conversions:,.0f}")
        st.divider()
        st.subheader("Forecast Visualizations (Monthly Aggregated)")
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.fill_between(df_historical_monthly['ds'], 0, df_historical_monthly['y'], color='#ADD8E6', alpha=0.8, label='Historical Actual Sessions')
        ax1.plot(df_historical_monthly['ds'], df_historical_monthly['y'], color='#1E90FF', linewidth=1.5, alpha=0.9)
        ax1.fill_between(forecast_future_monthly['ds'], 0, forecast_future_monthly['yhat'], color='#E0E0E0', alpha=0.8, label='Forecasted Baseline Sessions')
        ax1.plot(forecast_future_monthly['ds'], forecast_future_monthly['yhat'], color='#808080', linestyle='-', linewidth=1.5, alpha=0.9)
        forecast_future_monthly['uplift_diff'] = forecast_future_monthly['yhat_uplift'] - forecast_future_monthly['yhat']
        ax1.fill_between(forecast_future_monthly['ds'], forecast_future_monthly['yhat'], forecast_future_monthly['yhat_uplift'], where=(forecast_future_monthly['uplift_diff'] >= 0), color='#90EE90', alpha=0.8, label='Forecasted Uplift (Scenario)')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("SEO Sessions", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_title("SEO Traffic & Conversions Forecast")
        ax1.grid(True, linestyle=':', alpha=0.4)
        ax1.set_ylim(bottom=0)
        ax2 = ax1.twinx()
        ax2.plot(forecast_future_monthly['ds'], forecast_future_monthly['yhat_uplift_conversions'], label='Conversions with Scenarios', color='#FF4500', linestyle='-', linewidth=2.5)
        ax2.set_ylabel("Estimated Conversions", color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(bottom=0)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_handles = []
        legend_labels = []
        legend_handles.append(Patch(facecolor='#ADD8E6', alpha=0.8))
        legend_labels.append('Historical Actual Sessions')
        legend_handles.append(Patch(facecolor='#E0E0E0', alpha=0.8))
        legend_labels.append('Forecasted Baseline Sessions')
        legend_handles.append(Patch(facecolor='#90EE90', alpha=0.8))
        legend_labels.append('Forecasted Uplift (Scenario)')
        legend_handles.append(Line2D([0], [0], color='#FF4500', linestyle='-', linewidth=2.5))
        legend_labels.append('Conversions with Scenarios')
        ax2.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, fancybox=True, shadow=True, fontsize='medium')
        plt.tight_layout(rect=[0, 0.2, 1, 0.95])
        st.pyplot(fig)
        st.divider()
        st.subheader("Monthly Forecast Summary")
        forecast_monthly_summary = forecast_future_monthly.copy()
        forecast_monthly_summary['Monthly Uplift Sessions'] = forecast_monthly_summary['yhat_uplift'] - forecast_monthly_summary['yhat']
        forecast_monthly_summary['Monthly Uplift Conversions'] = forecast_monthly_summary['yhat_uplift_conversions'] - forecast_monthly_summary['yhat_conversions']
        forecast_monthly_summary = forecast_monthly_summary[['ds', 'yhat', 'yhat_uplift', 'Monthly Uplift Sessions', 'yhat_conversions', 'yhat_uplift_conversions', 'Monthly Uplift Conversions']]
        forecast_monthly_summary.columns = ['Month', 'Baseline Sessions', 'Uplift Sessions (Scenario)', 'Monthly Uplift/Decay Sessions', 'Baseline Conversions', 'Uplift Conversions (Scenario)', 'Monthly Uplift/Decay Conversions']
        forecast_monthly_summary['Month'] = forecast_monthly_summary['Month'].dt.strftime('%Y-%m')
        st.dataframe(forecast_monthly_summary.style.format({
            'Baseline Sessions': '{:,.0f}'.format,
            'Uplift Sessions (Scenario)': '{:,.0f}'.format,
            'Monthly Uplift/Decay Sessions': '{:,.0f}'.format,
            'Baseline Conversions': '{:,.0f}'.format,
            'Uplift Conversions (Scenario)': '{:,.0f}'.format,
            'Monthly Uplift/Decay Conversions': '{:,.0f}'.format
        }), use_container_width=True)
        st.divider()
        st.subheader("Download Forecast")
        export_choice = st.radio("Select Forecast Output Format", ["Weekly", "Monthly", "Daily (Full Detail)"], horizontal=True, key="export_format_radio")
        if export_choice == "Weekly":
            output_df = forecast_future.set_index('ds').resample('W').sum(numeric_only=True).reset_index()
            output_df['Uplift_Weekly_Sessions'] = output_df['yhat_uplift'] - output_df['yhat']
            output_df['Uplift_Weekly_Conversions'] = output_df['yhat_uplift_conversions'] - output_df['yhat_conversions']
            output = output_df[['ds', 'yhat', 'yhat_uplift', 'Uplift_Weekly_Sessions', 'yhat_conversions', 'yhat_uplift_conversions', 'Uplift_Weekly_Conversions']].copy()
            output.columns = ['Date', 'Baseline Sessions', 'Uplift Sessions (Scenario)', 'Uplift/Decay Sessions', 'Baseline Conversions', 'Uplift Conversions (Scenario)', 'Uplift/Decay Conversions']
        elif export_choice == "Monthly":
            output = forecast_monthly_summary.copy()
            output.columns = ['Date', 'Baseline Sessions', 'Uplift Sessions (Scenario)', 'Uplift/Decay Sessions', 'Baseline Conversions', 'Uplift Conversions (Scenario)', 'Uplift/Decay Conversions']
        else:
            output_df = forecast_future.copy()
            output_df['Uplift_Daily_Sessions'] = output_df['yhat_uplift'] - output_df['yhat']
            output_df['Uplift_Daily_Conversions'] = output_df['yhat_uplift_conversions'] - output_df['yhat_conversions']
            output = output_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'yhat_uplift', 'Uplift_Daily_Sessions',
                                'yhat_conversions', 'yhat_lower_conversions', 'yhat_upper_conversions', 'yhat_uplift_conversions', 'Uplift_Daily_Conversions']].copy()
            output.columns = ['Date', 'Baseline Sessions', 'Baseline Sessions Lower CI', 'Baseline Sessions Upper CI', 'Uplift Sessions (Scenario)', 'Uplift/Decay Sessions',
                              'Baseline Conversions', 'Baseline Conversions Lower CI', 'Baseline Conversions Upper CI', 'Uplift Conversions (Scenario)', 'Uplift/Decay Conversions']
        csv = output.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv,
            file_name=f"seo_forecast_{export_choice.lower().replace(' (full detail)', '')}_output.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    else:
        st.info("Click '🚀 Run Forecast' to generate and view the results!")
else:
    st.info("Please upload your historical data CSV file in the sidebar or load sample data to begin forecasting.")
