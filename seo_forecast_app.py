import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import ExponentialSmoothing as HoltWinters
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import timedelta
import calendar # For getting month names

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="SEO Forecasting Tool",  # Page title for browser tab
    page_icon="📈",  # You can use emojis or a path to an image file
    layout="wide",  # Use the full width of the browser
    initial_sidebar_state="expanded"  # Keep sidebar expanded by default
)

# --- Introduction Section (More Concise) ---
st.title("📈 SEO & Revenue Forecasting Tool")
st.markdown("""
Predict your organic search traffic and estimate potential revenue with this easy-to-use tool.

**Key Features:**
* **Traffic Forecasts:** Project future SEO sessions.
* **Scenario Planning:** Model impact of initiatives (uplifts/decays).
* **Revenue Estimates:** Calculate revenue based on your RPM.
* **Visual Insights:** See traffic & revenue trends on interactive graphs.
* **Exportable Data:** Download detailed forecasts for analysis.

Gain actionable insights to strategize and understand the financial impact of your SEO efforts.
""")

# --- How This Works Section ---
with st.expander("❓ How This App Works", expanded=False):
    st.markdown("""
    This application allows you to forecast future SEO traffic and estimate potential revenue based on your historical data.

    **Step-by-Step Guide:**

    1.  **Upload Data:** Upload a CSV file containing your historical `ds` (date in DD/MM/YYYY format) and `y` (SEO sessions) data using the sidebar.
        * **Note for provided CSV:** If using "Recommended Search PVs...", `Date` column will be mapped to `ds` and `Total pageviews` to `y`.
    2.  **Select Model:** Choose a suitable forecasting model from the sidebar. Each model has a brief explanation to guide your selection.
    3.  **Set Horizon:** Specify the number of months you wish to forecast into the future.
    4.  **Add Scenarios:** Use scenario modifiers to model anticipated changes (e.g., content launches, algorithm updates). Define a label, percentage change, and the start/end months for its effect using the sliders.
    5.  **Set RPM:** Enter your average Revenue Per Mille (RPM) in the sidebar to enable revenue forecasting.
    6.  **Run Forecast:** Click "🚀 Run Forecast" in the main section. The app will then display interactive traffic and revenue forecast plots and summary tables.
    7.  **Download Data:** Select your desired output format (Weekly or Monthly) and download the comprehensive forecast CSV.

    **Tips for Best Results:**
    * Ensure your historical data is clean and consistent.
    * For Prophet, adjust seasonality settings if your traffic has strong daily, weekly, or monthly patterns.
    * Experiment with different models and scenario modifiers to explore various potential outcomes.
    """)
st.divider()


# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")

# Initialize session state for df and forecast
if 'df_historical' not in st.session_state:
    st.session_state.df_historical = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'df_historical_revenue' not in st.session_state:
    st.session_state.df_historical_revenue = None # To store historical df with revenue for plotting

# --- REVENUE PER MILLE INPUT - Initialized for use in data processing ---
if 'current_rpm' not in st.session_state:
    st.session_state.current_rpm = 10.0


# Function to process and store historical data (including revenue)
def process_historical_data(input_df, rpm_value):
    """
    Processes historical data, renames columns, calculates revenue,
    and stores in session state.
    """
    df_temp = input_df.copy()

    # Try to standardize column names to 'ds' and 'y'
    if 'Date' in df_temp.columns:
        df_temp.rename(columns={'Date': 'ds'}, inplace=True)
    if 'Total pageviews' in df_temp.columns: # Specific to the uploaded CSV
        df_temp.rename(columns={'Total pageviews': 'y'}, inplace=True)

    if 'ds' not in df_temp.columns or 'y' not in df_temp.columns:
        st.error("Uploaded CSV must contain 'ds' (date) and 'y' (value) columns, or 'Date' and 'Total pageviews' columns.")
        st.session_state.df_historical = None
        st.session_state.df_historical_revenue = None
        st.session_state.forecast_data = None # Clear forecast as data is invalid
        return

    try:
        # Attempt to parse with dayfirst=True for DD/MM/YYYY, or infer
        df_temp['ds'] = pd.to_datetime(df_temp['ds'], dayfirst=True, errors='coerce')
        # Drop rows where 'ds' could not be parsed
        df_temp.dropna(subset=['ds'], inplace=True)
    except Exception as e:
        st.error(f"Error parsing date column 'ds': {e}. Please ensure it's in a recognizable date format, ideally 'DD/MM/YYYY'.")
        st.session_state.df_historical = None
        st.session_state.df_historical_revenue = None
        st.session_state.forecast_data = None # Clear forecast as data is invalid
        return

    # Ensure 'y' is numeric
    df_temp['y'] = pd.to_numeric(df_temp['y'], errors='coerce')
    df_temp.dropna(subset=['y'], inplace=True) # Drop rows where 'y' is not numeric

    if df_temp.empty:
        st.error("After processing, no valid historical data remains. Please check your CSV file.")
        st.session_state.df_historical = None
        st.session_state.df_historical_revenue = None
        st.session_state.forecast_data = None
        return

    processed_df = df_temp.sort_values('ds').reset_index(drop=True)
    st.session_state.df_historical = processed_df

    # Calculate historical revenue immediately upon loading data
    df_historical_with_revenue = processed_df.copy()
    df_historical_with_revenue['y_revenue'] = (df_historical_with_revenue['y'] / 1000) * rpm_value
    st.session_state.df_historical_revenue = df_historical_with_revenue
    st.sidebar.success("Data loaded successfully!")
    # Clear previous forecast if new data is loaded
    st.session_state.forecast_data = None


# --- Sidebar Section 1: Upload Historical Data ---
st.sidebar.subheader("1. Upload Historical Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'ds' and 'y' columns", type=["csv"])

# Logic for loading data
if uploaded_file:
    try:
        temp_df = pd.read_csv(uploaded_file)
        process_historical_data(temp_df, st.session_state.current_rpm)
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}. Please ensure it's a valid CSV. The app attempts to map 'Date' to 'ds' and 'Total pageviews' to 'y' if present, otherwise expects 'ds' and 'y'.")

# --- Sample Data Option ---
def load_sample_data_and_process_wrapper():
    """Generates, processes, and loads a sample DataFrame into session state."""
    sample_data = {
        'ds': pd.to_datetime(['01/01/2023', '01/02/2023', '01/03/2023', '01/04/2023', '01/05/2023',
                              '01/06/2023', '01/07/2023', '01/08/2023', '01/09/2023', '01/10/2023',
                              '01/11/2023', '01/12/2023', '01/01/2024', '01/02/2024', '01/03/2024']).strftime("%d/%m/%Y").tolist(),
        'y': [10000, 11000, 10500, 12000, 13000, 12500, 14000, 13500, 15000, 14500, 16000, 15500, 17000, 16500, 18000]
    }
    temp_df = pd.DataFrame(sample_data)
    process_historical_data(temp_df, st.session_state.current_rpm) # Use current RPM for sample data


# Show sample data button only if no historical data is in session state
if st.session_state.df_historical is None:
    st.sidebar.button("Load Sample Data", on_click=load_sample_data_and_process_wrapper, use_container_width=True)
    st.sidebar.markdown("---") # Visual separator

# Use df from session state for current operations
df = st.session_state.df_historical


if df is not None:
    st.subheader("📊 Historical Data Preview")
    st.dataframe(df.tail(10), use_container_width=True) # Show more rows and use full width

    st.sidebar.divider()

    # --- Sidebar Section 2: Model Selection ---
    st.sidebar.subheader("2. Model Selection")
    model_choice = st.sidebar.selectbox("Choose Forecasting Model", [
        "Prophet",
        "Exponential Smoothing",
        "ARIMA",
        "Decay Model (Logarithmic)"
    ], help="Select the core statistical model best suited for your data's characteristics.")

    with st.sidebar.expander("Advanced Model Options", expanded=False):
        advanced_model_choice = st.selectbox("More Models", [
            "None (Use Basic Model)",
            "Holt-Winters (Multiplicative)",
            "Custom Growth/Decay Combo",
            "Gradient Boosting (placeholder)",
            "Fourier Series Model (placeholder)",
            "Bayesian Structural Time Series (placeholder)"
        ], help="Explore more specialized forecasting techniques if basic models are not sufficient.")

        if advanced_model_choice != "None (Use Basic Model)":
            model_choice = advanced_model_choice

    model_descriptions = {
        "Prophet": "**What it is:** Great for general web traffic, it finds patterns like daily or weekly ups and downs, and overall trends. It's robust even with missing data or sudden changes. \n\n**When to use:** Ideal for most evergreen content, like blog posts or service pages, where traffic might dip on weekends but generally grows over time.",
        "Exponential Smoothing": "**What it is:** Gives more importance to your most recent data, making it good for stable or slowly changing traffic patterns without strong, repeating ups and downs. \n\n**When to use:** Good for very consistent, established pages where traffic doesn't fluctuate much day-to-day or week-to-week, or for short-term forecasts.",
        "Holt-Winters (Multiplicative)": "**What it is:** This model identifies trends (growth/decline) and repeating seasonal patterns (like yearly peaks or monthly drops), adjusting for how these patterns get bigger as traffic grows. \n\n**When to use:** Best for content with clear seasonal cycles, such as travel guides (peak holidays), e-commerce product pages (seasonal sales), or events-related content.",
        "ARIMA": "**What it is:** A classic statistical model that's good at forecasting consistent traffic patterns by looking at past values and errors. It works best when your data tends to stay around a certain level. \n\n**When to use:** Suitable for very stable, predictable traffic, often for long-term historical data that doesn't have extreme swings or clear seasonal patterns (e.g., highly consistent informational content).",
        "Decay Model (Logarithmic)": "**What it is:** Specifically designed for traffic that starts high and then gradually drops off over time. It mimics how interest in a new topic or product launch might fade. \n\n**When to use:** Perfect for forecasting traffic for news articles, one-off event pages, or new product announcements where you expect an initial surge followed by a steady decline.",
        "Custom Growth/Decay Combo": "**What it is:** This option is for when you want to define your own growth or decay rates. It lets you create highly specific forecasts based on your unique insights or campaign plans. \n\n**When to use:** Use this if you have a specific campaign with a predicted start and end, or know exactly how much a new initiative (e.g., a major site redesign) will impact traffic over time.",
        "Gradient Boosting (placeholder)": "**What it is:** (Placeholder) An advanced machine learning model that can learn complex relationships in your data. It requires more setup to tell it what factors (like holidays or promotions) influence your traffic. \n\n**When to use:** Will be useful for highly complex scenarios with many influencing factors, when available.",
        "Fourier Series Model (placeholder)": "**What it is:** (Placeholder) Captures very complex repeating patterns in your data, like intricate weekly or monthly cycles that might not be obvious at first glance. \n\n**When to use:** Will be useful for data with highly nuanced and specific recurring patterns, when available.",
        "Bayesian Structural Time Series (placeholder)": "**What it is:** (Placeholder) A sophisticated model that breaks down your traffic into different components (like long-term trend, seasonality, and sudden events) and also provides a measure of how certain its predictions are. \n\n**When to use:** Will be useful for detailed probabilistic forecasting and understanding uncertainty, when available."
    }
    st.sidebar.caption(model_descriptions[model_choice])

    st.sidebar.divider()

    # --- Sidebar Section 3: Forecast Horizon ---
    st.sidebar.subheader("3. Forecast Horizon")
    forecast_periods = st.sidebar.number_input("Months to Forecast", min_value=1, max_value=24, value=6, help="Enter the number of months you wish to forecast into the future (max 24).")
    forecast_days = forecast_periods * 30 # Approximation for simplicity

    st.sidebar.divider()

    # --- Sidebar Section 4: Scenario Modifiers ---
    st.sidebar.subheader("4. Scenario Modifiers")
    st.sidebar.info("Add percentage changes to your forecast based on planned initiatives. Define a start and end month for the effect.")

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
                st.session_state.modifiers[i]["value"] = st.number_input(f"% Change", -100, 1000, mod["value"], key=f"value_{i}", format="%d", help="Percentage change (e.g., 10 for +10%, -5 for -5%).")
            with col3:
                current_start = mod["start_month"]
                current_end = mod.get("end_month", forecast_periods)

                current_start = max(1, min(current_start, forecast_periods))
                current_end = max(current_start, min(current_end, forecast_periods))

                month_range = st.slider(
                    f"Effect Duration (Months)",
                    1, forecast_periods,
                    value=(current_start, current_end),
                    key=f"month_range_{i}",
                    help="Select the start and end months (inclusive) within the forecast period for this modifier to apply."
                )
                st.session_state.modifiers[i]["start_month"] = month_range[0]
                st.session_state.modifiers[i]["end_month"] = month_range[1]

            with col4:
                st.write("") # Spacer for alignment
                if st.button("🗑️", key=f"delete_{i}", help="Remove this modifier"):
                    st.session_state.modifiers.pop(i)
                    st.rerun()

    st.sidebar.divider()

    # --- Sidebar Section 5: Revenue ---
    st.sidebar.subheader("5. Revenue")
    st.sidebar.info("Enter your average Revenue Per Mille (RPM) to forecast revenue alongside traffic.")
    # Update st.session_state.current_rpm when the user changes the input
    new_rpm = st.sidebar.number_input("Revenue Per Mille (RPM)", min_value=0.0, value=st.session_state.current_rpm, step=0.1, format="%.2f", key="rpm_input")

    if new_rpm != st.session_state.current_rpm:
        st.session_state.current_rpm = new_rpm
        # Recalculate historical revenue with the new RPM
        if st.session_state.df_historical is not None:
            df_historical_with_revenue_recalc = st.session_state.df_historical.copy()
            df_historical_with_revenue_recalc['y_revenue'] = (df_historical_with_revenue_recalc['y'] / 1000) * st.session_state.current_rpm
            st.session_state.df_historical_revenue = df_historical_with_revenue_recalc
        # No need to rerun, as it will naturally rerun on button click or when other inputs change.

    st.divider() # Visual separator in the main content

    # --- Main Content Area ---
    st.subheader("6. Generate & View Forecast")

    prophet_seasonality = None
    if model_choice == "Prophet":
        with st.expander("Prophet Model Settings", expanded=False):
            prophet_seasonality = st.selectbox(
                "Select seasonality modes to include",
                ["None", "Daily", "Weekly", "Monthly", "All"],
                help="Choose the seasonality components Prophet should account for.",
                key="prophet_seasonality_selector"
            )

    run_forecast_button_clicked = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    if run_forecast_button_clicked:
        if model_choice in ["Gradient Boosting (placeholder)", "Fourier Series Model (placeholder)", "Bayesian Structural Time Series (placeholder)", "Custom Growth/Decay Combo", "None (Use Basic Model)"]:
            st.warning("This model is a placeholder or not a valid selection and will be available in a future version. Please select another model.")
            st.session_state.forecast_data = None
        else:
            with st.spinner("Generating forecast..."):
                # --- Perform Forecast based on selected model ---
                if model_choice == "Prophet":
                    model = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=False,
                        yearly_seasonality=False
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
                    forecast['yhat_uplift'] = forecast['yhat'] # Initialize for modifiers

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
                    try:
                        # Attempt to infer frequency or use a sensible default
                        # If data is daily (like the sample data after processing),
                        # seasonal_periods for yearly would be 365.
                        # If data is monthly, seasonal_periods=12.
                        # Assuming the `ds` column contains daily entries if it's not strictly monthly.
                        # For the provided sample data (monthly '01/MM/YYYY'), a monthly seasonality of 12 is more fitting.
                        # Let's check the frequency of the input data to set seasonal_periods.
                        if len(df['ds']) > 1:
                            freq_days = (df['ds'].iloc[1] - df['ds'].iloc[0]).days
                            if freq_days >= 28 and freq_days <= 31: # Approximately monthly
                                seasonal_p = 12
                            else: # Assume daily or mixed, use a generic monthly period for the model
                                seasonal_p = 30
                        else:
                            seasonal_p = 30 # Default if not enough data to infer frequency

                        if df['y'].isnull().any() or not pd.api.types.is_numeric_dtype(df['y']):
                             raise ValueError("Holt-Winters requires numeric 'y' values without NaNs.")

                        # HoltWinters can be sensitive to short data or data with zero/negative values for 'mul' seasonal.
                        # Ensure all 'y' values are positive for multiplicative seasonality.
                        if (df['y'] <= 0).any():
                            st.warning("Holt-Winters (Multiplicative) is sensitive to zero or negative values. Please ensure your 'y' data is strictly positive for this model. Falling back to additive trend/seasonal.")
                            model = HoltWinters(df['y'], trend='add', seasonal='add', seasonal_periods=seasonal_p)
                        else:
                            model = HoltWinters(df['y'], trend='add', seasonal='mul', seasonal_periods=seasonal_p)

                        fitted = model.fit()
                        forecast_values = fitted.forecast(forecast_days)
                        forecast = pd.DataFrame({
                            'ds': pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days),
                            'yhat': forecast_values
                        })
                        forecast['yhat_uplift'] = forecast['yhat']
                    except Exception as e:
                        st.error(f"Holt-Winters model failed: {e}. This model can be sensitive to data characteristics (e.g., requires positive values for multiplicative seasonality, enough data for seasonal periods).")
                        st.session_state.forecast_data = None
                        st.stop()


                elif model_choice == "ARIMA":
                    try:
                        # ARIMA (p,d,q) can be challenging to auto-select. (1,1,1) is a basic start.
                        # Consider auto_arima for production or user-configurable orders.
                        model = ARIMA(df['y'], order=(1, 1, 1))
                        fitted = model.fit()
                        forecast_values = fitted.forecast(steps=forecast_days)
                        forecast = pd.DataFrame({
                            'ds': pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days),
                            'yhat': forecast_values
                        })
                        forecast['yhat_uplift'] = forecast['yhat']
                    except Exception as e:
                        st.error(f"ARIMA model failed to fit: {e}. This might happen with short or non-stationary data, or if the chosen order is unsuitable. Try a different model.")
                        st.session_state.forecast_data = None
                        st.stop()

                elif model_choice == "Decay Model (Logarithmic)":
                    last_value = df['y'].iloc[-1]
                    decay_days = np.arange(1, forecast_days + 1)
                    decay_values = last_value * np.exp(-0.01 * decay_days) # Adjustable decay rate
                    forecast = pd.DataFrame({
                        'ds': pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days),
                        'yhat': decay_values,
                    })
                    forecast['yhat_uplift'] = forecast['yhat']

                # --- Apply Modifiers ---
                if st.session_state.forecast_data is not None: # Only proceed if a forecast was successfully generated
                    forecast['net_modifier_factor'] = 1.0
                    forecast['forecast_month_num'] = ((forecast['ds'].dt.to_period("M") - forecast['ds'].iloc[0].to_period("M")).apply(lambda x: x.n)) + 1

                    for mod in st.session_state.modifiers:
                        if mod['label'] and mod['value'] != 0:
                            change_as_decimal = mod['value'] / 100.0
                            forecast.loc[
                                (forecast['forecast_month_num'] >= mod['start_month']) &
                                (forecast['forecast_month_num'] <= mod['end_month']),
                                'net_modifier_factor'
                            ] += change_as_decimal

                    forecast['yhat_uplift'] = forecast['yhat'] * forecast['net_modifier_factor']

                    # Calculate Revenue metrics using the latest RPM from session state
                    current_rpm_for_forecast = st.session_state.current_rpm
                    forecast['yhat_revenue'] = (forecast['yhat'] / 1000) * current_rpm_for_forecast
                    forecast['yhat_uplift_revenue'] = (forecast['yhat_uplift'] / 1000) * current_rpm_for_forecast

                    # Ensure confidence intervals are handled
                    if 'yhat_lower' in forecast and 'yhat_upper' in forecast and \
                       not (forecast['yhat_lower'] == forecast['yhat']).all() and \
                       not (forecast['yhat_upper'] == forecast['yhat']).all():
                        forecast['yhat_lower_revenue'] = (forecast['yhat_lower'] / 1000) * current_rpm_for_forecast
                        forecast['yhat_upper_revenue'] = (forecast['yhat_upper'] / 1000) * current_rpm_for_forecast
                    else:
                        # If no confidence intervals from model, make them equal to yhat for plotting consistency
                        forecast['yhat_lower'] = forecast['yhat']
                        forecast['yhat_upper'] = forecast['yhat']
                        forecast['yhat_lower_revenue'] = forecast['yhat_revenue']
                        forecast['yhat_upper_revenue'] = forecast['yhat_revenue']

                    st.session_state.forecast_data = forecast
                    st.success("Forecast generated successfully!")
                else:
                    st.warning("Forecast could not be generated. Please check model settings and data.")

    # --- Display Forecast Results if data exists in session state ---
    if st.session_state.forecast_data is not None and st.session_state.df_historical_revenue is not None:
        forecast = st.session_state.forecast_data
        df_historical_revenue = st.session_state.df_historical_revenue

        st.divider()

        # --- KPI Summary Boxes (New Feature) ---
        st.subheader("Summary Insights")
        col1, col2, col3, col4, col5 = st.columns(5)

        total_historical_sessions = df_historical_revenue['y'].sum()
        total_baseline_forecast_sessions = forecast['yhat'].sum()
        total_uplift_sessions = forecast['yhat_uplift'].sum()
        total_scenario_revenue = forecast['yhat_uplift_revenue'].sum()

        with col1:
            st.metric(label="Total Historical Sessions", value=f"{total_historical_sessions:,.0f}")
        with col2:
            st.metric(label="Forecasted Baseline Sessions", value=f"{total_baseline_forecast_sessions:,.0f}")
        with col3:
            # Calculate the uplift amount
            uplift_amount = total_uplift_sessions - total_baseline_forecast_sessions
            st.metric(label="Forecasted Uplift Sessions", value=f"{uplift_uplift_amount:,.0f}",
                      delta=f"{uplift_amount / total_baseline_forecast_sessions:.1%}" if total_baseline_forecast_sessions > 0 else "0.0%")
        with col4:
            st.metric(label="Forecasted Revenue (with Scenarios)", value=f"${total_scenario_revenue:,.2f}")

        with col5:
            # Calculate average monthly increase % for forecasted period
            # Get start and end dates for the forecast period (adjusted to month start for clean calculation)
            forecast_start_month = forecast['ds'].min().to_period('M')
            forecast_end_month = forecast['ds'].max().to_period('M')
            num_forecast_months = (forecast_end_month - forecast_start_month).n + 1

            if num_forecast_months > 1 and total_baseline_forecast_sessions > 0:
                # Calculate simple average monthly growth for uplift scenario
                # This could be more sophisticated (e.g., CAGR)
                monthly_forecast_uplift = forecast.set_index('ds').resample('M')['yhat_uplift'].sum()
                if len(monthly_forecast_uplift) > 1:
                    first_month_val = monthly_forecast_uplift.iloc[0]
                    last_month_val = monthly_forecast_uplift.iloc[-1]
                    if first_month_val > 0:
                        avg_monthly_growth_rate = ((last_month_val / first_month_val)**(1/(len(monthly_forecast_uplift)-1)) - 1)
                        st.metric(label="Avg. Monthly Growth (Scenario)", value=f"{avg_monthly_growth_rate:.1%}")
                    else:
                        st.metric(label="Avg. Monthly Growth (Scenario)", value="N/A")
                else:
                    st.metric(label="Avg. Monthly Growth (Scenario)", value="N/A")
            else:
                st.metric(label="Avg. Monthly Growth (Scenario)", value="N/A")

        st.divider()

        # --- IMPROVED PLOTTING: Traffic (Area Chart) & Revenue (Line Chart on Secondary Axis) ---
        st.subheader("Forecast Visualizations")

        fig, ax1 = plt.subplots(figsize=(14, 7)) # Increased figure size

        # 1. Traffic Area Chart
        # Historical Data
        ax1.fill_between(df_historical_revenue['ds'], 0, df_historical_revenue['y'], color='#007bff', alpha=0.4, label='Historical Actual Sessions')
        ax1.plot(df_historical_revenue['ds'], df_historical_revenue['y'], color='#007bff', linewidth=1.5) # Line over area for clarity

        # Forecasted Data
        # Baseline (Inertial) - This is the core forecast without modifiers
        ax1.fill_between(forecast['ds'], 0, forecast['yhat'], color='#6c757d', alpha=0.3, label='Forecasted Baseline Sessions')
        ax1.plot(forecast['ds'], forecast['yhat'], color='#6c757d', linestyle='-', linewidth=1.5)

        # Uplift/Decay (difference between yhat_uplift and yhat)
        # We need to make sure the uplift/decay is positive or negative.
        forecast['uplift_diff'] = forecast['yhat_uplift'] - forecast['yhat']

        # Plot positive uplift as green area on top
        ax1.fill_between(forecast['ds'], forecast['yhat'], forecast['yhat_uplift'],
                         where=(forecast['uplift_diff'] > 0),
                         color='#28a745', alpha=0.3, label='Forecasted Uplift Sessions (Scenario)')
        # Plot negative decay as red area below
        ax1.fill_between(forecast['ds'], forecast['yhat_uplift'], forecast['yhat'],
                         where=(forecast['uplift_diff'] < 0),
                         color='#dc3545', alpha=0.3, label='Forecasted Decay Sessions (Scenario)')

        # Line for Total Sessions with Scenarios
        ax1.plot(forecast['ds'], forecast['yhat_uplift'], color='#17a2b8', linestyle='--', linewidth=2, label='Total Sessions with Scenarios')


        # Confidence Interval (optional, if Prophet provides them and they are distinct)
        if 'yhat_lower' in forecast and 'yhat_upper' in forecast and \
           not (forecast['yhat_lower'] == forecast['yhat']).all():
            ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                             color='gray', alpha=0.1, label='Forecast Confidence Interval')

        ax1.set_xlabel("Date")
        ax1.set_ylabel("SEO Sessions", color='black') # Make label black for neutrality
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_title("SEO Traffic Forecast (Actual, Baseline, and Scenario Uplift/Decay)")
        ax1.grid(True, linestyle=':', alpha=0.6)


        # 2. Revenue Line Chart (on secondary Y-axis)
        ax2 = ax1.twinx()
        ax2.plot(df_historical_revenue['ds'], df_historical_revenue['y_revenue'], label='Historical Revenue', color='#6f42c1', linewidth=1.5, marker='o', markersize=3) # Purple for revenue, with marker
        ax2.plot(forecast['ds'], forecast['yhat_revenue'], label='Baseline Revenue Forecast', color='#fd7e14', linestyle=':', linewidth=1.5) # Orange dotted
        ax2.plot(forecast['ds'], forecast['yhat_uplift_revenue'], label='Revenue with Scenarios', color='#e83e8c', linestyle='-', linewidth=2) # Pink solid

        if 'yhat_lower_revenue' in forecast and 'yhat_upper_revenue' in forecast and \
           not (forecast['yhat_lower_revenue'] == forecast['yhat_revenue']).all():
            ax2.fill_between(forecast['ds'], forecast['yhat_lower_revenue'], forecast['yhat_upper_revenue'],
                             color='#e83e8c', alpha=0.1, label='Revenue Confidence Interval')


        ax2.set_ylabel("Estimated Revenue ($)", color='black') # Neutral color
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(bottom=0) # Ensure revenue starts from 0


        # Combine and refine legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Sort labels to ensure consistent legend order
        combined_legend = sorted(zip(lines + lines2, labels + labels2), key=lambda x: x[1])

        ax2.legend(*zip(*combined_legend), loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True, fontsize='small') # Increased columns for better layout

        plt.tight_layout(rect=[0, 0.03, 1, 0.9]) # Adjust layout to prevent labels/legends/title from overlapping
        st.pyplot(fig)


        # Step 8.5: Show Monthly Forecast Summary
        st.divider()
        st.subheader("Monthly Forecast Summary")
        forecast_monthly = forecast.set_index('ds').resample('M').sum(numeric_only=True)
        forecast_monthly.reset_index(inplace=True)

        # Calculate monthly uplift difference for the table
        forecast_monthly['Monthly Uplift Sessions'] = forecast_monthly['yhat_uplift'] - forecast_monthly['yhat']
        forecast_monthly['Monthly Uplift Revenue'] = forecast_monthly['yhat_uplift_revenue'] - forecast_monthly['yhat_revenue']

        forecast_monthly = forecast_monthly[['ds', 'yhat', 'yhat_uplift', 'Monthly Uplift Sessions', 'yhat_revenue', 'yhat_uplift_revenue', 'Monthly Uplift Revenue']]
        forecast_monthly.columns = ['Month', 'Baseline Sessions', 'Uplift Sessions (Scenario)', 'Monthly Uplift/Decay Sessions', 'Baseline Revenue', 'Uplift Revenue (Scenario)', 'Monthly Uplift/Decay Revenue']
        forecast_monthly['Month'] = forecast_monthly['Month'].dt.strftime('%Y-%m') # Consistent YYYY-MM format

        # Format numeric columns for better readability
        st.dataframe(forecast_monthly.style.format({
            'Baseline Sessions': '{:,.0f}'.format,
            'Uplift Sessions (Scenario)': '{:,.0f}'.format,
            'Monthly Uplift/Decay Sessions': '{:,.0f}'.format,
            'Baseline Revenue': '${:,.2f}'.format,
            'Uplift Revenue (Scenario)': '${:,.2f}'.format,
            'Monthly Uplift/Decay Revenue': '${:,.2f}'.format
        }), use_container_width=True)

        # Step 9: Export Option Selection
        st.divider()
        st.subheader("Download Forecast")
        export_choice = st.radio("Select Forecast Output Format", ["Weekly", "Monthly", "Daily (Full Detail)"], horizontal=True, key="export_format_radio")

        if export_choice == "Weekly":
            output_df = forecast.set_index('ds').resample('W').sum(numeric_only=True).reset_index()
            output_df['Monthly Uplift Sessions'] = output_df['yhat_uplift'] - output_df['yhat']
            output_df['Monthly Uplift Revenue'] = output_df['yhat_uplift_revenue'] - output_df['yhat_revenue']
            output = output_df[['ds', 'yhat', 'yhat_uplift', 'Monthly Uplift Sessions', 'yhat_revenue', 'yhat_uplift_revenue', 'Monthly Uplift Revenue']].copy()
            output.columns = ['Date', 'Baseline Sessions', 'Uplift Sessions (Scenario)', 'Uplift/Decay Sessions', 'Baseline Revenue', 'Uplift Revenue (Scenario)', 'Uplift/Decay Revenue']
        elif export_choice == "Monthly":
            # Reuse forecast_monthly from above which is already formatted
            output = forecast_monthly.copy()
            output.columns = ['Date', 'Baseline Sessions', 'Uplift Sessions (Scenario)', 'Uplift/Decay Sessions', 'Baseline Revenue', 'Uplift Revenue (Scenario)', 'Uplift/Decay Revenue']
        else: # Daily (Full Detail)
            output_df = forecast.copy()
            output_df['Uplift_Daily_Sessions'] = output_df['yhat_uplift'] - output_df['yhat']
            output_df['Uplift_Daily_Revenue'] = output_df['yhat_uplift_revenue'] - output_df['yhat_revenue']
            # Select relevant columns for daily export, including confidence intervals if available
            output = output_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'yhat_uplift', 'Uplift_Daily_Sessions',
                                'yhat_revenue', 'yhat_lower_revenue', 'yhat_upper_revenue', 'yhat_uplift_revenue', 'Uplift_Daily_Revenue']].copy()
            output.columns = ['Date', 'Baseline Sessions', 'Baseline Sessions Lower CI', 'Baseline Sessions Upper CI', 'Uplift Sessions (Scenario)', 'Uplift/Decay Sessions',
                              'Baseline Revenue', 'Baseline Revenue Lower CI', 'Baseline Revenue Upper CI', 'Uplift Revenue (Scenario)', 'Uplift/Decay Revenue']


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
