import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import ExponentialSmoothing as HoltWinters
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import timedelta
import calendar # Not directly used for plotting or calculations now, but kept for general utility

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="SEO Forecasting Tool",  # Page title for browser tab
    page_icon="📈",  # You can use emojis or a path to an image file
    layout="wide",  # Use the full width of the browser
    initial_sidebar_state="expanded"  # Keep sidebar expanded by default
)

# --- Introduction Section (More Concise) ---
st.title("📈 SEO & Conversions Forecasting Tool") # Updated title
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

# --- How This Works Section ---
with st.expander("❓ How This App Works", expanded=False):
    st.markdown("""
    This application allows you to forecast future SEO traffic and estimate potential conversions based on your historical data.

    **Step-by-Step Guide:**

    1.  **Upload Data:** Upload a CSV file containing your historical `ds` (date in DD/MM/YYYY format) and `y` (SEO sessions) data using the sidebar.
        * **Note:** The "Load Sample Data" button now reads from `data/Search PVs Example.csv` in your project's repository.
    2.  **Select Model:** Choose a suitable forecasting model from the sidebar. Each model has a brief explanation to guide your selection.
    3.  **Set Horizon:** Specify the number of months you wish to forecast into the future.
    4.  **Add Scenarios:** Use scenario modifiers to model anticipated changes (e.g., content launches, algorithm updates). Define a label, percentage change, and the start/end months for its effect using the sliders.
    5.  **Set Conversion Rate:** Enter your average Conversion Rate (%) in the sidebar to enable conversions forecasting.
    6.  **Run Forecast:** Click "🚀 Run Forecast" in the main section. The app will then display interactive traffic and conversions forecast plots and summary tables.
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
if 'df_historical_conversions' not in st.session_state:
    st.session_state.df_historical_conversions = None # To store historical df with conversions for plotting

# --- CONVERSION RATE INPUT - Initialized for use in data processing ---
if 'current_conversion_rate' not in st.session_state:
    st.session_state.current_conversion_rate = 1.0 # Default to 1.0%


# Function to process and store historical data (including conversions)
def process_historical_data(input_df, conversion_rate_value):
    """
    Processes historical data, renames columns, calculates conversions,
    and stores in session state.
    """
    df_temp = input_df.copy()

    if 'ds' not in df_temp.columns or 'y' not in df_temp.columns:
        st.error("Uploaded CSV must contain 'ds' (date) and 'y' (value) columns.")
        st.session_state.df_historical = None
        st.session_state.df_historical_conversions = None
        st.session_state.forecast_data = None # Clear forecast as data is invalid
        return

    try:
        df_temp['ds'] = pd.to_datetime(df_temp['ds'], dayfirst=True, errors='coerce')
        df_temp.dropna(subset=['ds'], inplace=True)
    except Exception as e:
        st.error(f"Error parsing date column 'ds': {e}. Please ensure it's in a recognizable date format, ideally 'DD/MM/YYYY'.")
        st.session_state.df_historical = None
        st.session_state.df_historical_conversions = None
        st.session_state.forecast_data = None
        return

    df_temp['y'] = pd.to_numeric(df_temp['y'], errors='coerce')
    df_temp.dropna(subset=['y'], inplace=True)

    if df_temp.empty:
        st.error("After processing, no valid historical data remains. Please check your CSV file.")
        st.session_state.df_historical = None
        st.session_state.df_historical_conversions = None
        st.session_state.forecast_data = None
        return

    df_temp.drop_duplicates(subset=['ds'], inplace=True)
    processed_df = df_temp.sort_values('ds').reset_index(drop=True)
    st.session_state.df_historical = processed_df

    # Calculate historical conversions immediately upon loading data
    df_historical_with_conversions = processed_df.copy()
    df_historical_with_conversions['y_conversions'] = df_historical_with_conversions['y'] * (conversion_rate_value / 100)
    st.session_state.df_historical_conversions = df_historical_with_conversions
    st.sidebar.success("Data loaded successfully!")
    st.session_state.forecast_data = None


# --- Sidebar Section 1: Upload Historical Data ---
st.sidebar.subheader("1. Upload Historical Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'ds' and 'y' columns", type=["csv"])

# Logic for loading data
if uploaded_file:
    try:
        temp_df = pd.read_csv(uploaded_file)
        process_historical_data(temp_df, st.session_state.current_conversion_rate)
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}. Please ensure it's a valid CSV with 'ds' (date) and 'y' (value) columns.")

# --- Sample Data Option (Reads from file in 'data' folder) ---
def load_sample_data_and_process_wrapper():
    """
    Loads sample data from a CSV file expected to be in a 'data/' subdirectory.
    """
    sample_file_path = './data/Search PVs Example.csv'
    try:
        temp_df = pd.read_csv(sample_file_path)
        process_historical_data(temp_df, st.session_state.current_conversion_rate)
    except FileNotFoundError:
        st.error(f"Sample data file not found at '{sample_file_path}'. Please ensure you have a 'data' folder in your repository with the sample CSV file inside.")
    except Exception as e:
        st.error(f"Error loading sample data file: {e}. Please check the file format.")


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

    # --- Sidebar Section 5: Conversions ---
    st.sidebar.subheader("5. Conversions") # Updated subheader
    st.sidebar.info("Enter your average Conversion Rate (%) to forecast conversions alongside traffic.") # Updated info
    new_conversion_rate = st.sidebar.number_input("Conversion Rate (%)", min_value=0.0, value=st.session_state.current_conversion_rate, step=0.01, format="%.2f", help="Average conversion rate (e.g., 1.50 for 1.5% conversion rate).") # Updated label, value, step, format, help

    if new_conversion_rate != st.session_state.current_conversion_rate:
        st.session_state.current_conversion_rate = new_conversion_rate
        # Recalculate historical conversions with the new conversion rate
        if st.session_state.df_historical is not None:
            df_historical_with_conversions_recalc = st.session_state.df_historical.copy()
            df_historical_with_conversions_recalc['y_conversions'] = df_historical_with_conversions_recalc['y'] * (st.session_state.current_conversion_rate / 100)
            st.session_state.df_historical_conversions = df_historical_with_conversions_recalc
        # No need to rerun, as it will naturally rerun on button click or when other inputs change.

    st.divider() # Visual separator in the main content

    # --- Caching the core forecasting function ---
    @st.cache_data(show_spinner="Generating forecast (this might take a moment)...")
    def generate_forecast(df_input, forecast_days_input, modifiers_input, conversion_rate_input, model_choice_input, prophet_seasonality_input): # Renamed rpm_input
        """
        Generates the forecast based on selected model and parameters.
        This function is cached to prevent re-running unnecessarily.
        """
        df_for_model = df_input.copy()
        
        forecast_result = None

        if model_choice_input == "Prophet":
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False
            )

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
            try:
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
            except Exception as e:
                st.error(f"Holt-Winters model failed: {e}. Check data or model choice.")
                return None

        elif model_choice_input == "ARIMA":
            try:
                model = ARIMA(df_for_model['y'], order=(1, 1, 1))
                fitted = model.fit()
                forecast_values = fitted.forecast(steps=forecast_days_input)
                forecast_result = pd.DataFrame({
                    'ds': pd.date_range(start=df_for_model['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days_input),
                    'yhat': forecast_values
                })
                forecast_result['yhat_uplift'] = forecast_result['yhat']
            except Exception as e:
                st.error(f"ARIMA model failed to fit: {e}. Try different model or preprocess data.")
                return None

        elif model_choice_input == "Decay Model (Logarithmic)":
            last_value = df_for_model['y'].iloc[-1]
            decay_days = np.arange(1, forecast_days_input + 1)
            decay_values = last_value * np.exp(-0.01 * decay_days)
            forecast_result = pd.DataFrame({
                'ds': pd.date_range(start=df_for_model['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days_input),
                'yhat': decay_values,
            })
            forecast_result['yhat_uplift'] = forecast_result['yhat']
        
        # --- Apply Modifiers to the forecast_result ---
        if forecast_result is not None:
            forecast_result['net_modifier_factor'] = 1.0
            
            forecast_start_date_for_modifiers = df_for_model['ds'].max() + timedelta(days=1)
            
            # Create a temporary column to map forecast dates to months relative to the forecast start
            first_forecast_month_period = forecast_result['ds'].min().to_period("M")
            forecast_result['relative_month_num'] = (
                forecast_result['ds'].dt.to_period("M") - first_forecast_month_period
            ).apply(lambda x: x.n) + 1

            for mod in modifiers_input:
                if mod['label'] and mod['value'] != 0:
                    change_as_decimal = mod['value'] / 100.0
                    forecast_result.loc[
                        (forecast_result['relative_month_num'] >= mod['start_month']) &
                        (forecast_result['relative_month_num'] <= mod['end_month']) &
                        (forecast_result['ds'] >= forecast_start_date_for_modifiers),
                        'net_modifier_factor'
                    ] += change_as_decimal
            
            forecast_result.loc[
                forecast_result['ds'] >= forecast_start_date_for_modifiers,
                'yhat_uplift'
            ] = forecast_result.loc[forecast_result['ds'] >= forecast_start_date_for_modifiers, 'yhat'] * \
                forecast_result.loc[forecast_result['ds'] >= forecast_start_date_for_modifiers, 'net_modifier_factor']
            
            forecast_result.loc[
                forecast_result['ds'] < forecast_start_date_for_modifiers,
                'yhat_uplift'
            ] = forecast_result.loc[forecast_result['ds'] < forecast_start_date_for_modifiers, 'yhat']

            # Calculate Conversions metrics
            forecast_result['yhat_conversions'] = forecast_result['yhat'] * (conversion_rate_input / 100)
            forecast_result['yhat_uplift_conversions'] = forecast_result['yhat_uplift'] * (conversion_rate_input / 100)

            # Ensure confidence intervals for conversions are handled
            if 'yhat_lower' in forecast_result and 'yhat_upper' in forecast_result and \
               not (forecast_result['yhat_lower'] == forecast_result['yhat']).all():
                
                forecast_result.loc[
                    forecast_result['ds'] >= forecast_start_date_for_modifiers,
                    'yhat_lower_conversions'
                ] = forecast_result['yhat_lower'] * (conversion_rate_input / 100) * forecast_result['net_modifier_factor']
                forecast_result.loc[
                    forecast_result['ds'] >= forecast_start_date_for_modifiers,
                    'yhat_upper_conversions'
                ] = forecast_result['yhat_upper'] * (conversion_rate_input / 100) * forecast_result['net_modifier_factor']
                
                forecast_result.loc[
                    forecast_result['ds'] < forecast_start_date_for_modifiers,
                    'yhat_lower_conversions'
                ] = forecast_result['yhat_lower'] * (conversion_rate_input / 100)
                forecast_result.loc[
                    forecast_result['ds'] < forecast_start_date_for_modifiers,
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
                help="Choose the seasonality components Prophet should account for.",
                key="prophet_seasonality_selector"
            )

    run_forecast_button_clicked = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    if run_forecast_button_clicked:
        if model_choice in ["Gradient Boosting (placeholder)", "Fourier Series Model (placeholder)", "Bayesian Structural Time Series (placeholder)", "Custom Growth/Decay Combo", "None (Use Basic Model)"]:
            st.warning("This model is a placeholder or not a valid selection and will be available in a future version. Please select another model.")
            st.session_state.forecast_data = None
        else:
            st.session_state.forecast_data = generate_forecast(
                df_input=df,
                forecast_days_input=forecast_days,
                modifiers_input=st.session_state.modifiers,
                conversion_rate_input=st.session_state.current_conversion_rate,
                model_choice_input=model_choice,
                prophet_seasonality_input=prophet_seasonality
            )
            if st.session_state.forecast_data is not None:
                st.success("Forecast generated successfully!")
            else:
                st.warning("Forecast generation failed. Please check your data and model parameters.")

    # --- Display Forecast Results if data exists in session state ---
    if st.session_state.forecast_data is not None and st.session_state.df_historical_conversions is not None:
        forecast = st.session_state.forecast_data
        df_historical_conversions = st.session_state.df_historical_conversions

        # --- Filter forecast data into historical fit and future forecast ---
        last_historical_date = df_historical_conversions['ds'].max()
        forecast_future = forecast[forecast['ds'] > last_historical_date].copy()

        # --- Aggregate data to MONTHLY for plotting ---
        df_historical_monthly = df_historical_conversions.set_index('ds').resample('M').sum(numeric_only=True).reset_index()
        forecast_future_monthly = forecast_future.set_index('ds').resample('M').sum(numeric_only=True).reset_index()


        st.divider()

        # --- KPI Summary Boxes ---
        st.subheader("Summary Insights")
        col1, col2, col3, col4, col5 = st.columns(5)

        # KPIs for the Forecast Period (Sum of monthly totals from forecast_future_monthly)
        total_historical_sessions = df_historical_conversions['y'].sum()
        total_baseline_forecast_sessions_kpi = forecast_future_monthly['yhat'].sum()
        total_uplift_sessions_kpi = forecast_future_monthly['yhat_uplift'].sum()
        total_baseline_forecast_conversions_kpi = forecast_future_monthly['yhat_conversions'].sum()
        total_uplift_conversions_kpi = forecast_future_monthly['yhat_uplift_conversions'].sum()

        # Calculate uplift amounts for KPIs
        sessions_uplift_amount = total_uplift_sessions_kpi - total_baseline_forecast_sessions_kpi
        conversions_uplift_amount = total_uplift_conversions_kpi - total_baseline_forecast_conversions_kpi

        with col1:
            st.metric(label="Total Historical Sessions", value=f"{total_historical_sessions:,.0f}")
        
        with col2:
            st.metric(label="Total Forecasted Baseline Sessions", value=f"{total_baseline_forecast_sessions_kpi:,.0f}")
        
        with col3:
            delta_color_sessions = "normal"
            if sessions_uplift_amount > 0: delta_color_sessions = "inverse"
            elif sessions_uplift_amount < 0: delta_color_sessions = "off"
            
            st.metric(label="Total Session Uplift (Forecast Period)", value=f"{sessions_uplift_amount:,.0f}",
                      delta=f"{sessions_uplift_amount / total_baseline_forecast_sessions_kpi:.1%}" if total_baseline_forecast_sessions_kpi > 0 else "0.0%",
                      delta_color=delta_color_sessions)
        
        with col4:
            delta_color_conversions = "normal"
            if conversions_uplift_amount > 0: delta_color_conversions = "inverse"
            elif conversions_uplift_amount < 0: delta_color_conversions = "off"
            
            st.metric(label="Total Conversion Uplift (Forecast Period)", value=f"{conversions_uplift_amount:,.0f}",
                      delta=f"{conversions_uplift_amount / total_baseline_forecast_conversions_kpi:.1%}" if total_baseline_forecast_conversions_kpi > 0 else "0.0%",
                      delta_color=delta_color_conversions)

        with col5:
            # Additional Monthly Sessions (Average monthly uplift of scenarios over baseline)
            monthly_uplift_sessions = forecast_future_monthly['yhat_uplift'] - forecast_future_monthly['yhat']
            avg_additional_monthly_sessions = monthly_uplift_sessions.mean() if not monthly_uplift_sessions.empty else 0
            
            # Additional Monthly Conversions (Average monthly uplift of scenarios over baseline)
            monthly_uplift_conversions = forecast_future_monthly['yhat_uplift_conversions'] - forecast_future_monthly['yhat_conversions']
            avg_additional_monthly_conversions = monthly_uplift_conversions.mean() if not monthly_uplift_conversions.empty else 0
            
            st.markdown(f"**Additional Monthly Insights**")
            st.markdown(f"Sessions: **{avg_additional_monthly_sessions:,.0f}**")
            st.markdown(f"Conversions: **{avg_additional_monthly_conversions:,.0f}**")

        st.divider()

        # --- PLOTTING: Traffic (Area Chart - Monthly) & Conversions (Line Chart on Secondary Axis - Monthly) ---
        st.subheader("Forecast Visualizations (Monthly Aggregated)")

        fig, ax1 = plt.subplots(figsize=(14, 7))

        # 1. Traffic Area Chart (Monthly)
        # Historical Data (Actuals) - AWR-like light blue/teal
        ax1.fill_between(df_historical_monthly['ds'], 0, df_historical_monthly['y'], color='#8ECFFD', alpha=0.8, label='Historical Actual Sessions') # Lighter blue
        ax1.plot(df_historical_monthly['ds'], df_historical_monthly['y'], color='#007bff', linewidth=1.5, alpha=0.9) # Darker line on top

        # Forecasted Data (Future only, Monthly)
        # Baseline (Inertial) - AWR-like light gray/blue
        ax1.fill_between(forecast_future_monthly['ds'], 0, forecast_future_monthly['yhat'], color='#E0E0E0', alpha=0.8, label='Forecasted Baseline Sessions') # Light gray
        ax1.plot(forecast_future_monthly['ds'], forecast_future_monthly['yhat'], color='#6c757d', linestyle='-', linewidth=1.5, alpha=0.9) # Darker gray line

        # Uplift (difference between yhat_uplift and yhat), stacked on top of baseline
        # Only plot positive uplift - AWR-like green
        forecast_future_monthly['uplift_diff'] = forecast_future_monthly['yhat_uplift'] - forecast_future_monthly['yhat']
        ax1.fill_between(forecast_future_monthly['ds'], forecast_future_monthly['yhat'], forecast_future_monthly['yhat_uplift'],
                         where=(forecast_future_monthly['uplift_diff'] >= 0),
                         color='#6BE585', alpha=0.8, label='Forecasted Uplift (Scenario)') # Lighter green
        # Note: If there's negative uplift (decay), this area will simply not be filled in green.
        # The line for total sessions will still show the adjusted value.

        ax1.set_xlabel("Date")
        ax1.set_ylabel("SEO Sessions", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_title("SEO Traffic & Conversions Forecast")
        ax1.grid(True, linestyle=':', alpha=0.4) # Lighter grid
        ax1.set_ylim(bottom=0)


        # 2. Conversions Line Chart (Monthly) - Only "Conversions with Scenarios"
        ax2 = ax1.twinx()
        ax2.plot(forecast_future_monthly['ds'], forecast_future_monthly['yhat_uplift_conversions'], 
                 label='Conversions with Scenarios', color='#FF6347', linestyle='-', linewidth=2.5) # Prominent Red/Orange

        ax2.set_ylabel("Estimated Conversions", color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(bottom=0)


        # Combine and refine legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_handles = []
        legend_labels = []

        # Traffic-related items (mimic AWR example's legend structure)
        legend_handles.append(Patch(facecolor='#8ECFFD', alpha=0.8)) # Historical area color
        legend_labels.append('Historical Actual Sessions')

        legend_handles.append(Patch(facecolor='#E0E0E0', alpha=0.8)) # Baseline area color
        legend_labels.append('Forecasted Baseline Sessions')
        
        legend_handles.append(Patch(facecolor='#6BE585', alpha=0.8)) # Uplift area color
        legend_labels.append('Forecasted Uplift (Scenario)')
        
        # Conversions-related item (line, color consistent with plot)
        legend_handles.append(Line2D([0], [0], color='#FF6347', linestyle='-', linewidth=2.5))
        legend_labels.append('Conversions with Scenarios')


        ax2.legend(legend_handles, legend_labels, 
                   loc='lower center', bbox_to_anchor=(0.5, -0.25), # Moved legend significantly below
                   ncol=2, fancybox=True, shadow=True, fontsize='medium') # Adjusted ncol and fontsize

        # Adjust layout to make space for the legend below the plot
        plt.tight_layout(rect=[0, 0.2, 1, 0.95]) # Increased bottom margin more aggressively
        st.pyplot(fig)


        # --- Monthly Forecast Summary ---
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

        # --- Export Option Selection ---
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
        else: # Daily (Full Detail)
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
