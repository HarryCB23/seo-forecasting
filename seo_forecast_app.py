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
    page_title="SEO Forecasting Tool", # Page title for browser tab
    page_icon="📈", # You can use emojis or a path to an image file
    layout="wide", # Use the full width of the browser
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

# --- Introduction Section ---
st.title("📈 SEO & Revenue Forecasting Tool")
st.markdown("""
Welcome to the **SEO & Revenue Forecasting Tool**! This application helps you predict future organic search traffic and estimate potential revenue for evergreen content.

**Key Features:**
* **Traffic Forecasting:** Project future SEO sessions based on your historical data using various statistical models.
* **Scenario Planning:** Apply custom uplift or decay scenarios to model the impact of new initiatives.
* **Revenue Estimation:** Integrate your Revenue Per Mille (RPM) to translate traffic forecasts into estimated revenue.
* **Interactive Visualizations:** See your traffic and revenue forecasts on a clear, dual-axis graph.
* **Exportable Data:** Download detailed weekly or monthly forecast data for further analysis.

This tool is designed to provide actionable insights for your SEO strategy, helping you plan for growth and understand the financial impact of your efforts.
""")

# --- How This Works Section ---
with st.expander("❓ How This App Works", expanded=False):
    st.markdown("""
    This application allows you to forecast future SEO traffic and estimate potential revenue based on your historical data.

    **Step-by-Step Guide:**

    1.  **Upload Historical Data (Sidebar - Section 1):**
        * Prepare a CSV file with two columns:
            * `ds`: Date column (e.g., 'YYYY-MM-DD').
            * `y`: Numeric value representing your SEO sessions for that date.
        * Upload this CSV using the file uploader in the sidebar.

    2.  **Choose Forecasting Model (Sidebar - Section 2):**
        * Select the statistical model you believe best fits your data's patterns.
        * Hover over or click on the model name for a brief description.
        * *Prophet* is generally a good starting point for web traffic.

    3.  **Set Forecast Horizon (Sidebar - Section 3):**
        * Specify how many months into the future you want to forecast.

    4.  **Add Scenario Modifiers (Sidebar - Section 4):**
        * Model potential impacts of future initiatives (e.g., new content, PR campaigns).
        * Add a label, percentage change (positive for growth, negative for decay), and define the start and end months for the effect.

    5.  **Set Revenue Per Mille (Sidebar - Section 5):**
        * Input your average Revenue Per Mille (RPM). This is your estimated revenue per 1,000 sessions.
        * This will allow the forecast to include revenue projections.

    6.  **Run Forecast (Main Content):**
        * Click the "🚀 Run Forecast" button in the main area.
        * The app will process your data and display the historical data, a forecast plot, and monthly/weekly summaries.

    7.  **Download Forecast (Main Content):**
        * Choose your preferred output format (Weekly or Monthly) and download the forecast data as a CSV.

    **Tips for Best Results:**
    * Ensure your historical data is clean and consistent.
    * For Prophet, consider the seasonality options if your data has strong daily, weekly, or monthly patterns.
    * Experiment with different models and scenario modifiers to see various outcomes.
    """)
st.divider() # Only one divider here now

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")

# Step 1: Upload data moved to sidebar
st.sidebar.subheader("1. Upload Historical Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'ds' and 'y' columns", type=["csv"])

df = None # Initialize df outside the if block
# Initialize session state for df and forecast
if 'df_historical' not in st.session_state:
    st.session_state.df_historical = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'df_historical_revenue' not in st.session_state:
    st.session_state.df_historical_revenue = None # To store historical df with revenue for plotting


if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        # Corrected format to match "DD/MM/YYYY" as per the error message
        df['ds'] = pd.to_datetime(df['ds'], format="%d/%m/%Y")
        df = df.sort_values('ds')
        st.session_state.df_historical = df # Store in session state
        st.sidebar.success("Data uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}. Please ensure it's a CSV with 'ds' (date) and 'y' (value) columns. The date format should be 'DD/MM/YYYY' (e.g., '15/06/2023').")

# Use df from session state if available, otherwise use freshly uploaded df (if any)
df = st.session_state.df_historical


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

    st.sidebar.subheader("3. Forecast Horizon")
    # Step 4: Number of forecast months moved to sidebar
    forecast_periods = st.sidebar.number_input("Months to Forecast", min_value=1, max_value=24, value=6, help="Enter the number of months you wish to forecast into the future (max 24).")
    forecast_days = forecast_periods * 30 # Approximation for simplicity

    st.sidebar.divider()

    # Step 5: Scenario Modifiers moved to sidebar and wrapped in a container
    st.sidebar.subheader("4. Scenario Modifiers")
    st.sidebar.info("Add percentage changes to your forecast based on planned initiatives. Define a start and end month for the effect.")

    # Using st.container for grouping the modifiers section in the sidebar
    with st.sidebar.container():
        # Initialize modifiers with 'end_month' in session state if not present
        if "modifiers" not in st.session_state:
            st.session_state.modifiers = [{"label": "", "value": 0, "start_month": 1, "end_month": forecast_periods}] # Default end_month

        if st.button("➕ Add Another Scenario Modifier", type="secondary", use_container_width=True):
            st.session_state.modifiers.append({"label": "", "value": 0, "start_month": 1, "end_month": forecast_periods}) # Default end_month

        for i, mod in enumerate(st.session_state.modifiers):
            # Adjusted columns for new input types
            col1, col2, col3, col4, col5 = st.columns([2.5, 1.5, 1.5, 1.5, 0.5])
            with col1:
                st.session_state.modifiers[i]["label"] = st.text_input(f"Modifier {i+1} Label", mod["label"], key=f"label_{i}", placeholder="e.g., New Content Series")
            with col2:
                # Changed to number_input for % Change
                st.session_state.modifiers[i]["value"] = st.number_input(f"% Change", -50, 100, mod["value"], key=f"value_{i}", format="%d")
            with col3:
                # Changed to slider for Start Month, max value based on forecast_periods
                st.session_state.modifiers[i]["start_month"] = st.slider(
                    f"Start Month",
                    1, forecast_periods,
                    mod["start_month"],
                    key=f"start_{i}"
                )
            with col4:
                # Changed to slider for End Month, max value based on forecast_periods
                current_end_month_val = min(mod.get("end_month", forecast_periods), forecast_periods)
                st.session_state.modifiers[i]["end_month"] = st.slider(
                    "End Month (inclusive)",
                    st.session_state.modifiers[i]["start_month"], # Min value based on start month
                    forecast_periods,
                    current_end_month_val,
                    key=f"end_{i}",
                    help="The last month this modifier will apply. Defaults to the end of the forecast period if not changed."
                )
                # Ensure start_month doesn't exceed end_month if user changes end_month first (adjust for slider behavior)
                # This might still be tricky with sliders as they update instantly,
                # but the min_value constraint helps.
                if st.session_state.modifiers[i]["start_month"] > st.session_state.modifiers[i]["end_month"]:
                     st.session_state.modifiers[i]["start_month"] = st.session_state.modifiers[i]["end_month"]

            with col5:
                st.write("") # Spacer for alignment
                if st.button("🗑️", key=f"delete_{i}", help="Remove this modifier"):
                    st.session_state.modifiers.pop(i)
                    st.rerun() # Rerun to update the list instantly

    st.sidebar.divider()

    # New: Step 5: Revenue Section in Sidebar
    st.sidebar.subheader("5. Revenue")
    st.sidebar.info("Enter your average Revenue Per Mille (RPM) to forecast revenue alongside traffic.")
    revenue_per_mille = st.sidebar.number_input("Revenue Per Mille (RPM)", min_value=0.0, value=10.0, step=0.1, format="%.2f", help="Average revenue generated per 1000 sessions. E.g., 10.00 for $10 per 1000 sessions.")


    st.divider() # Visual separator in the main content

    # --- Main Content Area ---
    st.subheader("📊 Historical Data Preview")
    st.dataframe(df.tail(10), use_container_width=True) # Show more rows and use full width

    st.divider()

    st.subheader("6. Generate & View Forecast") # Changed section number

    # Define prophet_seasonality outside the button click if it needs to persist or be interactive
    # This is key to preventing resets.
    prophet_seasonality = None # Initialize to None or a default value
    if model_choice == "Prophet":
        with st.expander("Prophet Model Settings", expanded=False):
            # The selectbox will now live here and update `prophet_seasonality` directly
            # Adding a unique key to the selectbox to help with state management across reruns
            prophet_seasonality = st.selectbox(
                "Select seasonality modes to include",
                ["None", "Daily", "Weekly", "Monthly", "All"],
                help="Choose the seasonality components Prophet should account for.",
                key="prophet_seasonality_selector"
            )

    run_forecast_button_clicked = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    if run_forecast_button_clicked: # Primary button for main action
        if model_choice in ["Gradient Boosting (placeholder)", "Fourier Series Model (placeholder)", "Bayesian Structural Time Series (placeholder)", "Custom Growth/Decay Combo"]:
            st.warning("This model is a placeholder and will be available in a future version. Please select another model.")
            # Clear previous forecast if a placeholder is selected
            st.session_state.forecast_data = None
            st.session_state.df_historical_revenue = None
        else:
            with st.spinner("Generating forecast..."):
                # Step 6: Forecast based on selected model
                if model_choice == "Prophet":
                    model = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=False,
                        yearly_seasonality=False # Prophet includes yearly by default if enough data
                    )

                    # Apply seasonality based on the selection *before* fitting
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
                        # Apply modifier from start_month to end_month (inclusive)
                        uplift_factor = 1 + (mod['value'] / 100)
                        forecast.loc[
                            (forecast['month'] >= mod['start_month']) &
                            (forecast['month'] <= mod['end_month']),
                            'yhat_uplift'
                        ] *= uplift_factor

                # Calculate Revenue metrics
                forecast['yhat_revenue'] = (forecast['yhat'] / 1000) * revenue_per_mille
                forecast['yhat_uplift_revenue'] = (forecast['yhat_uplift'] / 1000) * revenue_per_mille
                if 'yhat_lower' in forecast: # Apply to lower/upper bounds if they exist (Prophet)
                    forecast['yhat_lower_revenue'] = (forecast['yhat_lower'] / 1000) * revenue_per_mille
                    forecast['yhat_upper_revenue'] = (forecast['yhat_upper'] / 1000) * revenue_per_mille

                # Calculate historical revenue for plotting
                df_with_revenue = df.copy() # Create a copy to add revenue without modifying original df
                df_with_revenue['y_revenue'] = (df_with_revenue['y'] / 1000) * revenue_per_mille
                st.session_state.df_historical_revenue = df_with_revenue # Store in session state


                st.session_state.forecast_data = forecast # Store forecast in session state

                st.success("Forecast generated successfully!")


    # Display forecast results if data exists in session state
    if st.session_state.forecast_data is not None and st.session_state.df_historical_revenue is not None:
        forecast = st.session_state.forecast_data
        df_historical_revenue = st.session_state.df_historical_revenue # Use the df with revenue data

        st.divider()

        # Step 8: Plot Forecast with Dual Axis (Traffic and Revenue)
        st.subheader("Forecast Plot (Traffic & Revenue)")

        fig, ax1 = plt.subplots(figsize=(12, 6)) # Increased figure size for better readability

        # Plot Traffic (Sessions) on the left Y-axis
        ax1.plot(df_historical_revenue['ds'], df_historical_revenue['y'], label='Historical Traffic', color='blue', linewidth=1.5)
        ax1.plot(forecast['ds'], forecast['yhat'], label='Baseline Traffic Forecast', color='blue', linestyle='-.', linewidth=1.5)
        ax1.plot(forecast['ds'], forecast['yhat_uplift'], label='Traffic with Scenarios', linestyle='--', color='darkblue', linewidth=2)
        if 'yhat_lower' in forecast and 'yhat_upper' in forecast:
            ax1.fill_between(forecast['ds'], forecast.get('yhat_lower', forecast['yhat']), forecast.get('yhat_upper', forecast['yhat']), alpha=0.1, color='lightblue')

        ax1.set_xlabel("Date")
        ax1.set_ylabel("SEO Sessions", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title("SEO Traffic and Estimated Revenue Forecast")

        # Create a second Y-axis for Revenue
        ax2 = ax1.twinx()
        ax2.plot(df_historical_revenue['ds'], df_historical_revenue['y_revenue'], label='Historical Revenue', color='red', linewidth=1.5)
        ax2.plot(forecast['ds'], forecast['yhat_revenue'], label='Baseline Revenue Forecast', color='red', linestyle='-.', linewidth=1.5)
        ax2.plot(forecast['ds'], forecast['yhat_uplift_revenue'], label='Revenue with Scenarios', linestyle='--', color='darkred', linewidth=2)

        ax2.set_ylabel("Estimated Revenue ($)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Combine legends from both axes and adjust position
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Adjusted bbox_to_anchor for the legend to place it clearly below the title and above the plot
        ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fancybox=True, shadow=True) # Adjusted bbox_to_anchor for better placement

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent labels/legends/title from overlapping, specify rect for padding
        st.pyplot(fig)


        # Step 8.5: Show Monthly Forecast Summary
        st.divider()
        st.subheader("Monthly Forecast Summary")
        forecast_monthly = forecast.set_index('ds').resample('M').sum(numeric_only=True)
        forecast_monthly.reset_index(inplace=True)

        # Always show both traffic and revenue in the monthly summary table
        forecast_monthly = forecast_monthly[['ds', 'yhat', 'yhat_uplift', 'yhat_revenue', 'yhat_uplift_revenue']]
        forecast_monthly.columns = ['Month', 'Baseline Sessions', 'Uplift Sessions', 'Baseline Revenue', 'Uplift Revenue']
        # Changed the date format for the 'Month' column in the summary table to 'YY-MM-DD'
        forecast_monthly['Month'] = forecast_monthly['Month'].dt.strftime('%y-%m-%d')


        st.dataframe(forecast_monthly, use_container_width=True)

        # Step 9: Export Option Selection
        st.divider()
        st.subheader("Download Forecast")
        # Added a unique key to the radio button to ensure its state persists independently
        export_choice = st.radio("Select Forecast Output Format", ["Weekly", "Monthly"], horizontal=True, key="export_format_radio")

        if export_choice == "Weekly":
            forecast_weekly = forecast.set_index('ds').resample('W').sum(numeric_only=True)
            forecast_weekly.reset_index(inplace=True)
            # Include both sessions and revenue in weekly export
            output = forecast_weekly[['ds', 'yhat', 'yhat_uplift', 'yhat_revenue', 'yhat_uplift_revenue']].copy()
            output.columns = ['Date', 'Baseline Sessions', 'Uplift Sessions', 'Baseline Revenue', 'Uplift Revenue']
        else:
            # Include both sessions and revenue in monthly export
            output = forecast_monthly.copy()
            # Ensure monthly output has the same columns as weekly for consistency in download
            output.columns = ['Date', 'Baseline Sessions', 'Uplift Sessions', 'Baseline Revenue', 'Uplift Revenue']


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
        st.info("Click '🚀 Run Forecast' to generate and view the results!")
else:
    st.info("Please upload your historical data CSV file in the sidebar to begin forecasting.")
    # Optional: Add an image or instructions here to guide the user
    # st.image("https://example.com/your-upload-image.png", width=300)
