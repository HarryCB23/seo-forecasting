import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="SEO Forecasting Tool", layout="wide")

st.title("🔮 SEO Forecasting Tool")
st.markdown("Upload historic traffic data to forecast SEO performance and apply scenario-based modifiers.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with 'ds' and 'y' columns", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Validate columns
    if 'ds' not in data.columns or 'y' not in data.columns:
        st.error("CSV must contain 'ds' (date) and 'y' (value) columns.")
    else:
        data["ds"] = pd.to_datetime(data["ds"])
        st.subheader("📈 Historical Data")
        st.line_chart(data.set_index("ds")["y"])

        # Forecast horizon input
        periods_input = st.number_input("How many months to forecast?", min_value=1, max_value=36, value=12)

        # Fit Prophet model
        model = Prophet()
        model.fit(data)

        # Make future dataframe
        future = model.make_future_dataframe(periods=periods_input * 30, freq='D')
        forecast = model.predict(future)

        # --- Scenario Modifiers Section ---
        st.subheader("📊 Scenario Modifiers")

        # Initialize session state for dynamic sliders
        if "modifiers" not in st.session_state:
            st.session_state.modifiers = [{"label": "", "value": 0}]

        # Add new modifier
        if st.button("➕ Add new modifier"):
            st.session_state.modifiers.append({"label": "", "value": 0})

        # Optional reset
        if st.button("♻️ Reset modifiers"):
            st.session_state.modifiers = [{"label": "", "value": 0}]

        # Display sliders and calculate effect
        total_multiplier = 1.0
        updated_modifiers = []

        for i, mod in enumerate(st.session_state.modifiers):
            cols = st.columns([2, 1])
            label = cols[0].text_input(f"Modifier #{i+1} label", value=mod["label"], key=f"label_{i}")
            value = cols[1].slider(" ", min_value=-50, max_value=100, value=mod["value"], key=f"value_{i}")
            updated_modifiers.append({"label": label, "value": value})
            total_multiplier *= (1 + value / 100)

        # Update session state
        st.session_state.modifiers = updated_modifiers

        # Show net effect
        net_pct = round((total_multiplier - 1) * 100, 1)
        st.markdown(f"**Combined scenario effect:** {net_pct:+.1f}%")

        # Apply adjusted forecast
        forecast["yhat_adjusted"] = forecast["yhat"] * total_multiplier

        # --- Plot Forecast ---
        st.subheader("📉 Forecast vs Adjusted Forecast")

        fig = go.Figure()

        # Original forecast
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Original Forecast"
        ))

        # Adjusted forecast
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_adjusted"],
            mode="lines",
            name="Adjusted Forecast",
            line=dict(dash="dash", color="firebrick")
        ))

        fig.update_layout(title="Forecast with Scenario Modifiers", xaxis_title="Date", yaxis_title="Traffic")
        st.plotly_chart(fig, use_container_width=True)

        # Download option
        st.download_button(
            label="📥 Download Adjusted Forecast (CSV)",
            data=forecast[["ds", "yhat", "yhat_adjusted"]].to_csv(index=False),
            file_name="adjusted_forecast.csv",
            mime="text/csv"
        )
