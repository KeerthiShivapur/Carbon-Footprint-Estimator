
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model and dataset
model = joblib.load("retail_carbon_model.pkl")
df = pd.read_csv("retail_carbon_data_1000.csv")

if 'store_id' in df.columns:
    df.drop('store_id', axis=1, inplace=True)

X = df.drop('carbon_footprint_kg', axis=1)
y = df['carbon_footprint_kg']
y_pred = model.predict(X)

# App configuration
st.set_page_config(page_title="Walmart Carbon Estimator", layout="wide")
st.title("🌍 Walmart Retail Carbon Footprint Estimator")
st.markdown("**Powered by Team: RetailGreeners 💡**")

# Sidebar
st.sidebar.image("https://1000logos.net/wp-content/uploads/2017/06/Walmart-Logo-2019.png", use_column_width=True)
st.sidebar.header("🔧 Input Emissions")

transport = st.sidebar.number_input("🚚 Transport (km)", 0.0, 2000.0, 100.0, step=10.0)
electricity = st.sidebar.number_input("⚡ Electricity (kWh)", 0.0, 10000.0, 500.0, step=50.0)
packaging = st.sidebar.number_input("📦 Packaging (kg)", 0.0, 2000.0, 50.0, step=5.0)

# Columns layout
col1, col2 = st.columns(2)

# Prediction block
with col1:
    st.subheader("🧮 Estimated Carbon Footprint")
    user_input = np.array([[transport, electricity, packaging]])
    prediction = model.predict(user_input)[0]

    if prediction > 600:
        st.error(f"🚨 High Carbon Footprint: **{prediction:.2f} kg CO₂**")
    elif prediction < 300:
        st.success(f"✅ Efficient Footprint: **{prediction:.2f} kg CO₂**")
    else:
        st.warning(f"⚠ Moderate Footprint: **{prediction:.2f} kg CO₂**")

    st.markdown("💡 _Try optimizing energy usage or packaging for better efficiency._")

# Metrics block
with col2:
    st.subheader("📊 Model Accuracy")
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.metric(label="📉 Mean Absolute Error (MAE)", value=f"{mae:.2f}")
    st.metric(label="📉 Mean Squared Error (MSE)", value=f"{mse:.2f}")
    st.metric(label="📈 R² Score", value=f"{r2:.3f}")

# Charts section
st.subheader("📈 Charts & Insights")

chart1, chart2 = st.columns(2)

# Scatter Plot
with chart1:
    st.markdown("**Actual vs Predicted Carbon Footprint**")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(y, y_pred, alpha=0.6, c='green')
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted")
    st.pyplot(fig1)

# Feature Importance Plot
with chart2:
    st.markdown("**📌 Feature Importance**")
    try:
        importance = model.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        importance_df.sort_values(by='Importance', ascending=True, inplace=True)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax2)
        ax2.set_title("Feature Importance")
        st.pyplot(fig2)
    except:
        st.error("Feature importance not available for this model.")

# Optional: Add forecast feature in future
st.markdown("---")
st.info("📅 Coming Soon: Forecast future CO₂ emissions based on monthly usage trends.")

# Footer
st.markdown("---")
st.markdown("© 2025 Team RetailGreeners | Walmart Hackathon")
