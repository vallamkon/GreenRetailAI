import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
from utils import load_delivery_data, compute_distances, estimate_emissions
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="GreenRetailAI", layout="wide")

# 🌍 Title & Header
st.markdown("<h1 style='color:#2E8B57;'>✨ GreenRetailAI</h1>", unsafe_allow_html=True)
st.markdown("**Empowering Sustainable Retail Logistics through Real-Time CO₂ Insights.**")

# Sidebar Controls
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2573/2573843.png", width=100)
st.sidebar.header("⚙️ Settings")
nrows = st.sidebar.slider("📦 Rows to Load", 1000, 100000, 5000, step=1000)
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])
distance_filter = st.sidebar.slider("📏 Filter by Distance (km)", 0, 50, (0, 50))
theme = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"])
map_style = "mapbox://styles/mapbox/dark-v9" if theme == "Dark" else "mapbox://styles/mapbox/light-v9"

# Load and process data
try:
    with st.spinner("🔄 Loading and processing..."):
        df = pd.read_csv(uploaded_file) if uploaded_file else load_delivery_data(nrows=nrows)
        df = compute_distances(df)
        df = estimate_emissions(df)
        df = df[(df["distance_km"] >= distance_filter[0]) & (df["distance_km"] <= distance_filter[1])]
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# 🔢 KPIs
col1, col2, col3 = st.columns(3)
col1.metric("🌍 CO₂ Emissions (kg)", f"{df['co2_kg'].sum():,.2f}")
col2.metric("⚡ EV-Friendly Deliveries", f"{df['suggest_ev'].sum():,}")
col3.metric("📏 Avg. Distance (km)", f"{df['distance_km'].mean():.2f}")

# 📊 EV Transition Impact Simulator
st.markdown("### ⚙️ EV Transition Impact Simulator")
ev_slider = st.slider("Assumed % EV adoption for <10 km", 0, 100, 50)
simulated_saving = df[df["suggest_ev"]]["co2_kg"].sum() * (ev_slider / 100)
st.success(f"🌱 Potential CO₂ Saved: **{simulated_saving:,.2f} kg**")

# 📉 Before vs After EV Adoption
st.markdown("### 📉 CO₂ Emissions: Before vs After EV Adoption")
before = df["co2_kg"].sum()
after = before - simulated_saving
fig_bar, ax_bar = plt.subplots()
ax_bar.bar(["Before", "After EV Adoption"], [before, after], color=["red", "green"])
ax_bar.set_ylabel("Total CO₂ (kg)")
st.pyplot(fig_bar)

# 🥧 EV Suitability Breakdown
st.markdown("### 🥧 EV Suitability Breakdown")
ev_labels = ['EV-suitable (<10km)', 'Not EV-Suitable']
ev_values = [df[df["suggest_ev"]].shape[0], df[~df["suggest_ev"]].shape[0]]
fig2, ax2 = plt.subplots()
ax2.pie(ev_values, labels=ev_labels, autopct='%1.1f%%', colors=["#4CAF50", "#FF7043"])
st.pyplot(fig2)

# 📈 Histogram of CO2 Emissions
st.markdown("### 📈 CO₂ Emissions Histogram")
fig, ax = plt.subplots()
df["co2_kg"].hist(bins=40, ax=ax, color="green", edgecolor='black')
ax.set_xlabel("CO₂ Emissions (kg)")
ax.set_ylabel("Deliveries")
st.pyplot(fig)

# 🌆 City-wise Emissions
if "city" in df.columns:
    st.markdown("### 🌆 City-wise CO₂ Emissions")
    city_emissions = df.groupby("city")["co2_kg"].sum().sort_values(ascending=False)
    fig_city, ax_city = plt.subplots()
    city_emissions.plot(kind="bar", ax=ax_city, color="#66BB6A")
    ax_city.set_ylabel("CO₂ (kg)")
    ax_city.set_title("CO₂ by City")
    st.pyplot(fig_city)

# 🗺️ 3D Interactive Delivery Origins (Hexagon Map)
st.markdown("### 🗺️ 3D Map: Delivery Origins")
try:
    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position='[poi_lng, poi_lat]',
        auto_highlight=True,
        radius=700,
        elevation_scale=50,
        elevation_range=[0, 1000],
        extruded=True,
        pickable=True
    )
    view_state = pdk.ViewState(
        latitude=df["poi_lat"].mean(),
        longitude=df["poi_lng"].mean(),
        zoom=10,
        pitch=45
    )
    st.pydeck_chart(pdk.Deck(
        map_style=map_style,
        initial_view_state=view_state,
        layers=[hex_layer],
        tooltip={"text": "Density here!"}
    ))
except Exception as e:
    st.warning(f"Could not render 3D hex map: {e}")

# 🔥 Heatmap for EV-Suitable Deliveries
st.markdown("### 🔥 EV Suitability Heatmap")
try:
    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=df[df["suggest_ev"]],
        get_position='[poi_lng, poi_lat]',
        opacity=0.8
    )
    st.pydeck_chart(pdk.Deck(
        initial_view_state=view_state,
        layers=[heat_layer],
        map_style=map_style
    ))
except Exception as e:
    st.warning(f"EV Heatmap issue: {e}")

# 📈 Forecasting Emissions
st.markdown("### 📈 Predicted CO₂ Emissions (ML Forecast)")
try:
    X = df[["distance_km"]]
    y = df["co2_kg"]
    model = LinearRegression().fit(X, y)
    df["predicted_co2"] = model.predict(X)
    fig_pred, ax_pred = plt.subplots()
    ax_pred.scatter(X, y, label="Actual", alpha=0.5, color="green")
    ax_pred.plot(X, df["predicted_co2"], label="Predicted", color="blue")
    ax_pred.set_xlabel("Distance (km)")
    ax_pred.set_ylabel("CO₂ Emissions (kg)")
    ax_pred.legend()
    st.pyplot(fig_pred)
except Exception as e:
    st.warning(f"Forecast model issue: {e}")

# 🏆 Store/Region Leaderboards
if "store_id" in df.columns:
    st.markdown("### 🏆 Top & Bottom CO₂ Emitting Stores")
    top = df.groupby("store_id")["co2_kg"].sum().sort_values().head(5)
    bottom = df.groupby("store_id")["co2_kg"].sum().sort_values().tail(5)
    st.markdown("#### ✅ Top 5 Sustainable Stores")
    st.bar_chart(top)
    st.markdown("#### ❌ Worst 5 Emitting Stores")
    st.bar_chart(bottom)

# 💸 Carbon Cost Estimation
if st.checkbox("💸 Show Estimated Carbon Cost"):
    carbon_price = st.selectbox("Price per kg of CO₂", [0.02, 0.03, 0.05])
    total_cost = df["co2_kg"].sum() * carbon_price
    st.metric("💰 Estimated Carbon Cost", f"₹{total_cost:,.2f}")

# ⬇️ Download Processed Data
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, "greenretailai_data.csv", "text/csv")

# 🧾 Sample Data Preview
st.markdown("### 🧾 Sample Processed Data")
st.dataframe(df.head(20))

# 🚀 Future Enhancements
st.markdown("---")
st.markdown("### 🚀 Future Scope & Integration")
st.markdown("""
- 🔄 Real-time logistics from Walmart Spark API
- 🛰️ Satellite-based pollution tracking
- 🧠 AI-driven delivery route planning
- 🏆 Emissions leaderboard by region/store
- 🤖 Chatbot assistant for delivery advice
""")

# 📣 FooterYou are offline
st.markdown("---")
st.caption("🚚 Built with 💚 by Team EcoFleet | Python • Streamlit • Pydeck • Pandas")




