import streamlit as st
import geopandas as gpd
import subprocess
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(layout="wide", page_title="Spatial Bushfire Analytics")

st.title("🔥 Geospatial Bushfire Susceptibility Dashboard")
st.markdown(
    "This dashboard showcases machine learning predictions mapped directly onto interactive GIS layers without any data downloading friction.")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "processed_spatial_data.geojson")
model_path = os.path.join(BASE_DIR, "bushfire_model.pkl")
pipeline_path = os.path.join(BASE_DIR, "spatial_pipeline.py")

if not os.path.exists(data_path) or not os.path.exists(model_path):
    st.info("Generating spatial data and model files from pipeline... Please wait.")
    try:
        # 🔍 CAPTURE ERROR OUTPUT: Added capture_output=True to read the crash logs
        result = subprocess.run(
            ["python", pipeline_path], 
            capture_output=True, 
            text=True, 
            cwd=BASE_DIR
        )
        
        if result.returncode != 0:
            # Display the exact python error inside spatial_pipeline.py
            st.error("### ❌ Pipeline Script Crashed!")
            st.code(result.stderr)
            st.stop()
            
        st.success("Pipeline executed successfully! Files created.")
    except Exception as e:
        st.error(f"Failed to execute subprocess wrapper: {e}")
        st.stop()

@st.cache_data
def load_spatial_data():
    return gpd.read_file(data_path)

try:
    gdf = load_spatial_data()
    model = joblib.load(model_path)
    st.success("All data loaded! Rendering app...")
except Exception as e:
    st.error(f"Error loading generated files: {e}")
    st.stop()



# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Model Diagnostics")
    features = ['elevation', 'slope', 'ndvi', 'temperature', 'wind_speed', 'dist_to_powerline_m']

    # Feature Importances
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance',
                                                                                        ascending=False)
    st.dataframe(df_imp, use_container_width=True)

    st.subheader("🎛️ Simulation Parameters")
    temp_filter = st.slider("Filter Minimum Temperature (°C)", 25, 45, 30)
    filtered_gdf = gdf[gdf['temperature'] >= temp_filter]
    st.metric("Total Risk Points Displayed", len(filtered_gdf))

with col2:
    st.subheader("🗺️ Interactive Risk Map")

    # Initialize Folium Map centered around data coordinates
    m = folium.Map(location=[-33.5, 151.0], zoom_start=10, tiles="CartoDB positron")

    # Plot a representation of the hazardous power line
    folium.PolyLine(
        locations=[[-33.5, 150.5], [-33.5, 151.5]],
        color="black", weight=4, opacity=0.7, tooltip="Simulated High-Voltage Infrastructure"
    ).add_to(m)

    # Add predicted hazard locations
    for _, row in filtered_gdf.iterrows():
        color = "red" if row['fire_occurred'] == 1 else "green"
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"Temp: {row['temperature']:.1f}°C<br>NDVI: {row['ndvi']:.2f}<br>Risk Category: {'HIGH' if color == 'red' else 'LOW'}"
        ).add_to(m)

    st_folium(m, width=800, height=500, returned_objects=[])

