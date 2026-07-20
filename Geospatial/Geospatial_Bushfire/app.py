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


import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "processed_spatial_data.geojson")
model_path = os.path.join(BASE_DIR, "bushfire_model.pkl")
pipeline_path = os.path.join(BASE_DIR, "spatial_pipeline.py")

import streamlit as st

if not os.path.exists(data_path) or not os.path.exists(model_path):
    st.info("Preparing system and generating spatial data... Please wait.")
    try:
        # 🚀 FORCE INSTALL: Installs numpy and scikit-learn directly to the active system path
        st.write("Installing pipeline dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "pandas", "scikit-learn", "geopandas", "pyogrio", "joblib"], check=True)
        
        # Now run the pipeline script using the exact same python executable
        st.write("Running spatial_pipeline.py...")
        result = subprocess.run(
            [sys.executable, pipeline_path], 
            capture_output=True, 
            text=True, 
            cwd=BASE_DIR
        )
        
        if result.returncode != 0:
            st.error("### ❌ Pipeline Script Crashed!")
            st.code(result.stderr)
            st.stop()
            
        st.success("Pipeline executed successfully! Files created.")
    except Exception as e:
        st.error(f"Failed during environment preparation: {e}")
        st.stop()

# Now load the frameworks after ensuring environment preparation is complete
import geopandas as gpd
import joblib

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
    st.subheader("🎛️ Simulation Parameters")
    # 1. Slide to add extreme temperature changes to the canvas environment
    temp_offset = st.slider("Simulate Climate Temperature Offset (°C)", -10, 15, 0)
    
    # 2. Modify features downstream
    simulated_gdf = gdf.copy()
    simulated_gdf['temperature'] = simulated_gdf['temperature'] + temp_offset
    
    # 3. Request raw CONTINUOUS probability scores instead of fixed binary classes
    features_list = ['elevation', 'slope', 'ndvi', 'temperature', 'wind_speed', 'dist_to_powerline_m']
    
    # Extract the probability column corresponding to high-risk risk status [:, 1]
    simulated_gdf['risk_probability'] = model.predict_proba(simulated_gdf[features_list])[:, 1]
    
    # 4. Count anomalies using an adjustable risk threshold index
    high_risk_count = int((simulated_gdf['risk_probability'] > 0.55).sum())
    st.metric("Total Simulated HIGH-RISK Points (>55% Prob)", high_risk_count)

with col2:
    st.subheader("🗺️ Interactive Risk Map")

    m = folium.Map(location=[-33.5, 151.0], zoom_start=10, tiles="CartoDB positron")

    folium.PolyLine(
        locations=[[-33.5, 150.5], [-33.5, 151.5]],
        color="black", weight=4, opacity=0.7, tooltip="Simulated High-Voltage Infrastructure"
    ).add_to(m)

    marker_cluster = MarkerCluster(name="Susceptibility Points").add_to(m)

    # 5. Determine dot coloration continuously from the probability profile
    for _, row in simulated_gdf.iterrows():
        # High dynamic probability = Red, Medium = Orange, Low = Green
        if row['risk_probability'] > 0.55:
            color = "red"
        elif row['risk_probability'] > 0.35:
            color = "orange"
        else:
            color = "green"
            
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"Simulated Temp: {row['temperature']:.1f}°C<br>Risk Prob: {row['risk_probability']:.1%}"
        ).add_to(marker_cluster)

    st_folium(m, width=800, height=500, returned_objects=[])

