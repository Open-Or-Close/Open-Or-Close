import os
import subprocess
import sys
import pandas as pd
import streamlit as st

# 1. Page Configuration (Must be the absolute first Streamlit command)
st.set_page_config(layout="wide", page_title="Spatial Bushfire Analytics")

st.title("🔥 Geospatial Bushfire Susceptibility Dashboard")
st.markdown(
    "This dashboard showcases machine learning predictions mapped directly onto interactive GIS layers without any data downloading friction."
)

# 2. Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "processed_spatial_data.geojson")
model_path = os.path.join(BASE_DIR, "bushfire_model.pkl")
pipeline_path = os.path.join(BASE_DIR, "spatial_pipeline.py")

# 3. Environment Preparation / Dependencies Bootstrapper
if not os.path.exists(data_path) or not os.path.exists(model_path):
    st.info("Preparing system and generating spatial data... Please wait.")
    try:
        st.write("Installing pipeline dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "numpy", "pandas", "scikit-learn", "geopandas", "pyogrio", "joblib", "folium", "streamlit-folium"], 
            check=True
        )
        
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

# 4. Framework Imports
import geopandas as gpd
import joblib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

@st.cache_data
def load_spatial_data():
    return gpd.read_file(data_path)

try:
    gdf = load_spatial_data()
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading generated files: {e}")
    st.stop()

# 5. Dashboard Layout & Interactions
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎛️ Simulation Parameters")
    
    # Request 3: Educational Notes about the parameters
    st.info(
        "💡 **What is Climate Temperature Offset?**\n"
        "This slider simulates a regional climate shift. Adjusting it uniformly adds or subtracts degrees "
        "across all mapped geographic coordinates, allowing you to pressure-test infrastructure resilience against heat waves.\n\n"
        "💡 **What are Total High-Risk Points (>55% Prob)?**\n"
        "This metric counts how many coordinates have crossed a strict 55% machine learning probability boundary for "
        "wildfire susceptibility based on the current shifted environment conditions."
    )
    
    # Slide to add extreme temperature changes to the canvas environment
    temp_offset = st.slider("Simulate Climate Temperature Offset (°C)", -10, 15, 0)
    
    # Modify features downstream
    simulated_gdf = gdf.copy()
    simulated_gdf['temperature'] = simulated_gdf['temperature'] + temp_offset
    
    # Request raw CONTINUOUS probability scores
    features_list = ['elevation', 'slope', 'ndvi', 'temperature', 'wind_speed', 'dist_to_powerline_m']
    simulated_gdf['risk_probability'] = model.predict_proba(simulated_gdf[features_list])[:, 1]
    
    # Count anomalies using an adjustable risk threshold index
    high_risk_count = int((simulated_gdf['risk_probability'] > 0.55).sum())
    st.metric("Total Simulated HIGH-RISK Points (>55% Prob)", high_risk_count)

    # Request 1: Dynamic Information Table including wind, temperature, elevation, etc.
    st.subheader("📊 High-Risk Points Table")
    high_risk_df = simulated_gdf[simulated_gdf['risk_probability'] > 0.55][
        ['temperature', 'wind_speed', 'elevation', 'slope', 'ndvi', 'risk_probability']
    ].sort_values(by='risk_probability', ascending=False)
    
    if not high_risk_df.empty:
        # Style the probability to percentage for clean visualization
        high_risk_df['risk_probability'] = high_risk_df['risk_probability'].map(lambda x: f"{x:.1%}")
        st.dataframe(high_risk_df, use_container_width=True, height=250)
    else:
        st.write("No points currently cross the high-risk threshold.")

with col2:
    st.subheader("🗺️ Interactive Risk Map")

    # Initialize Base Leaflet Canvas
    m = folium.Map(location=[-33.5, 151.0], zoom_start=10, tiles="CartoDB positron")

    # Plot the hazardous infrastructure corridor
    folium.PolyLine(
        locations=[[-33.5, 150.5], [-33.5, 151.5]],
        color="black", weight=4, opacity=0.7, tooltip="Simulated High-Voltage Infrastructure"
    ).add_to(m)

    marker_cluster = MarkerCluster(name="Susceptibility Points").add_to(m)

    # Determine dot coloration continuously from the probability profile
    for _, row in simulated_gdf.iterrows():
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
            popup=f"Simulated Temp: {row['temperature']:.1f}°C<br>Wind: {row['wind_speed']:.1f} km/h<br>Risk Prob: {row['risk_probability']:.1%}"
        ).add_to(marker_cluster)

    # Render optimized map element
    st_folium(m, width=800, height=500, returned_objects=[])

    # Request 2: Elegant Custom Dashboard Map Legend
    st.markdown("### 📋 Map Legend & Key Guide")
    
    legend_html = """
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 8px; border: 1px solid #ddd; font-family: sans-serif;">
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div>
                <h4 style="margin-top: 0; margin-bottom: 8px;">🔴 Risk Classifications</h4>
                <p style="margin: 4px 0;"><span style="color: red; font-weight: bold;">● Red Points:</span> High Risk (&gt; 55% Probability). Demands immediate environmental mitigation.</p>
                <p style="margin: 4px 0;"><span style="color: orange; font-weight: bold;">● Orange Points:</span> Moderate/Cautionary Risk (35% to 55% Probability).</p>
                <p style="margin: 4px 0;"><span style="color: green; font-weight: bold;">● Green Points:</span> Low/Safe Baseline Risk (&lt; 35% Probability).</p>
            </div>
            <div>
                <h4 style="margin-top: 0; margin-bottom: 8px;">🔢 Map Elements</h4>
                <p style="margin: 4px 0;"><b>⚫ Heavy Black Line:</b> High-Voltage Grid Infrastructure. Proximity to this line mathematically drives up point risks due to electrical spark hazards.</p>
                <p style="margin: 4px 0;"><b>🔵 Multi-Colored Numbered Circles:</b> These are <b>Marker Clusters</b>. To prevent browser lag, nearby points are bundled together. The number shows <i>how many individual survey coordinates</i> reside inside that region. Click a cluster circle to automatically zoom in and expand its contents.</p>
            </div>
        </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
