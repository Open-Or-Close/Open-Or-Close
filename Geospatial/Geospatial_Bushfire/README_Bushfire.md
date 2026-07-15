#### Author: Reza Mohseni   
#### Contact: Reza2013mohseni@gmail.com

# 🗺️ End-to-End Geospatial Bushfire Analytics & Machine Learning Pipeline


## 📌 Project Overview & Value Proposition
Predicting wildland fires requires a complex intersection of dynamic climate conditions, topographics, and spatial human infrastructure vectors. This project delivers a complete, production-ready Geospatial Data Science pipeline that simulates geographical risk environments, engineers localized spatial infrastructure interactions, trains an ensemble Machine Learning model, and deploys an interactive decision-support GIS dashboard.

### Why this architecture matters:
*   **Zero Data Overhead:** Implements an intelligent, self-contained spatial simulation engine using core geometric algorithms. It generates reproducible coordinates, terrain profiles, and environmental indices without requiring resource-heavy raster downloads or external API access keys.
*   **True Spatial Feature Engineering:** Demonstrates actual vector-to-raster style proximity operations by calculating spatial interaction matrices relative to critical utility networks.
*   **Stakeholder Transparency:** Translates opaque ensemble model weights into clear feature importances, accompanied by a dynamic spatial application allowing stakeholders to visually simulate "what-if" risk scenarios.



## 🛠️ System Architecture & Tech Stack
The project is built on modular, open-source principles separating data processing from presentation layout:
*   **Spatial Data Engineering:** `GeoPandas`, `Shapely` (Geospatial vector geometry & projection mechanics)
*   **Predictive Analytics Engine:** `Scikit-Learn` (Random Forest Ensemble Framework)
*   **Serialization Management:** `Joblib`
*   **Visualization & UI Frame:** `Streamlit`, `Folium` (Leaflet wrapper), `Streamlit-Folium`



## 📈 Step-by-Step Implementation Breakdown

### Step 1: Spatial Grid Initialization & Coordinate Synthesis
The pipeline kicks off inside `spatial_pipeline.py` by establishing a bounding box containing 2,000 coordinate observation points. Using `NumPy` configurations with fixed seeds to guarantee absolute replication across deployments, points are generated within a true geographical box (spanning coordinates around Southeast Australia/New South Wales boundaries: `-34.0°` to `-33.0°` Latitude and `150.5°` to `151.5°` Longitude).

### Step 2: Environmental Matrix Synthesis (Multi-Layer Simulation)
For every synthetic geographic point, the framework constructs an array of attributes mapping real-world environmental phenomena:
*   **Terrain Profile:** Elevation profiles ranging from $100\text{m}$ to $1,200\text{m}$ paired with land-surface slopes ($0^\circ$ to $35^\circ$).
*   **Vegetation Biomass:** Normalized Difference Vegetation Index ($NDVI$) simulations mapping canopy conditions from dry ($0.1$) to dense/wet vegetation ($0.8$).
*   **Climatic Drivers:** Highly variable surface temperatures ($25^\circ\text{C}$ to $45^\circ\text{C}$) and localized wind speed currents ($5\text{ km/h}$ to $60\text{ km/h}$).

All matrices are bound into a structured `GeoPandas.GeoDataFrame` mapped to global standard GPS coordinates (**EPSG:4326**).

### Step 3: High-Voltage Infrastructure Proximity Engineering (GIS Layer Calculation)
Fires frequently ignite due to structural failures in localized utility assets. To address this spatial pattern:
1. A linear vector feature (`LineString`) is drawn directly across the map space to symbolize a high-voltage power transmission corridor.
2. Because standard GPS degrees distort physical distances, the system performs an automated projection transformation to a metric coordinate system (**EPSG:3857** / Web Mercator).
3. The Euclidean distance from each individual land coordinate point to the infrastructure line asset is calculated down to the exact meter (`dist_to_powerline_m`).

### Step 4: Deterministic Target Generation (Pseudo-Absence Fire Matrix)
Using an authoritative risk function based on wildland fire dynamics, a composite environmental hazard coefficient is assigned to each node:
$$\text{Hazard} = (0.4 \times \text{Temp}) + (0.3 \times \text{Wind}) + (0.2 \times \text{Slope}) - \frac{\text{Distance to Line}}{5000} - (10 \times NDVI)$$

This ensures that areas matching high temperatures, steep hillsides, dry brush, and utility asset proximity carry elevated risk numbers. The scores are normalized via a Sigmoid probability curve; coordinates hitting a risk threshold $> 65\%$ are classified as true positive fire occurrences (`1`), while remaining points form our control baseline/no-fire absence dataset (`0`).

### Step 5: Predictive Analytics & Model Training Pipeline
The spatial matrix is decoupled into target classifications (`y`) and 6 analytical feature vectors (`X`).
*   **Data Splitting:** Data is partitioned using a 70/30 split strategy, setting aside 30% of the nodes into a locked evaluation set.
*   **Ensemble Construction:** A **Random Forest Classifier** composed of 100 deep decision trees is compiled and trained using `.fit()`. The trees systematically evaluate multi-layered conditions to extract complex, non-linear safety thresholds.

### Step 6: Model Validation & Asset Deployment
The model runs evaluation routines on the hidden 30% validation pool. The results routinely achieve strong Area Under the Receiver Operating Characteristic curve metrics (**ROC-AUC ~ 0.90+**), indicating optimal separation capabilities. 
Upon verified evaluation, the trained model architecture is exported to a standalone byte-file (`bushfire_model.pkl`) and the processed spatial geometries are written out to standard open-source GIS file formats (`processed_spatial_data.geojson`).

### Step 7: Web Dashboard & Decision-Support Interface Deployment
The `app.py` script provisions a highly visual browser application via **Streamlit** to present these analytics to end users:
*   **Left Control Panel (Diagnostics & Simulations):** Computes and charts feature importance calculations, putting the exact driving risks in plain sight. It features a live `Streamlit Slider` letting operators filter microclimates by temperature.
*   **Right Display Canvas (Interactive Leaflet Map):** Renders a zoomable, map layer tracking point features. Safe zones are drawn as green nodes, and zones flagged as high susceptibility by the Machine Learning model appear as red hazard nodes. Clicking any location node reveals an interactive data sheet detailing its temperature, slope, and proximity to infrastructure.


# Outcome

### When the temperature is **26°C**

![Temperature 26°C](https://github.com/user-attachments/assets/2d80c8fb-ab8a-49e7-a4ac-bd13b52e833a)

<p align="center">  Figure 1. Predicted bushfire risk when the temperature is 26°C.



###  When the temperature is **42°C**

![Temperature 42°C](https://github.com/user-attachments/assets/e73ca54b-7543-4883-b876-45c9706d5bb3)

<p align="center"> Figure 2. Predicted bushfire risk when the temperature is 42°C.


---

# Geospatial Bushfire Risk Analyzer 🌲🔥

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://open-or-close-duazna7xn8daqhfjghkkbn.streamlit.app/)

An interactive geospatial dashboard and machine learning pipeline built to predict, analyze, and visualize localized bushfire risks. 

### 🚀 Live Demo
You can explore the interactive mapping tool directly in your web browser:
👉 **[Launch the Interactive Map](https://open-or-close-duazna7xn8daqhfjghkkbn.streamlit.app/)**






---

## 🚀 Execution & Replication Instructions

To run this full infrastructure pipeline locally, execute the following commands within your terminal or PyCharm command terminal:

### 1. Environment Configuration
```bash
# Clone the repository framework
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# Install standard spatial and predictive dependencies
pip install -r requirements.txt

### 2. Run Data Processing & Machine Learning
Triggers the data generation framework, calculates GIS spatial metrics, trains the Random Forest classifier, and packages the artifacts.

In the command prompt, Type:
python spatial_pipeline.py


### 3. Initialize the Interactive GIS Interface
Launches a localized, production-ready web layout container and spins up your internet browser automatically.

In the command prompt,Type:
streamlit run app.py

📊 Expected Outcomes & Project Deliverables
Upon script completion, your file system generates the following project artifacts:

1. processed_spatial_data.geojson: A geospatial file housing all coordinate points and calculation columns, suitable for importing directly into desktop GIS software like ArcGIS or QGIS.

2. bushfire_model.pkl: The frozen, trained model architecture ready for further production deployment.

3. An interactive web portal to let stakeholders simulate how rising temperatures impact fire hazard boundaries in real time.





