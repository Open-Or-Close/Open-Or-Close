import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib


def generate_synthetic_spatial_data(n_samples=2000):
    print("🌍 Generating synthetic geospatial environment...")
    np.random.seed(42)

    # 1. Simulate Lat/Lon coordinates centered around a study area (e.g., California/Australia-like bounding box)
    lats = np.random.uniform(-34.0, -33.0, n_samples)
    lons = np.random.uniform(150.5, 151.5, n_samples)

    # 2. Environmental & Climate Features
    elevation = np.random.uniform(100, 1200, n_samples)  # meters
    slope = np.random.uniform(0, 35, n_samples)  # degrees
    ndvi = np.random.uniform(0.1, 0.8, n_samples)  # Vegetation index (0=dead, 1=lush)
    temp = np.random.uniform(25, 45, n_samples)  # Celsius
    wind_speed = np.random.uniform(5, 60, n_samples)  # km/h

    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(lons, lats)]
    gdf = gpd.GeoDataFrame({
        'latitude': lats,
        'longitude': lons,
        'elevation': elevation,
        'slope': slope,
        'ndvi': ndvi,
        'temperature': temp,
        'wind_speed': wind_speed
    }, geometry=geometry, crs="EPSG:4326")

    # 3. GIS Analytics: Simulate a "High-Risk Power Line" vector layer
    # Generating a line string passing through the center of the study area
    line_geom = LineString([(150.5, -33.5), (151.5, -33.5)])
    power_line = gpd.GeoSeries([line_geom], crs="EPSG:4326")

    # Calculate distance from each point to the power line (Spatial Analysis)
    # Projecting briefly to a metric system (UTM zone) for accurate distance calculation
    gdf_metric = gdf.to_crs(epsg=3857)
    line_metric = power_line.to_crs(epsg=3857).iloc[0]
    gdf['dist_to_powerline_m'] = gdf_metric.distance(line_metric)

    # 4. Generate Target Variable (Bushfire Occurrence Probability)
    # Formula creates a realistic dynamic: High temp, high wind, steep slope, proximity to powerline = Fire
    risk_score = (
            (temp * 0.4) +
            (wind_speed * 0.3) +
            (slope * 0.2) -
            (gdf['dist_to_powerline_m'] / 5000) -
            (ndvi * 10)
    )
    # Normalize and convert to binary target (1 = Fire, 0 = No Fire)
    prob = 1 / (1 + np.exp(-0.1 * (risk_score - 40)))
    gdf['fire_occurred'] = np.where(prob > 0.50, 1, 0)

    return gdf


def train_spatial_model(gdf):
    print("🤖 Processing data and training Machine Learning model...")
    # Features and Target split
    features = ['elevation', 'slope', 'ndvi', 'temperature', 'wind_speed', 'dist_to_powerline_m']
    X = gdf[features]
    y = gdf['fire_occurred']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    prob_preds = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, prob_preds):.4f}")

    # Save Model and Data for Dashboard
    joblib.dump(model, 'bushfire_model.pkl')
    gdf.to_file("processed_spatial_data.geojson", driver="GeoJSON")
    print("💾 Model and spatial layers saved successfully!")


if __name__ == "__main__":
    spatial_data = generate_synthetic_spatial_data()
    train_spatial_model(spatial_data)
