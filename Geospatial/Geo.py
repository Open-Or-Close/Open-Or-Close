

import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rasterio.plot import show
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from shapely.geometry import Point


# Load Bushfire History


url = (
    "https://services-ap1.arcgis.com/ypkPEy1AmwPKGNNv/"
    "arcgis/rest/services/Bushfire_Boundaries_Historic_Dec_view/"
    "FeatureServer/0/query?"
    "where=1%3D1&"
    "outFields=*&"
    "f=geojson"
)

fire = gpd.read_file(url)

print(fire.head())

fire.plot(figsize=(10,8), color="red")
plt.title("Historical Bushfires")
plt.show()

# Load Raster Layers

temperature = rasterio.open(r"C:\Users\Eshgh\Desktop\My Business\Data Science\Practical projects\3\wc2.1_10m_tavg\wc2.1_10m_tavg_01.tif")
rainfall = rasterio.open(r"C:\Users\Eshgh\Desktop\My Business\Data Science\Practical projects\3\wc2.1_10m_prec\wc2.1_10m_prec_01.tif")
vegetation = rasterio.open(r"C:\Users\Eshgh\Desktop\My Business\Data Science\Practical projects\3\geom.png")
elevation = rasterio.open(r"C:\Users\Eshgh\Desktop\My Business\Data Science\Practical projects\3\2072469.tif")

# Read raster values

temp = temperature.read(1)
rain = rainfall.read(1)
veg = vegetation.read(1)
elev = elevation.read(1)

# Visualise Temperature

plt.figure(figsize=(10,8))
show(temp, cmap='hot')
plt.title("Temperature")
#plt.colorbar()
plt.show()

# Build Predictor Dataset

X = pd.DataFrame({
    "temperature": temp.flatten(),
    "rainfall": rain.flatten(),
    "vegetation": veg.flatten(),
    "elevation": elev.flatten()
})

y = np.random.randint(0,2,len(X))


# Remove Missing Values

X.replace(-9999,np.nan,inplace=True)

mask = X.notna().all(axis=1)

X = X[mask]
y = y[mask]



#Train/Test Split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)


# Accuracy
pred = model.predict(X_test)

print(classification_report(y_test,pred))

#Feature Importance

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})

importance.sort_values(
    by="Importance",
    ascending=False,
    inplace=True
)

print(importance)


# Predict Bushfire Risk
risk = model.predict_proba(X)[:,1]

# Convert Prediction Back to Raster
risk_map = np.full(temp.shape,np.nan)

risk_map.flat[mask] = risk


#Visualise Bushfire Risk

plt.figure(figsize=(12,8))

plt.imshow(risk_map,cmap="RdYlGn_r")

plt.title("Bushfire Risk Prediction")

plt.colorbar(label="Risk")

plt.show()


#Save Raster

with rasterio.open(
    "outputs/bushfire_risk.tif",
    "w",
    driver="GTiff",
    height=risk_map.shape[0],
    width=risk_map.shape[1],
    count=1,
    dtype=risk_map.dtype,
    crs=temperature.crs,
    transform=temperature.transform,
) as dst:
    dst.write(risk_map,1)


# Interactive Web Map

import folium

m = folium.Map(
    location=[-25,133],
    zoom_start=4
)

for idx,row in fire.iterrows():

    folium.CircleMarker(
        location=[row.geometry.y,row.geometry.x],
        radius=3,
        color="red",
        fill=True
    ).add_to(m)

m.save("outputs/fire_map.html")


