import osmnx as ox
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
from streamlit_folium import folium_static
from geopy.distance import geodesic

# Download OpenStreetMap data for Chennai
place_name = "Chennai, India"
tags = {'amenity': ['restaurant', 'hospital']}
graph = ox.graph_from_place(place_name, network_type='all')
nodes, edges = ox.graph_to_gdfs(graph)

# Extract restaurants and hospitals data
restaurants = nodes[nodes['amenity'] == 'restaurant'][['name', 'latitude', 'longitude']]
hospitals = nodes[nodes['amenity'] == 'hospital'][['name', 'latitude', 'longitude']]

# Add labels
restaurants["label"] = "restaurant"
hospitals["label"] = "hospital"
data = pd.concat([restaurants, hospitals], ignore_index=True)

# Machine Learning model for amenity prediction
X = data[["latitude", "longitude"]]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit app layout
st.title("Chennai Location Analysis Dashboard")

# Sidebar Filters
show_restaurants = st.sidebar.checkbox("Show Restaurants", True)
show_hospitals = st.sidebar.checkbox("Show Hospitals", True)
search_radius = st.sidebar.slider("Search Radius (km)", 1, 10, 2)

# Generate map
chennai_map = folium.Map(location=[13.0827, 80.2707], zoom_start=12)

# Function to find nearby amenities based on lat, lon
def find_nearby(df, lat, lon, radius_km):
    return df[df.apply(lambda row: geodesic((lat, lon), (row["latitude"], row["longitude"])).km <= radius_km, axis=1)]

# Display amenities based on filter options
if show_restaurants:
    heat_data = [[row["latitude"], row["longitude"]] for _, row in restaurants.iterrows()]
    HeatMap(heat_data, radius=10, blur=15).add_to(chennai_map)

if show_hospitals:
    marker_cluster = MarkerCluster()
    for _, row in hospitals.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=row["name"],
            icon=folium.Icon(color="blue")
        ).add_to(marker_cluster)
    chennai_map.add_child(marker_cluster)

# Display map
folium_static(chennai_map)

# Search Nearby Amenities
st.sidebar.header("Search Nearby Amenities")
lat = st.sidebar.number_input("Enter Latitude", value=13.0827)
lon = st.sidebar.number_input("Enter Longitude", value=80.2707)

if st.sidebar.button("Search"):
    nearby_restaurants = find_nearby(restaurants, lat, lon, search_radius)
    nearby_hospitals = find_nearby(hospitals, lat, lon, search_radius)
    
    st.write(f"Nearby restaurants within {search_radius} km:", nearby_restaurants)
    st.write(f"Nearby hospitals within {search_radius} km:", nearby_hospitals)

# Upload Custom GPS Data
st.sidebar.header("Upload GPS Data")
uploaded_file = st.sidebar.file_uploader("D:/SRM/Project/Location/Data.csv")

if uploaded_file:
    user_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", user_data)

    # Plot uploaded points on the map
    for _, row in user_data.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"Uploaded Point: ({row['latitude']}, {row['longitude']})",
            icon=folium.Icon(color="purple")
        ).add_to(chennai_map)

    folium_static(chennai_map)

# Amenity Prediction (using ML)
st.sidebar.header("Predict Amenity Type")
predict_lat = st.sidebar.number_input("Enter Latitude for Prediction", value=13.0827)
predict_lon = st.sidebar.number_input("Enter Longitude for Prediction", value=80.2707)

if st.sidebar.button("Predict"):
    prediction = model.predict([[predict_lat, predict_lon]])
    st.write(f"The predicted amenity at this location is: {prediction[0]}")
