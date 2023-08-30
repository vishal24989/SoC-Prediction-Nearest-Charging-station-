import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import folium_static
from geopy.distance import geodesic

# Load data
@st.cache
def load_data():
    soc_df = pd.read_csv('soc_dataset.csv')
    charging_stations_df = pd.read_csv('bangalore_charging_stations_dataset.csv')
    return soc_df, charging_stations_df

soc_df, charging_stations_df = load_data()

# Train SOC prediction model
@st.cache
def train_model(soc_df):
    X = soc_df[['Current (A)', 'Voltage (V)', 'Temperature (°C)', 'Battery Capacity (Ah)', 'Accessory Load (W)', 'Elevation (m)', 'Temperature Outside (°C)']]
    y = soc_df['SOC (%)']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model(soc_df)

# Function to suggest the nearest charging station
def suggest_charging_station(current_location, charging_stations_df):
    available_stations = charging_stations_df[charging_stations_df['Charging Station Status'] == 1]
    distances = available_stations['Charging Station Location'].apply(lambda x: geodesic(current_location, tuple(map(float, x.strip('()').split(',')))).km)
    nearest_station = available_stations.loc[distances.idxmin()]
    return nearest_station

# Application interface
st.title('SOC Prediction and Charging Station Finder')

st.write('Enter the input features for SOC prediction:')

current = st.number_input('Current (A)', value=0.0)
voltage = st.number_input('Voltage (V)', value=0.0)
temperature = st.number_input('Temperature (°C)', value=0.0)
battery_capacity = st.number_input('Battery Capacity (Ah)', value=0.0)
accessory_load = st.number_input('Accessory Load (W)', value=0.0)
elevation = st.number_input('Elevation (m)', value=0.0)
temperature_outside = st.number_input('Temperature Outside (°C)', value=0.0)

input_features = [[current, voltage, temperature, battery_capacity, accessory_load, elevation, temperature_outside]]

if st.button('Predict SOC'):
    predicted_soc = model.predict(input_features)
    st.write('Predicted SOC: {:.2f}%'.format(predicted_soc[0]))

st.write('Enter your current location to find the nearest charging station:')

latitude = st.number_input('Latitude', value=0.0)
longitude = st.number_input('Longitude', value=0.0)

current_location = (latitude, longitude)

if st.button('Find Nearest Charging Station'):
    nearest_station = suggest_charging_station(current_location, charging_stations_df)

    st.write('Nearest Charging Station:')
    st.write('Location: {}'.format(nearest_station['Charging Station Location']))
    st.write('Charging Speed: {:.2f} kW'.format(nearest_station['Charging Speed (kW)']))

    map_center = [latitude, longitude]
    m = folium.Map(location=map_center, zoom_start=14)
    folium.Marker(location=current_location, popup='You are here', icon=folium.Icon(color='blue')).add_to(m)
    folium.Marker(location=tuple(map(float, nearest_station['Charging Station Location'].strip('()').split(','))), 
                  popup='Nearest Charging Station<br>Speed: {:.2f} kW'.format(nearest_station['Charging Speed (kW)']),
                  icon=folium.Icon(color='green')).add_to(m)
    folium_static(m)
