###  to run use --> streamlit run app.py
import streamlit as st # used for Creating a dashboard to visualize and display predictions
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub
from tensorflow.keras.models import load_model
import joblib  # For saving encoders


df = pd.read_csv('data\data.csv')

# Load the trained model
model = load_model('models\GRU-model.h5')  # Load the trained model

# Load the saved encoders and scalers
country_encoder = joblib.load('other\country_encoder.pkl')
city_encoder = joblib.load('other\city_encoder.pkl')
season_encoder = joblib.load('other\season_encoder.pkl')

scaler = joblib.load('other\scaler.pkl')


# Function to preprocess input data
def preprocess_input(country_encoded, city_encoded, season_encoded, month, day, year, air_pressure, humidity):
    # Encode categorical variables using the saved encoders
    try:
        country_encoded = country_encoder.transform([country])[0]
    except ValueError:
        st.error(f"Country '{country}' not found in the encoder. Please select a valid country.")
        return None

    try:
        city_encoded = city_encoder.transform([city])[0]
    except ValueError:
        st.error(f"City '{city}' not found in the encoder. Please select a valid city.")
        return None
    
    try:
        season_encoded = season_encoder.transform([season_encoded])[0]
    except ValueError:
        st.error(f"City '{city}' not found in the encoder. Please select a valid city.")
        return None

    # Create input array
    input_data = np.array([[country_encoded, city_encoded, season_encoded, month, day, year, air_pressure, humidity]])
    
    # Normalize only air_pressure and humidity
    input_data[:, [6, 7]] = scaler.transform(input_data[:, [6, 7]])  # Columns 6 and 7 are air_pressure and humidity
    
    # Reshape for model input
    input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    return input_data

# Streamlit app
st.title('Climate Change Analysis Dashboard')
st.write('Visualize and predict temperature trends using deep learning models.')

# Sidebar for user input
st.sidebar.header('User Input')
country = st.sidebar.selectbox('Country', ['Canada', 'Japan', 'Germany', 'France'])
city = st.sidebar.selectbox('City', ['Calgary', 'Montreal', 'Tokyo', 'Paris', 'Munich', 'Hamburg'])
season = st.sidebar.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Autumn'])  
month = st.sidebar.slider('Month', 1, 12, 1)
day = st.sidebar.slider('Day', 1, 31, 1)
year = st.sidebar.slider('Year', 1995, 2023, 2020)
air_pressure = st.sidebar.slider('Air Pressure (hPa)', 1000, 1020, 1010)
humidity = st.sidebar.slider('Humidity (g/m³)', 5, 25, 15)

# Preprocess input and make prediction
input_data = preprocess_input(country, city, season, month, day, year, air_pressure, humidity)
prediction = model.predict(input_data)[0][0]

# Display prediction
st.subheader('Predicted Temperature')
st.write(f'The predicted temperature for {city}, {country} on {month}/{day}/{year} is **{prediction:.2f}°C**.')


# Visualize historical data and correlation heatmap
historical_data = df[['Year', 'Temperature']]


fig, ax = plt.subplots()
sns.lineplot(data=historical_data, x='Year', y='Temperature', marker='o', ax=ax)
ax.set_title('Temperature Trends Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (°C)')
st.pyplot(fig)

# Correlation heatmap
st.subheader('Correlation Heatmap')
corr_data = df[['Temperature', 'Air Pressure', 'Humidity']]

fig, ax = plt.subplots()
sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)