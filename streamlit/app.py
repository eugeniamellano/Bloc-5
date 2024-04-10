import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie
import numpy as np
from PIL import Image
import requests
import json
import openpyxl
from pydantic import BaseModel
import streamlit as st
import mlflow.pyfunc
import joblib
### Configuration
st.set_page_config(
    page_title="Getaround Project Dashboard",
    page_icon="🚗🚙🚕",
    layout="wide"
)
st.title("GETAROUND DASHBOARD 🚗")
st.markdown('')
st_lottie("https://lottie.host/822aa838-56f3-4354-bf16-f73921f315ed/rCWYXbiaXf.json")
# Data URL
DATA_URL = 'get_around_pricing_project.csv'

### App


st.subheader("""
    Welcome to the Getaround dashboard. This dashboard provides a visualization of Getaround data.
    You can explore Delay Analysis or Pricing Analysis, and use the rental value predictor.
""")

# Data loading
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

df_pricing = load_data()

# Data loading
@st.cache_data
def load_data_delay():
    data = pd.read_excel('get_around_delay_analysis.xlsx')
    return data

df_delay = load_data_delay()

st.subheader("DELAY ANALYSIS")
st.markdown('')
# Displaying the first few rows of data
st.subheader("🔸Data Visualization")
st.write(df_delay.head())


# types of check in
percentage_checkin_type = (df_delay['checkin_type'].value_counts() / df_delay['checkin_type'].count()) * 100

# grph
fig1 = px.pie(names=percentage_checkin_type.index, values=percentage_checkin_type.values,
            title='Percentage of Check-in Types', color_discrete_sequence=px.colors.sequential.Blues)
fig1.update_traces(marker=dict(colors=['#004c6d', '#002640', '#003959', '#001b2e']))
fig1.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig1)

# Layout de Streamlit
col1, col2 = st.columns(2)

with col1:
    percentage_state = (df_delay['state'].value_counts() / df_delay['state'].count()) * 100
    
    # graph pie state types
    fig2 = px.pie(names=percentage_state.index, values=percentage_state.values,
                  title='Percentage of State Types', color_discrete_sequence=px.colors.sequential.Mint)
    fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=1))
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    
    fig2.update_traces(textfont_size=12)

    st.plotly_chart(fig2)

with col2:
    cancelled_mobile_count = df_delay[(df_delay['checkin_type'] == 'mobile') & (df_delay['state'] == 'canceled')].shape[0]
    cancelled_connect_count = df_delay[(df_delay['checkin_type'] == 'connect') & (df_delay['state'] == 'canceled')].shape[0]
    total_cancelled_count = df_delay[df_delay['state'] == 'canceled'].shape[0]
    percentage_cancelled_mobile = (cancelled_mobile_count / total_cancelled_count) * 100
    percentage_cancelled_connect = (cancelled_connect_count / total_cancelled_count) * 100
    
    # graph pie cancelled per type
    fig3 = px.pie(names=['Mobile', 'Connect'], values=[percentage_cancelled_mobile, percentage_cancelled_connect],
                  title='Percentage of Cancelled Check-ins by Type', color_discrete_sequence=px.colors.sequential.Blues)
    fig3.update_traces(marker=dict(colors=['#004c6d', '#002640', '#003959', '#001b2e']))  # Cambiar a azules más fuertes
    fig3.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=1))
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    fig2.update_traces(textfont_size=12)

    st.plotly_chart(fig3)


st.subheader("PRICING ANALYSIS")
st.subheader("🔹 Data Visualization")
st.markdown('')
# Displaying the first few rows of data
df_pricing.drop(columns=['Unnamed: 0'],inplace=True)
st.write(df_pricing.head())

# Rental price distribution per day visualization
st.subheader("🔹 Rental Price Distribution per Day")
fig_price = px.histogram(df_pricing, x="rental_price_per_day", title="Rental Price Distribution per Day")
st.plotly_chart(fig_price)

# Visualization of categorical feature distributions
st.subheader("🔹 Features Distribution")
features = ['model_key', 'fuel', 'paint_color', 'car_type']
for feature in features:
    fig = px.bar(df_pricing, x=feature, title=f'{feature} Distribution',color_discrete_sequence=px.colors.sequential.Blues_r,color=feature)
    st.plotly_chart(fig)

# Layout de Streamlit
col1, col2 = st.columns(2)

with col1:
    # Primer gráfico: 'mileage'
    fig = px.scatter(df_pricing, x='mileage', y='rental_price_per_day', title='Rental Price per Day vs Mileage', color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(xaxis_title='Mileage', yaxis_title='Rental Price per Day')
    st.plotly_chart(fig)

with col2:
    # Segundo gráfico: 'engine_power'
    fig = px.scatter(df_pricing, x='engine_power', y='rental_price_per_day', title='Rental Price per Day vs Engine Power', color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(xaxis_title='Engine Power', yaxis_title='Rental Price per Day')
    st.plotly_chart(fig)


# Basic statistics
st.subheader("Basic Statistics of the Dataset")
st.write(df_pricing.describe())




# Definir la clase para los parámetros de predicción
class PredictionFeatures(BaseModel):
    model_key: str
    mileage: int
    engine_power: int
    private_parking_available: bool
    has_gps: bool
    fuel: str
    paint_color: str
    car_type: str
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

# Función para realizar la predicción usando el modelo cargado
def make_prediction(model, preprocessor, features):
    # Convertir los parámetros de predicción a un DataFrame
    df = pd.DataFrame([features.dict()])
    # Preprocesar las características usando el preprocesador cargado
    df = preprocessor.transform(df)
    # Hacer la predicción usando el modelo cargado
    prediction = model.predict(df)
    return prediction[0]

# Cargar el modelo desde un archivo .joblib
loaded_model = joblib.load('lin_reg.joblib')
loaded_preprocessor = joblib.load('preprocessor.joblib')

# Definir la aplicación de Streamlit
def main():
    st.title('Car Value Prediction')

    # Componentes de interfaz de usuario para ingresar los parámetros de predicción
    features = PredictionFeatures()
    features.model_key = st.text_input("Model Key")
    features.mileage = st.slider("Mileage", min_value=0, max_value=500000, value=0, step=1000)
    features.engine_power = st.slider("Engine Power", min_value=50, max_value=1000, value=200, step=10)
    features.private_parking_available = st.checkbox("Private Parking Available")
    features.has_gps = st.checkbox("Has GPS")
    features.fuel = st.selectbox("Fuel Type", ["petrol", "hybrid_petrol", "electro"])
    features.paint_color = st.selectbox("Paint Color", ["black", "grey", "white", "red", "silver", "blue", "orange", "beige", "brown", "green"])
    features.car_type = st.selectbox("Car Type", ["convertible", "coupe", "estate", "hatchback", "sedan", "subcompact", "suv", "van"])
    features.has_air_conditioning = st.checkbox("Has Air Conditioning")
    features.automatic_car = st.checkbox("Automatic Car")
    features.has_getaround_connect = st.checkbox("Has Getaround Connect")
    features.has_speed_regulator = st.checkbox("Has Speed Regulator")
    features.winter_tires = st.checkbox("Has Winter Tires")

    # Botón de predicción
    if st.button("Predict"):
        # Realizar la predicción
        prediction = make_prediction(loaded_model, loaded_preprocessor, features)
        # Mostrar el resultado de la predicción
        st.success(f"Predicted Car Value: ${prediction:.2f}")



# Streamlit
st.title('CAR RENTAL VALUE PREDICTION')
st.markdown('Enter the car features below to get the predicted price')

with st.form('Enter the car features'):
    model_key = st.selectbox("Model Key", ['Citroën', 'Peugeot', 'PGO', 'Renault', 'Audi', 'BMW', 'Ford', 'Mercedes', 'Opel', 'Porsche', 'Volkswagen', 'KIA Motors', 'Alfa Romeo', 'Ferrari', 'Fiat', 'Lamborghini', 'Maserati', 'Lexus', 'Honda', 'Mazda', 'Mini', 'Mitsubishi', 'Nissan', 'SEAT', 'Subaru', 'Suzuki', 'Toyota', 'Yamaha'])
    mileage = st.slider("Mileage", min_value=0, max_value=500000, value=0, step=1000)
    engine_power = st.slider("Engine Power", min_value=50, max_value=1000, value=200, step=10)
    private_parking_available = st.checkbox("Private Parking Available")
    has_gps = st.checkbox("Has GPS")
    fuel = st.selectbox("Fuel Type", ["petrol", "hybrid_petrol", "electro"])
    paint_color = st.selectbox("Paint Color", ["black", "grey", "white", "red", "silver", "blue", "orange", "beige", "brown", "green"])
    car_type = st.selectbox("Car Type", ["convertible", "coupe", "estate", "hatchback", "sedan", "subcompact", "suv", "van"])
    has_air_conditioning = st.checkbox("Has Air Conditioning")
    automatic_car = st.checkbox("Automatic Car")
    has_getaround_connect = st.checkbox("Has Getaround Connect")
    has_speed_regulator = st.checkbox("Has Speed Regulator")
    winter_tires = st.checkbox("Has Winter Tires")
    submit = st.form_submit_button("Predict")

    # En la parte de abajo, donde se realiza la predicción
    if submit:
        # Getting values entered by user
        features = PredictionFeatures(
            model_key=model_key,
            mileage=mileage,
            engine_power=engine_power,
            private_parking_available=private_parking_available,
            has_gps=has_gps,
            fuel=fuel,
            paint_color=paint_color,
            car_type=car_type,
            has_air_conditioning=has_air_conditioning,
            automatic_car=automatic_car,
            has_getaround_connect=has_getaround_connect,
            has_speed_regulator=has_speed_regulator,
            winter_tires=winter_tires
        )
        # Prediction
        prediction = make_prediction(loaded_model, loaded_preprocessor, features)
        # Display the result
        st.success(f"Predicted Car Value: ${prediction:.2f}")
### Footer 
st.sidebar.header("Getaround Dashboard")
st.sidebar.markdown("""
    * [DELAY ANALYSIS](#delay-analysis)
    * [PRICING ANALYSIS](#pricing-analysis)
    * [CAR RENTAL VALUE PREDICTION](#car-rental-value-prediction)
""")
st.sidebar.markdown("Created by Eugenia M")