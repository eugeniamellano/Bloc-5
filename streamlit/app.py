import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie
import numpy as np
from PIL import Image
import requests
import json
import openpyxl
### Configuration
st.set_page_config(
    page_title="Getaround Project Dashboard",
    page_icon="🚗🚙🚕",
    layout="wide"
)
st.title("Getaround Dashboard 🚗")
st.markdown('')
st_lottie("https://lottie.host/822aa838-56f3-4354-bf16-f73921f315ed/rCWYXbiaXf.json")
# Data URL
DATA_URL = 'get_around_pricing_project.csv'

### App


st.subheader("""
    Welcome to the Getaround dashboard. This dashboard provides a visualization of Getaround data.
    You can explore Delay Analysis or Pricing Analysis.
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

### Footer 
st.sidebar.header("Getaround Dashboard")
st.sidebar.markdown("""
    * [DELAY ANALYSIS](#delay-analysis)
    * [PRICING ANALYSIS](#pricing-analysis)
""")
st.sidebar.markdown("Created by Eugenia M")