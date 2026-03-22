import streamlit as st
import joblib
import numpy as np
import pandas as pd

# load model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")
st.image(
    "https://images.unsplash.com/photo-1568605114967-8130f3a36994",
    caption="Dream House Price Predictor 🏡",
    
)
st.markdown("### 🔍 Predict your house price instantly using ML")
st.subheader("Enter House Details")

# SIDEBAR INPUTS (professional look)
st.sidebar.header("Input Features")

overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
total_sf = st.sidebar.number_input("Total Square Feet", 500, 5000, 1500)
bathrooms = st.sidebar.number_input("Total Bathrooms", 1.0, 5.0, 2.0)
garage_cars = st.sidebar.number_input("Garage Cars", 0, 5, 1)
year_built = st.sidebar.number_input("Year Built", 1900, 2025, 2000)

garage_area = st.sidebar.number_input("Garage Area", 0, 1500, 500)
lot_area = st.sidebar.number_input("Lot Area", 500, 20000, 5000)
overall_cond = st.sidebar.slider("Overall Condition", 1, 10, 5)
house_age = st.sidebar.number_input("House Age", 0, 100, 10)
gr_liv_area = st.sidebar.number_input("Living Area", 500, 5000, 1500)

# predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'Overall Qual': [overall_qual],
        'TotalSF': [total_sf],
        'TotalBathrooms': [bathrooms],
        'Garage Cars': [garage_cars],
        'Year Built': [year_built],
        'Garage Area': [garage_area],
        'Lot Area': [lot_area],
        'Overall Cond': [overall_cond],
        'HouseAge': [house_age],
        'Gr Liv Area': [gr_liv_area]
    })

    prediction = model.predict(input_data)
    final_price = np.expm1(prediction[0])

    st.success(f"🏡 Estimated Price: ₹ {int(final_price):,}")
    st.write("Model Accuracy (R²): 0.87")
