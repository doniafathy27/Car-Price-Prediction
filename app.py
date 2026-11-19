import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

@st.cache_resource
def load_artifacts():
    return joblib.load("car_price_model.joblib")

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
feature_columns = artifacts["feature_columns"]
num_cols = artifacts["num_cols"]
cat_cols = artifacts["cat_cols"]
num_medians = artifacts["num_medians"]
cat_modes = artifacts["cat_modes"]

st.title("🚗 Car Price Prediction")
st.write("Enter the car details below to estimate its price.")

brand_options = ["BMW", "Honda", "Hyundai", "Kia", "Mazda", "Mercedes", "Nissan", "Toyota"]
fuel_options = ["Diesel", "Electric", "Hybrid", "Petrol"]
transmission_options = ["Automatic", "Manual"]

with st.form("car_details"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", brand_options)
        year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2018, step=1)
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=60000, step=1000)
        engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    with col2:
        fuel_type = st.selectbox("Fuel Type", fuel_options)
        transmission = st.selectbox("Transmission", transmission_options)
        previous_owners = st.selectbox("Previous Owners", [1, 2, 3, 4, 5])
    submitted = st.form_submit_button("Predict Car Price", use_container_width=True)

if submitted:
    row = {
        "Brand": brand,
        "Year": year,
        "Mileage": mileage,
        "Engine_Size": engine_size,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Previous_Owners": previous_owners
    }
    input_df = pd.DataFrame([row])

    for col in num_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(num_medians[col])
    for col in cat_cols:
        input_df[col] = input_df[col].fillna(cat_modes[col]).astype(str)

    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=False)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    prediction = float(model.predict(input_scaled)[0])
    st.success(f"Estimated car price: {prediction:,.2f}")
    st.caption(f"Prediction generated using the {artifacts['model_name']} model.")
    st.info("This is an estimate based on patterns in the training dataset, not a guaranteed market price.")

st.divider()
st.caption("Car Price Prediction • Machine Learning Regression Project")
