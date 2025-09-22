import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("gradientboostmodel1.pkl")
scaler = joblib.load("scaler1.pkl")


mappings = {
    "view": {'Average': 0, 'Excellent': 1, 'Fair': 2, 'Good': 3, 'No View': 4},
    "waterfront": {'No': 0, 'Yes': 1},
    "condition": {'Average': 0, 'Fair- Badly worn': 1, 'Good': 2, 'Poor- Worn out': 3, 'Very Good': 4}
}

st.title("Washington Housing Price Prediction")


col1, col2, col3 = st.columns(3)
with col1:
    bedrooms = st.number_input("Bedrooms", 0, 20, 3)
    bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0, 0.5)
    floors = st.number_input("Floors", 1.0, 5.0, 1.0)

with col2:
    sqft_living = st.number_input("Living Area (sqft)", 100, 20000, 1500)
    sqft_lot = st.number_input("Lot Size (sqft)", 100, 100000, 5000)
    grade = st.slider("Grade", 1, 13, 7)

with col3:
    zipcode = st.number_input("Zipcode", 10000, 99999, 98103)
    age = st.number_input("Age of the House (Years)", 0, 200, 30)


col4, col5, col6 = st.columns(3)
with col4:
    view_label = st.selectbox("View", list(mappings['view'].keys()))
with col5:
    waterfront_label = st.selectbox("Waterfront", list(mappings['waterfront'].keys()))
with col6:
    condition_label = st.selectbox("Condition", list(mappings['condition'].keys()))


view = mappings['view'][view_label]
waterfront = mappings['waterfront'][waterfront_label]
condition = mappings['condition'][condition_label]


if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'waterfront': [waterfront],
        'view': [view],
        'condition': [condition],
        'grade': [grade],
        'zipcode': [zipcode],
        'age': [age]
    })
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    pred_price = np.expm1(prediction)  
    st.success(f"Predicted House Price: ${pred_price[0]:,.2f}")
