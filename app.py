# Imports

import streamlit as st

from preprocessing.cleaning_data import Preprocessor
from predict.prediction import Predictor

# Code

preprocessing = Preprocessor()
preprocessing.load_data()

st.title("Property price prediction using KNN regression")
st.warning(
    f"Please only use this tool to predict property prices ranging from 200,000 € to 600,000 €"
)

postal_code = st.sidebar.selectbox(
    "Postal code of property location", preprocessing.postalcodes, index=None
)
property_type = st.sidebar.selectbox(
    "Select property type", preprocessing.property_type, index=None
)
building_state = st.sidebar.selectbox(
    "Select building state", preprocessing.building_state, index=None
)
num_facades = st.sidebar.selectbox(
    "Select number of facades", preprocessing.num_facades, index=None
)
num_rooms = st.sidebar.selectbox(
    "Select number of rooms", preprocessing.num_rooms, index=None
)
surface_land = st.sidebar.number_input(
    "Enter surface area of the plot of land in $m^2$", min_value=0.0
)
living_area = st.sidebar.number_input("Enter living area in $m^2$", min_value=0.0)

if st.button("Predict price"):
    with st.spinner("Calculating price..."):
        new_data = {
            "PostalCodes": [postal_code],
            "Subtype of property": [property_type],
            "State of the building": [building_state],
            "Number of facades": [num_facades],
            "Number of rooms": [num_rooms],
            "Surface area of the plot of land": [surface_land],
            "Living Area": [living_area],
        }

        clean_distances = preprocessing.preprocess(new_data=new_data)

        predictor = Predictor(data=clean_distances)
        predicted_price = predictor.predict()
        CI_lower, CI_upper = predictor.confidence_bootstrap(
            X_dist=preprocessing.X_distance,
            y=preprocessing.y_train,
            new_data_dist=preprocessing.distance,
        )
    st.success(f"The predicted property price is {predicted_price}€")
    st.warning(
        f"The 95% Confidence Interval for this price estimation is: ({CI_lower} - {CI_upper})"
    )
