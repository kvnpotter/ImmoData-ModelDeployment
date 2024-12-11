# Imports

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from preprocessing.cleaning_data import Preprocessor, Distance_calc

# Code

preprocessing = Preprocessor()
preprocessing.load_data()

st.title("Property price prediction using KNN regression")

postal_code = st.selectbox("Postal code of property location", preprocessing.postalcodes, index= None)
property_type = st.selectbox('Select property type', preprocessing.property_type, index= None)
building_state = st.selectbox('Select building state', preprocessing.building_state, index= None)
num_facades = st.selectbox('Select number of facades', preprocessing.num_facades, index= None)
num_rooms = st.selectbox('Select number of rooms', preprocessing.num_rooms, index= None)
surface_land = st.number_input('Enter surface area of the plot of land in $m^2$', min_value= 0.0)
living_area = st.number_input('Enter living area in $m^2$', min_value= 0.0)

new_data = {'PostalCodes': [postal_code],
            'Subtype of property': [property_type],
            'State of the building': [building_state],
            'Number of facades': [num_facades],
            'Number of rooms': [num_rooms],
            'Surface area of the plot of land': [surface_land],
            'Living Area': [living_area]}

clean_data = preprocessing.preprocess(new_data= new_data)

predictor = Predictor()

ApplicantIncome = st.sidebar.slider('ApplicantIncome', 0, 10000, 0)
LoanAmount = st.sidebar.slider('LoanAmount in K$', 9.0, 700.0, 200.0)

# Assuming additional input features here...    
# 
# # Prediction Logic    if st.button("Predict"):        loaded_model = pickle.load(open('Random_Forest.sav', 'rb'))        prediction = loaded_model.predict(np.array([ApplicantIncome, LoanAmount]).reshape(1, -1))        if prediction[0] == 0:            st.error('According to our calculations, you will not get the loan.')        else:            st.success('Congratulations! You will get the loan.')











st.write("Hello ,let's learn how to build a streamlit app together")

st.title("This is the app title")
st.header("This is the header")
st.markdown("This is the markdown")
st.subheader("This is the subheader")
st.caption("This is the caption")
st.code("x = 2021")
st.latex(r''' a+a r^1+a r^2+a r^3 ''')

st.image("kid.jpg", caption="A kid playing")

st.checkbox('Yes')
st.button('Click Me')
st.radio('Pick your gender', ['Male', 'Female'])
st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange'])
st.multiselect('Choose a planet', ['Jupiter', 'Mars', 'Neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0, 50)

with st.spinner('Wait for it...'):    
    time.sleep(10)

st.success("You did it!")
st.error("Error occurred")
st.warning("This is a warning")
st.info("It's easy to build a Streamlit app")
st.exception(RuntimeError("RuntimeError exception"))

st.sidebar.title("Sidebar Title")