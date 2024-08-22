import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle
pickle_in = open("flight-predict.pkl", "rb")
grid_search = pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_flight_prices(Airline, Source, Destination, Total_Stops, duration_hour, Routes, day, month, distance_mode, departure_mode):
    """
    Lets Predict flight prices in India
    ---
    parameters:
      - name: Airline
        in: query
        type: number
        required: true
      - name: Source
        in: query
        type: number
        required: true
      - name: Destination
        in: query
        type: number
        required: true
      - name: Total_Stops
        in: query
        type: number
        required: true
      - name: duration_hour
        in: query
        type: number
        required: true
      - name: Routes
        in: query
        type: number
        required: true
      - name: day
        in: query
        type: number
        required: true
      - name: month
        in: query
        type: number
        required: true
      - name: distance_mode
        in: query
        type: number
        required: true
      - name: departure_mode
        in: query
        type: number
        required: true
    responses:
      200:
        description: The output values
    """
    prediction = grid_search.predict([[Airline, Source, Destination, Total_Stops, duration_hour, Routes, day, month, distance_mode, departure_mode]])
    print(prediction)
    return prediction

def main():
    st.title("Flight predictor")
    html_temp = """
    <div style="background-color: tomato; padding: 10px">
        <h2 style="color: white; text-align: center;">Streamlit Flight predictor ML app</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Airline = st.text_input("Airline", "Type here")
    Source = st.text_input("Source", "Type here")
    Destination = st.text_input("destination", "Type here")
    Total_Stops = st.text_input("Total_Stops", "Type here")
    duration_hour = st.text_input("duration_hour", "Type here")
    Routes = st.text_input("Routes", "Type here")
    day = st.text_input("day", "Type here")
    month = st.text_input("month", "Type here")
    distance_mode = st.text_input("distance_mode", "Type here")
    departure_mode = st.text_input("departure_mode", "Type here")
    result = ""
    if st.button("Predict"):
        result = predict_flight_prices(Airline, Source, Destination, Total_Stops, duration_hour, Routes, day, month, distance_mode, departure_mode)
    st.success(f"The output is {result}")
    if st.button("About"):
        st.text("Built with Streamlit by Osinowo Abdulazeez")

if __name__ == '__main__':
    main()
