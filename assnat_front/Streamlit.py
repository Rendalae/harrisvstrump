import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import os

# URL of the API that returns the political group prediction
#API_URL = "http://127.0.0.1:8000/predictproba"
if not os.environ.get("API_URL"):
    raise ValueError("API_URL environment variable is not set")
API_URL = os.environ.get("API_URL")
TOKEN_GIF= os.environ.get("TOKEN_GIF")

print(f"API_URL: {API_URL}")
print(f"TOKEN_GIF: {TOKEN_GIF}")

# Function to query the API and get a prediction
def get_political_group_prediction(params):
    print(params)
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return {}


# Function to display a GIF image
def display_gif(predicted_class):
    gif_key = TOKEN_GIF
    gif_topic = predicted_class
    if gif_topic:
        response = requests.get(f'https://api.giphy.com/v1/gifs/random?api_key={gif_key}&tag={gif_topic}&rating=g')
        #
        if response.status_code == 200:
            data = response.json()
            gif_url = data['data']['images']['original']['url']
            st.image(gif_url, caption=f"Hello little {gif_topic}! (it will not lower your taxi fare, but might make you smile)")
        else:
            st.error("Failed to fetch GIF. Please try again.")
# Streamlit interface
st.title("Political Group Prediction")
st.write("Enter text and theme to get a political group prediction.")

# User input fields
user_input_text = st.text_area("Enter your text here:")
user_input_theme = st.text_input("Enter the theme here:")


if st.button("Predict"):
    params = {
    "Texte": user_input_text,
    "Theme": user_input_theme
}
    if params:
        predictions = get_political_group_prediction(params)
        if predictions:
            # Get the predicted class with the highest probability
            predicted_class = max(predictions, key=predictions.get)
            st.write(f"The predicted political group is: **{predicted_class}**")

            # Display a GIF (you need to have a GIF stored locally or provide a URL)
            display_gif(predicted_class)
    else:
        st.write("Please enter some text.")
