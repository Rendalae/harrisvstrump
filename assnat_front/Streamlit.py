import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import os
import random

# URL of the API that returns the political group prediction
#API_URL = "http://127.0.0.1:8000/predictproba"
if not os.environ.get("API_URL"):
    raise ValueError("API_URL environment variable is not set")
API_URL = os.environ.get("API_URL")
TOKEN_GIF= os.environ.get("TOKEN_GIF")

mapping_political_people = {
    "Centre": ["Macron", "Bayrou", "Gabriel Attal", "Jean Lassalle"],
    "Droite": ["Nicolas Sarkozy", "François Fillon", "Eric Ciotti", "Marine le Pen"],
    "Gauche": ["François Hollande", "Melenchon", "Arnaud Montebourg"]
}

img_background_url="assnat_front/Img_assnat.png"
st.image(img_background_url, caption=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


print(f"API_URL: {API_URL}")

# Function to query the API and get a prediction
def get_political_group_prediction(params):
    print(params)
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return {}

def plot_probability_chart(proba_dict):
    groups = list(proba_dict.keys())
    probabilities = list(proba_dict.values())

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(groups, probabilities, color=['orange', 'blue', 'red'])

    ax.set_xlim(0, 1)  # Probability is between 0 and 1
    ax.set_xlabel('Probability')
    ax.set_title('Predicted Probabilities by Group')

    for i in range(len(groups)):
        ax.text(probabilities[i] + 0.01, i, f'{probabilities[i]:.2f}', va='center')

    st.pyplot(fig)

def validate_input(input_text):
    if len(input_text.split()) <= 5:
        return False, "Input must be at least 6 words long."
    return True, ""

# Function to display a GIF image
def display_gif(predicted_class):
    gif_key = TOKEN_GIF
    gif_topic = random.choice(mapping_political_people.get(predicted_class))
    if gif_topic:
        response = requests.get(f'https://api.giphy.com/v1/gifs/random?api_key={gif_key}&tag={gif_topic}&rating=g')
        #
        if response.status_code == 200:
            data = response.json()
            gif_url = data['data']['images']['original']['url']
            st.image(gif_url, caption=f"credit giphy - random image of '{gif_topic}'")
        else:
            st.error("Failed to fetch GIF. Please try again.")
# Streamlit interface


st.title("French Political 'Who said what'")
st.write("Enter a sentence along with its theme.\nYou will receive a prediction of which political group in the French National Assembly might say it.")

# User input fields
user_input_text = st.text_area("Sentence")
user_input_theme = st.text_input("Theme")


# Step 2: Use st.text_input to get the user input
is_valid, error_message = validate_input(user_input_text)
if user_input_text and not is_valid:
    st.error(error_message)

params = {
    "Texte": user_input_text,
    "Theme": user_input_theme
}

if st.button("Predict"):
    if is_valid:
        if params:
            predictions = get_political_group_prediction(params)
            if predictions:
                # Get the predicted class with the highest probability
                predicted_class = max(predictions, key=predictions.get)
                st.write(f"The predicted political group is: **{predicted_class}**")

                # Display a GIF (you need to have a GIF stored locally or provide a URL)
                display_gif(predicted_class)

                # Display the bar chart in the Streamlit app
                plot_probability_chart(predictions)
        else:
            st.write("Please enter some text.")
