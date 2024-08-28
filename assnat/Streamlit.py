import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from assnat.params import *

# URL of the API that returns the political group prediction
API_URL = "http://127.0.0.1:8000/predictproba"

#img_background_url="https://data.assemblee-nationale.fr/var/ezflow_site/storage/images/media/opendata/frontpage-image-2/263460-1-fre-FR/frontpage-image-2_small.jpg"
#img_background_url= "https://data.assemblee-nationale.fr/extension/anopendata/design/an_opendata/images/logo-gris.png"
img_background_url="assnat/Img_assnat.png"
# background image
st.image(img_background_url, caption=None, width=1000, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# Function to query the API and get a prediction
def get_political_group_prediction(params):
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return {}


# Function to display a GIF image
def display_gif(predicted_class):
    gif_key = "l6KgXnsKfRrQDGuydU3k75AYEzRWRAS6"
    gif_topic = f"homme ou femme politique de {predicted_class}"
    if gif_topic:
        response = requests.get(f'https://api.giphy.com/v1/gifs/random?api_key={gif_key}&tag={gif_topic}&rating=g')
        #
        if response.status_code == 200:
            data = response.json()
            gif_url = data['data']['images']['original']['url']
            st.image(gif_url, caption=f"It will not change your political lean but might make you smile")
        else:
            st.error("Failed to fetch GIF. Please try again.")

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


# Streamlit interface
st.title("Welcome to the French Political 'who say what' ")
st.write("Enter a sentence and the sentence theme to get a prediction on which political group from the French National Assembly would pronunce it.")

# User input fields
user_input_text = st.text_area("Enter your sentence here:")
user_input_theme = st.text_input("Enter the sentence theme here:")

params = {
    "Texte": user_input_text,
    "famille": user_input_theme
}

if st.button("Predict"):
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
