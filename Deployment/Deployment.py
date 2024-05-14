import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import cv2
import requests
# from streamlit_lottie import st_lottie
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit.components.v1 as components
from streamlit_modal import Modal
from streamlit import components
from PIL import Image

classes_mapping = {0:'AK',1:'Ala Idris',2:'Buzgulu',3:'Dimnit',4:'Nazli'}

# Load Model
model = tf.keras.models.load_model(r"D:\Neural Networks assignments\Project\Local Run\Inception\InceptionV5_black.h5")

# Set title of page

st.title("  "* 1000 + "Grapevines Species Prediction" + ""*1000)
def preprocess_Inception(img):
    img = np.array(img)
    img = img[:,:,:3]
    generator1 = ImageDataGenerator(zoom_range=[0.6, 0.7])
    generator2 = ImageDataGenerator(zoom_range=[0.9, 1.0])
    white_pixels = (img[:, :, 0] > 250) & (img[:, :, 1] > 250) & (img[:, :, 2] > 250)
    percentage = np.sum(white_pixels) / (img.shape[0] * img.shape[1]) * 100
    img[white_pixels] = 0

    if percentage >= 80:
        img = generator1.random_transform(img)
    else:
        img = generator2.random_transform(img)

    img = cv2.resize(img, (480, 480))
    img = img / 255.0
    return img



def predict(img):
   prediction = model.predict(img)
   prediction = np.argmax(prediction, axis=1)
   return prediction[0]


img = None

uploaded_file = st.file_uploader("Upload Grapevine Image ", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    img = preprocess_Inception(image)
    img = np.expand_dims(img, axis=0)



    # Display the image
    st.markdown(
        f'<style>'
        f'.stImage > img {{display: block; margin-left: 100px; margin-right: 100px; max-width: {100}px; max-height: {100}px;}}'
        f'</style>',
        unsafe_allow_html=True
    )
    st.image(image, caption='Uploaded Image',width=375)

with st.sidebar:
    # Use CSS to center the button
    st.markdown(
        """
        <style>
        .stButton>button {
            margin: 0 auto;
            display: block;
            # width: 170px; /* Set the width of the button */
            # height: 50px; /* Set the height of the button */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


show_popup = st.button(label='Submit', disabled=(uploaded_file is None))
# Check if the button is clicked
if show_popup:
    # Call the predict function to obtain the predicted class
    pred = predict(img)
    predicted_class = classes_mapping[pred]

    # Create a modal with the predicted class as title
    modal = Modal(
        key="demo-modal",
        title=predicted_class,

        # Optional
        padding=60,    # default value
        max_width=500  # default value
    )
    # Define the content of the modal
    with modal.container():
        # Display the content of the modal
        st.write(" ")

        # Add a button at the bottom right to close the modal
        if st.button("OK", key="close-modal"):
            modal.toggle()