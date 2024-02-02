import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "dogVsCatTL_model.h5"

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    raise e
# Function to make a single prediction
def make_prediction(image_path):
    test_image = image.load_img(image_path, target_size=(224, 224, 3))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    # Assuming result contains the model's prediction
    if result[0] < 0:
        prediction = "cat"
    else:
        prediction = "dog"

    # Assuming test_image is a NumPy array with shape (1, 224, 224, 3)
    testImage = test_image.squeeze()

    return testImage, prediction

# Streamlit app
st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:

    # Make prediction
    testImage, prediction = make_prediction(uploaded_file)

    # Display the image with the prediction
    st.image(testImage, caption=f"Predicted: {prediction}", use_column_width=True)
