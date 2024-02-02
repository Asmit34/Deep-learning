import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model_path = "model_checkpoint.h5"

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    raise e

# Constants
img_width, img_height = 150, 150
class_labels = {0: "Cat", 1: "Dog"}

def preprocess_image(image):
    img = image.resize((img_width, img_height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image_class(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class], prediction[0][predicted_class]

def main():
    st.title("Cat vs. Dog Image Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Make prediction
        class_name, confidence = predict_image_class(image)
        st.write(f"Class: {class_name}, Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main()
