import streamlit as st
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model

# Load the trained model
model = load_model('model\cifar10_model.h5')  # Replace with your model file path

# Define class labels
class_labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# Create a Streamlit app
st.title("CIFAR-10 Image Classification App")

# Upload an image
st.write("Upload an image for classification:")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and make predictions
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (32, 32), interpolation=cv2.INTER_CUBIC)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_num = np.argmax(prediction)

    st.write("Prediction:")
    st.write(f"Class: {class_labels[class_num]}")
    st.write(f"Confidence: {prediction[0][class_num]:.2%}")
