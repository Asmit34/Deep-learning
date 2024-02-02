import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the predict_caption function
def predict_caption(model, image, tokenizer, max_length):
    # Add start tag for generation process
    in_text = 'startseq'
    # Iterate over the max length of the sequence
    for i in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # Predict the next word
        yhat = model.predict([image, sequence], verbose=0)
        # Get the index with high probability
        yhat = np.argmax(yhat)
        # Convert the index to word
        word = idx_to_word(yhat, tokenizer)
        # Stop if the word is not found
        if word is None:
            break
        # Append the word as input for generating the next word
        in_text += " " + word
        # Stop if we reach the end tag
        if word == 'endseq':
            break
    return in_text

# Load the model and tokenizer from the "Models" directory
model_file_path = os.path.join("Models", "image_captioning_model.h5")
tokenizer_file_path = os.path.join("Models", "tokenizer.pkl")

model = load_model(model_file_path)  # Load your trained model
tokenizer = Tokenizer()
tokenizer.word_index = pickle.load(open(tokenizer_file_path, 'rb'))  # Load the tokenizer used during training

st.title("Image Captioning App")

# File uploader for the user to upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, use_column_width=True)

    # Load and preprocess the uploaded image
    img = Image.open(uploaded_image)
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Generate a caption for the image
    max_length = 35  # Adjust this based on your model's max sequence length
    caption = predict_caption(model, img, tokenizer.word_index, max_length)

    # Display the predicted caption
    st.subheader("Predicted Caption:")
    st.write(caption)
