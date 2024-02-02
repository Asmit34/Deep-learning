import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the image captioning model and tokenizer
model = load_model("Models/image_captioning_model.h5")
tokenizer = pickle.load(open("Models/tokenizer.pkl", "rb"))

# Load the VGG16 model for image feature extraction
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

vgg_model = VGG16(weights='imagenet')
new_input = vgg_model.input
hidden_layer = vgg_model.layers[-2].output
vgg_model = Model(inputs=new_input, outputs=hidden_layer)

# Define max caption length (adjust based on your model)
max_caption_length = 35

# Streamlit app
st.title("Image Captioning App")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the uploaded image for the model
    img = Image.open(uploaded_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.resize((64, 64))  # Resize the image to (224, 224)
    img = np.array(img)
    img = preprocess_input(img)  # Apply VGG16 preprocessing
    img = np.expand_dims(img, axis=0)  # Add batch dimension


    # Generate a caption for the image
    def generate_caption(model, tokenizer, image, max_caption_length):
        # Initialize the caption with the start sequence
        caption = 'startseq'

        for _ in range(max_caption_length):
            # Tokenize the caption
            sequence = tokenizer.texts_to_sequences([caption])[0]
            sequence = pad_sequences([sequence], maxlen=max_caption_length)

            # Predict the next word
            yhat = model.predict([image, sequence], verbose=0)
            yhat = np.argmax(yhat)

            # Map the integer to a word
            word = tokenizer.index_word[yhat]

            # Append the word to the caption
            caption += ' ' + word

            # Break if we predict the end of the sequence
            if word == 'endseq':
                break

        return caption


    caption = generate_caption(model, tokenizer, img, max_caption_length)

    # Display the generated caption
    st.write("Generated Caption:")
    st.write(caption)
