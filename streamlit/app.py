import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your pre-trained image classification model for CIFAR-100
model = tf.keras.models.load_model('cifar100.h5')  # Replace with your model file

# CIFAR-100 class labels
class_labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
    "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

def main():
    st.title("CIFAR-100 Image Classification")

    # Upload image through Streamlit widget
    uploaded_image = st.file_uploader("Upload a CIFAR-100 image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for model prediction
        image = np.array(image)
        image = tf.image.resize(image, (32, 32))  # Resize to match CIFAR-100 image size
        image = image / 255.0  # Normalize

        # Make predictions
        predictions = model.predict(np.expand_dims(image, axis=0))

        # Display class probabilities
        st.write("Class Probabilities:")
        for i, prob in enumerate(predictions[0]):
            st.write(f"{class_labels[i]}: {prob:.2f}")

        # Get the predicted class
        predicted_class = np.argmax(predictions)
        st.write(f"Predicted Class: {class_labels[predicted_class]}")

if __name__ == "__main__":
    main()
