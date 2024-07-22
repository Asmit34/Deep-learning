import streamlit as st
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
import matplotlib.pyplot as plt

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(150, 150))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 150, 150, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

# load an image and predict the class
def run_example(image_path):
    # load the image
    img = load_image(image_path)
    # load model
    model = load_model('model\DogVSCat.h5')
    # predict the class
    result = model.predict(img)

    # Convert the predicted probability to a class label
    if result[0] > 0.5:
        label = 'Dog'
    else:
        label = 'Cat'

    # Convert the array back to image
    img = array_to_img(img[0])

    # Display the image and label using Streamlit
    st.image(img, caption=f'PREDICTED LABEL: {label}', use_column_width=True)

def main():
    st.title("Dog vs Cat Classifier")
    st.write("Upload an image for classification.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        # st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Run the classifier
        run_example(uploaded_file)
    

if __name__ == "__main__":
    main()
