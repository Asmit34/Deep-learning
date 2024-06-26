from dotenv import load_dotenv

load_dotenv() # load all the environment variable from .env
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

genai.configure(api_key =os.getenv("GOOGLE_API_KEY"))

#function to load Gemini pro vision
model = genai.GenerativeModel("gemini-pro-vision")

def get_gemini_response(input, image, prompt):
    response = model.generate_content([input,image[0],prompt])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        #Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
            'mime_type':uploaded_file.type, # get the mime type of the uploaded file
            "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
## initiliaze the streamlit app
st.set_page_config(page_title="Multilanguage Invoice Extractor")
st.header("Multilanguage Invoice Extractor")
input = st.text_input("Input Propt", key = "input")
uploaded_file = st.file_uploader("Choose an image of invoice..", type = ["jpg", "jpge", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded Image.", use_column_width= True)

submit_button = st.button("Tell me about the image")

input_prompt = """
You are an expert in understanding invoice. We will upload an inage as invoice 
and you will have to answer based on the uploaded invoice image 
"""

# if submit button is clicked
if submit_button:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)
    st.subheader("The response is")
    st.write(response)