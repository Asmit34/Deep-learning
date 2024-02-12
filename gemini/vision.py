from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

load_dotenv()  # Load environment variables from .env.

# Configure Gemini API key
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load OpenAI model and get responses
def get_gemini_response(input_text, image, user_info):
    model = genai.GenerativeModel('gemini-pro-vision')

    # Combine input_text, image, and user_info into a list
    input_data = [input_text, image, user_info]

    # Filter out empty values
    input_data = [item for item in input_data if item]

    # Generate content using the model
    response = model.generate_content(input_data)

    return response.text

# Initialize Streamlit app
st.set_page_config(page_title="Gemini Image Demo")

# Header
st.header("Gemini Application")

# Text input for input prompt
input_text = st.text_input("Input Prompt:", key="input_text")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""

# Display uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

# Form for collecting user information
with st.form("user_info_form"):
    st.subheader("Provide your information:")
    user_name = st.text_input("Your Name:", key="user_name")
    user_phone = st.text_input("Phone Number:", key="user_phone")
    user_email = st.text_input("Email:", key="user_email")

    # Checkbox for call request
    call_request = st.checkbox("I would like a call", key="call_request")

    # Submit button for the user information form
    submit_info_button = st.form_submit_button("Submit User Info")

# Button to trigger Gemini model
submit_button = st.button("Tell me about the image")

# If the user info form is submitted
if submit_info_button:
    # Acknowledge the request for a call and prompt for information
    if call_request:
        st.session_state['chat_history'].append(("You", "I would like a call"))
        st.session_state['chat_history'].append(("Bot", "Sure, please provide your information."))

    # Append valid user information to chat history with proper line breaks
    user_info_text = f"Name: {user_name}\nPhone: {user_phone}\nEmail: {user_email}"
    st.session_state['chat_history'].append(("User Info", user_info_text))

# If the Gemini model button is clicked
if submit_button:
    # Call the model and get the response
    response = get_gemini_response(input_text, image, user_info_text)
    
    # Append user query and response to chat history
    st.session_state['chat_history'].append(("You", input_text))
    st.session_state['chat_history'].append(("Bot", response))

# Display chat history
st.subheader("The Chat History is:")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
