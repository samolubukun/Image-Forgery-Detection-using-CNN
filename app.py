import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from tensorflow.keras.models import load_model

# Function to convert an image to ELA format
def convert_to_ela_image(image, quality=90):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    max_diff = max_diff if max_diff != 0 else 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

# Function to prepare an image for forgery detection
def prepare_image_for_forgery(image):
    ela_image = convert_to_ela_image(image, 90).resize((128, 128))
    return np.array(ela_image).flatten() / 255.0

# Function to load the image forgery detection model
@st.cache_resource
def load_image_forgery_model():
    return load_model("imageforgerydetection.h5")

# Streamlit app
st.title("Image Forgery Detection")

# Task: Image Forgery Detection
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare the image for prediction
    prepared_image = prepare_image_for_forgery(image).reshape(-1, 128, 128, 3)

    # Load the model and make a prediction
    model = load_image_forgery_model()
    prediction = model.predict(prepared_image)
    confidence_real = prediction[0][1] * 100
    confidence_fake = prediction[0][0] * 100

    # Display the result
    if confidence_real > confidence_fake:
        st.success(f"Result: Real Image with {confidence_real:.2f}% confidence")
    else:
        st.error(f"Result: Forged Image with {confidence_fake:.2f}% confidence")
