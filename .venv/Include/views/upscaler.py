import streamlit as st
from PIL import Image
from imageupscaler.utilities import load_model, upscale_image
import os
import matplotlib.pyplot as plt
import numpy as np

st.title("Image Upscaler")

# Add a dropdown for the user to select the model
model_option = st.selectbox("Select Model", ["EDSR", "EDSRplus", "MDSR", "MDSRplus"])

# Update the model path based on the selected model
model_path = os.path.join(os.path.dirname(__file__), '..', 'imageupscaler', f'{model_option.lower()}_model.keras')
model = load_model(model_path)

# Add a slider for the user to select the scale factor
scale_factor = st.slider("Select Scale Factor", min_value=2, max_value=4, value=2, step=1)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Upscaling...")

    high_res = upscale_image(model, uploaded_file, scale=scale_factor)

    # Convert images to numpy arrays for plotting
    low_res_array = np.array(img.resize((img.width // scale_factor, img.height // scale_factor), Image.BICUBIC))
    high_res_array = np.array(high_res)

    # Plotting results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(low_res_array)
    axes[0].set_title("Low Resolution")
    axes[0].axis("off")
    axes[1].imshow(high_res_array)
    axes[1].set_title("Super Resolution")
    axes[1].axis("off")
    st.pyplot(fig)
