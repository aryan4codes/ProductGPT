import streamlit as st
import cv2
import numpy as np

st.title("Image Processing App")

# Upload two image files
image1 = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"])
image2 = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"])

if image1 and image2:
    st.write("First Image:")
    st.image(image1, use_column_width=True)

    st.write("Second Image:")
    st.image(image2, use_column_width=True)

    # Process the images
    image1 = cv2.imdecode(np.frombuffer(image1.read(), np.uint8), -1)
    image2 = cv2.imdecode(np.frombuffer(image2.read(), np.uint8), -1)

    # Your image processing logic goes here
    # For demonstration, we'll just return the first image
    result_image = image1

    st.write("Processed Image:")
    st.image(result_image, use_column_width=True)
