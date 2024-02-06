import streamlit as st
import os
from PIL import Image

def load_images_from_folder(folder):
    """Load images from a specified folder"""
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path)
            images.append(img)
        except (IOError, FileNotFoundError):
            # You can print an error message or pass if an image can't be opened
            print(f"Error opening {img_path}")
    return images

def display_images_in_grid(images, cols=3):
    """Display images in a grid with specified number of columns"""
    rows = (len(images) + cols - 1) // cols
    for row in range(rows):
        cols_list = st.columns(cols)
        for col in range(cols):
            index = row * cols + col
            if index < len(images):
                with cols_list[col]:
                    st.image(images[index], use_column_width=True)

# Folder containing the images
folder = 'C:\AIML\ProducGPT\Image_Recognition\shelf_images'

# Load images
images = load_images_from_folder(folder)
st.title("Annotated Images with Bounding Boxes using trained model on SKU110k dataset")
st.write("The Sku110k dataset provides 11,762 images with more than 1.7 million annotated bounding boxes captured in densely packed scenarios, including 8,233 images for training, 588 images for validation, and 2,941 images for testing.")


st.write("Here are annotated images with bounding boxes:")
# Display images in a grid
display_images_in_grid(images, cols=3)  # Adjust cols as per requirement
