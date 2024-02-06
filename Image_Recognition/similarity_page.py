import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import os
import io

# Function to perform image similarity comparison
def compare_images(image1, image2):
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, _ = structural_similarity(image1_gray, image2_gray, full=True)
    return score

# Streamlit UI elements
st.title("Shelf Image Similarity Comparison")
st.write("Upload a shelf image and a target product image, and the app will compare them.")

# Upload shelf image
shelf_image = st.file_uploader("Upload a shelf image (JPEG)", type=["jpg", "jpeg"])

# Upload target product image
target_product_image = st.file_uploader("Upload the target product image (JPEG)", type=["jpg", "jpeg"])

# User-defined similarity threshold
similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3)

# Placeholder for matched images
matched_images = []

if shelf_image is not None and target_product_image is not None:
    # Load the uploaded shelf image
    shelf_image = cv2.imdecode(np.frombuffer(shelf_image.read(), np.uint8), cv2.IMREAD_COLOR)

    
    if shelf_image is not None:
        # Load the uploaded target product image
        target_product_image = cv2.imdecode(np.frombuffer(target_product_image.read(), np.uint8), cv2.IMREAD_COLOR)

        

        if target_product_image is not None:
            st.image(shelf_image, caption="Uploaded Shelf Image", use_column_width=True)
            st.image(target_product_image, caption="Uploaded Target Product Image", use_column_width=True)
            
            # List to store matched bounding boxes
            matched_boxes = []
            
            # Perform image similarity comparison
            similarity_score = compare_images(target_product_image, shelf_image)
            
            # Check if the similarity score exceeds the threshold
            if similarity_score >= similarity_threshold:
                matched_boxes.append(((0, 0), (shelf_image.shape[1], shelf_image.shape[0])))
                
                # Calculate the occupancy percentage of the target product
                total_area = shelf_image.shape[0] * shelf_image.shape[1]
                occupancy_percentage = (sum([(x2 - x1) * (y2 - y1) for ((x1, y1), (x2, y2)) in matched_boxes]) / total_area) * 100
                
                # Display the occupancy percentage
                st.write(f"Occupancy Percentage of Target Product: {occupancy_percentage:.2f}%")
                
                # Add the shelf image to the list of matched images
                matched_images.append((occupancy_percentage, shelf_image))
            else:
                st.write("No matching target product found. Similarity score:", similarity_score)
        
        else:
            st.write("Failed to load the uploaded target product image.")
    else:
        st.write("Failed to load the uploaded shelf image.")

# Display matched images where occupancy > 0%
if matched_images:
    st.subheader("Matched Images (Occupancy > 0%):")
    matched_images.sort(reverse=True, key=lambda x: x[0])
    for occupancy_percentage, matched_image in matched_images:
        st.image(matched_image, caption=f"Occupancy Percentage: {occupancy_percentage:.2f}%", use_column_width=True)

# Display the app
st.write("Modify the similarity threshold and upload different shelf and target product images to compare.")
