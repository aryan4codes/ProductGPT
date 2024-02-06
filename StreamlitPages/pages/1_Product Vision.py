import os
import streamlit as st
from pathlib import Path
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
# Set up the Google API key and model
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Dummy data for bar chart
product_names = ["Product A", "Product B", "Product C"]
occupancy_percentage = [30, 45, 25]

# Dummy data for line chart
time_periods = ["Week 1", "Week 2", "Week 3", "Week 4"]
visibility_score = [80, 75, 90, 85]

# Dummy data for heatmap
shelf_grid = [
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 0, 1]
]

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Streamlit UI elements
st.title("Product Vision")
st.write("Welcome to Product Vision! This feature allows you to request a statistical analysis of a product's placement based on user-defined parameters.")
shelf_occupancy_percentage = st.slider("Shelf Occupancy Percentage", 0, 100, 75)
visibility_rating = st.number_input("Visibility Rating", min_value=0.0, max_value=10.0, value=8.2, step=0.1)
shelf_placement = st.text_input("Shelf Placement", "Middle shelf")
competitor_products_beside = st.slider("Competitor Products Beside", 0, 10, 3)
incentive_provided = st.number_input("Incentive Provided to Pharma Store", min_value=0, value=15000)

st.sidebar.header("Settings")
# Upload images
image0 = st.sidebar.file_uploader("Upload store image(JPEG)", type=["jpg", "jpeg"])
image1 = st.sidebar.file_uploader("Upload medicine to search (PNG)", type=["jpg", "jpeg"])


# User-defined parameters

if image0 and image1:
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": image0.read()
        },
        {
            "mime_type": "image/png",
            "data": image1.read()
        },
    ]

    prompt_parts = [
        image_parts[0],
        image_parts[1],
        '''
        Provide a statistical analysis based on the following parameters:
         1. Shelf Occupancy Percentage: {shelf_occupancy_percentage}
         2. Visibility Rating: {visibility_rating}
         3. Shelf Placement: {shelf_placement}
         4. Competitor Products Beside: {competitor_products_beside}
         5. Compensation Provided to Pharma Store: {incentive_provided}
         Generate statistical insights into the product's performance and placement. Increase or decrease total compensation and give reason. Give a conclusion''',
    ]

    if st.button("Generate"):
        st.write("Generating response...")

        # Generate content using the model
        response = model.generate_content(prompt_parts)

        # Display the model's response
        st.subheader("Model's Response:")
        st.write(response.text)

        st.subheader("Dummy Visualizations:")
        
        # Bar Chart
        fig1, ax1 = plt.subplots()
        ax1.bar(product_names, occupancy_percentage)
        ax1.set_xlabel("Products")
        ax1.set_ylabel("Percentage Occupancy")
        ax1.set_title("Proportional Shelf Occupancy")
        st.pyplot(fig1)

        # Line Chart
        fig2, ax2 = plt.subplots()
        ax2.plot(time_periods, visibility_score, marker='o')
        ax2.set_xlabel("Time Period")
        ax2.set_ylabel("Visibility Score")
        ax2.set_title("Product Visibility Over Time")
        st.pyplot(fig2)

        # Heatmap
        fig3, ax3 = plt.subplots()
        sns.heatmap(shelf_grid, cmap='coolwarm', annot=True, fmt='d')
        ax3.set_xlabel("Column")
        ax3.set_ylabel("Row")
        ax3.set_title("Competitor Products' Placement")
        st.pyplot(fig3)
