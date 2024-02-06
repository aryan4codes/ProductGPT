import streamlit as st
import cv2
import numpy as np

# Open the video stream from the mobile camera
vid = cv2.VideoCapture('http://192.168.7.31:8080/video')

# Set the Streamlit app title
st.title('Using Mobile Camera with Streamlit')
darkornot=st.empty()
# Create a Streamlit image placeholder
frame_window = st.image([])

# Create a button to take a picture
take_picture_button = st.button('Take Picture')

# Run the app in an infinite loop
while True:
    # Read a frame from the video stream
    got_frame, frame = vid.read()

    # Convert the color space from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Check if a frame is obtained
    if got_frame:
        # Check if the frame is too dark
        if np.mean(frame) < 0.3 * 255:
            darkornot.write("Dark")
        else:
            darkornot.write("Acceptable")

        # Display the frame in Streamlit
        frame_window.image(frame)

        # Display the result text using st.text
        

    # Check if the "Take Picture" button is pressed
    if take_picture_button:
        # Perform any additional processing or model inference here if needed
        break

# Release the video stream
vid.release()
