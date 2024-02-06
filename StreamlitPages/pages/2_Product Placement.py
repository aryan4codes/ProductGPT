import streamlit as st
import cv2
import os
import numpy as np

# Function to create bounding boxes in an image
def create_bounding_boxes(image):
    bounding_boxes=[]
     # Resize the image to a smaller size (e.g., 800x600)
    resized_image = cv2.resize(image, (800, 600))

    # Display the resized image for the user to draw bounding boxes
    cv2.imshow("Draw Bounding Boxes (press 's' to save)", resized_image)

    # Initialize variables for drawing bounding boxes
    drawing = False
    bbox = []

    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, bbox

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            bbox = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # Create a copy of the resized image to avoid multiple annotations
                image_copy = resized_image.copy()
                cv2.rectangle(image_copy, bbox[0], (x, y), (0, 255, 0), 2)
                cv2.imshow("Draw Bounding Boxes (press 's' to save)", image_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            if drawing:
                drawing = False
                bbox.append((x, y))
                cv2.rectangle(resized_image, bbox[0], bbox[1], (0, 255, 0), 2)
                cv2.imshow("Draw Bounding Boxes (press 's' to save)", resized_image)
                print(bbox)
                bounding_boxes.append(bbox)
        

    cv2.setMouseCallback("Draw Bounding Boxes (press 's' to save)", draw_rectangle)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break

    cv2.destroyWindow("Draw Bounding Boxes (press 's' to save)")
    
    return bounding_boxes, resized_image

# Function to compute placement scores
def compute_placement_scores(target_bbox, shelf_bbox):
    # Compute the dimensions of the shelf
    shelf_width = shelf_bbox[2] - shelf_bbox[0]
    shelf_height = shelf_bbox[3] - shelf_bbox[1]

    # Compute the bottom placement score
    bottom_distance = shelf_bbox[3] - target_bbox[1]  # Vertical distance from product bottom to shelf bottom
    bottom_placement_score = bottom_distance / shelf_height

    # Compute the horizontal corner placement score
    target_center_x = (target_bbox[0] + target_bbox[2]) / 2
    horizontal_distance_to_left = target_center_x - shelf_bbox[0]
    horizontal_distance_to_right = shelf_bbox[2] - target_center_x
    horizontal_corner_score = abs(target_center_x - (shelf_bbox[0] + shelf_bbox[2]) / 2) / (shelf_width / 2)

    return abs(1-abs(bottom_placement_score)), abs(horizontal_corner_score)

def main():
    st.title("Product Placement Analysis")

    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        bounding_boxes, resized_image = create_bounding_boxes(image)
        # for bbox in bounding_boxes:
        #     top_left = (min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1]))
        #     bottom_right = (max(bbox[0][0], bbox[1][0]), max(bbox[0][1], bbox[1][1]))
        #     cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        st.image(resized_image, channels="BGR", use_column_width=True, caption="Annotated Image")

        if len(bounding_boxes) == 2:
            bottom_placement_score, horizontal_corner_score = compute_placement_scores(
                (bounding_boxes[1][0][0], bounding_boxes[1][0][1], bounding_boxes[1][1][0], bounding_boxes[1][1][1]),
                (bounding_boxes[0][0][0], bounding_boxes[0][0][1], bounding_boxes[0][1][0], bounding_boxes[0][1][1])
            )
            st.warning(f"Bottom Placement Score: {bottom_placement_score*100:.2f}%  from top")
            st.warning(f"Horizontal Corner Score: {horizontal_corner_score*100:.2f}%   from centre")
            st.warning(f"Lighting Score: {100/255 * np.mean(resized_image[bounding_boxes[1][0][1]:bounding_boxes[1][1][1],bounding_boxes[1][0][0]:bounding_boxes[1][1][0]]):.2f}%")
        else:
            st.sidebar.warning("Please draw two bounding boxes on the image.")

if __name__ == "__main__":
    main()
