import os
import cv2
# Define the folder containing the shelf images
folder_path = 'shelf_images'

# List to store bounding boxes for all images
all_bounding_boxes = []

# Function to create bounding boxes in an image
def create_bounding_boxes(image):
      # Display the image for user to draw bounding boxes
    cv2.imshow("Draw Bounding Boxes (press 's' to save)", image)
    
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
                cv2.rectangle(image, bbox[0], (x, y), (0, 255, 0), 2)
                cv2.imshow("Draw Bounding Boxes (press 's' to save)", image)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            bbox.append((x, y))
            cv2.rectangle(image, bbox[0], bbox[1], (0, 255, 0), 2)
            cv2.imshow("Draw Bounding Boxes (press 's' to save)", image)
            bounding_boxes.append(bbox)

    cv2.setMouseCallback("Draw Bounding Boxes (press 's' to save)", draw_rectangle)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break

    cv2.destroyWindow("Draw Bounding Boxes (press 's' to save)")

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Construct the full path to the image
        image_path = os.path.join(folder_path, filename)

        # Load the image
        image = cv2.imread(image_path)

        # Create bounding boxes for the image
        bounding_boxes = []
        create_bounding_boxes(image)

        # Append the bounding boxes to the list
        all_bounding_boxes.append((image_path, bounding_boxes))

# Print the list of bounding boxes for all images
for i, (image_path, bounding_boxes) in enumerate(all_bounding_boxes, start=1):
    print(f"Image {i}: {image_path}")
    for j, bbox in enumerate(bounding_boxes, start=1):
        print(f"  Bounding Box {j}: {bbox}")
