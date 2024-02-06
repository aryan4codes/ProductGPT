import streamlit as st
import cv2
import os
import numpy as np

# Function to create bounding boxes in an image
def create_bounding_boxesq(image):
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

st.title("Shelf Placement Analysis")

st.write("This analyzes the placement of products on shelves.")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    
    # for bbox in bounding_boxes:
    #     top_left = (min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1]))
    #     bottom_right = (max(bbox[0][0], bbox[1][0]), max(bbox[0][1], bbox[1][1]))
    #     cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    st.image(image, channels="BGR", use_column_width=True, caption="Annotated Image")
from skimage.metrics import structural_similarity
import os

# Function to perform image similarity comparison
def compare_images(image1, image2):
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, _ = structural_similarity(image1_gray, image2_gray, full=True)
    return score

# Define the image data with bounding box coordinates
image_data = [
    ("C:\AIML\ProducGPT\Image_Recognition\shelf_images\IMG-20240206-WA0010_annotated_marked.jpg", [
         [(20, 271), (183, 358)],
    [(22, 371), (22, 371)],
    [(23, 373), (189, 453)],
    [(24, 466), (195, 559)],
    [(32, 565), (192, 655)],
    [(22, 667), (193, 755)],
    [(197, 252), (371, 340)],
    [(199, 355), (377, 431)],
    [(203, 445), (368, 532)],
    [(217, 543), (359, 637)],
    [(212, 662), (359, 742)],
    [(219, 742), (343, 790)],
    [(383, 387), (544, 490)],
    [(377, 489), (540, 584)],
    [(374, 589), (498, 714)],
    [(382, 723), (494, 785)],
    [(554, 413), (671, 517)],
    [(548, 521), (663, 624)],
    [(647, 629), (786, 734)],
    [(678, 525), (807, 630)],
    [(682, 423), (800, 525)],
    [(679, 317), (795, 416)],
    [(557, 309), (665, 409)],
    [(805, 312), (912, 423)],
    [(816, 430), (917, 524)],
    [(809, 527), (919, 618)],
    [(780, 625), (901, 723)],
    [(791, 726), (889, 792)],
    [(924, 273), (1051, 376)],
    [(916, 383), (1041, 490)],
    [(923, 497), (1049, 590)],
    [(921, 594), (1043, 714)],
    [(900, 712), (1012, 777)],
    [(1062, 277), (1189, 374)],
    [(1062, 386), (1177, 483)],
    [(1046, 490), (1166, 596)],
    [(1043, 601), (1165, 710)],
    [(1021, 714), (1135, 783)],
    [(402, 294), (526, 388)],
    [(519, 628), (633, 725)],
    ]),
    ("C:\AIML\ProducGPT\Image_Recognition/shelf_images/IMG-20240206-WA0011_annotated.jpg", [
         [(328, 209), (647, 397)],
   [(351, 408), (657, 587)],
   [(383, 590), (692, 771)],
   [(155, 277), (329, 567)],
   [(183, 599), (346, 767)],
    ]),
    ("C:\AIML\ProducGPT\Image_Recognition/shelf_images/IMG-20240206-WA0012_annotated.jpg", [
    [(130, 231), (301, 349)],
[(104, 355), (288, 478)],
[(341, 296), (456, 391)],
[(338, 401), (449, 496)],
[(322, 604), (445, 704)],
[(456, 504), (598, 605)],
[(459, 402), (597, 496)],
[(473, 297), (609, 395)],
[(606, 417), (735, 501)],
  [(603, 509), (732, 599)],
  [(615, 608), (742, 701)],
  [(614, 706), (739, 783)],
  [(756, 275), (882, 403)],
  [(741, 405), (877, 505)],
  [(742, 507), (873, 604)],
  [(746, 608), (875, 699)],
  [(744, 701), (874, 789)],
  [(913, 264), (1041, 369)],
  [(889, 371), (1014, 486)],
  [(875, 484), (1017, 592)],
  [(879, 601), (1003, 691)],
  [(880, 701), (1006, 779)],
  [(1050, 315), (1172, 404)],
  [(1021, 408), (1149, 505)],
  [(1021, 509), (1143, 595)],
  [(1021, 602), (1146, 699)],
  [(1012, 701), (1145, 786)],
  [(1174, 294), (1292, 392)],
  [(1162, 414), (1295, 540)],
  [(1173, 543), (1311, 671)],
  [(1184, 680), (1326, 787)],
  [(328, 706), (454, 786)],
  [(470, 618), (586, 717)],
  [(459, 722), (604, 784)],
  [(341, 496), (458, 595)],
  [(95, 487), (249, 604)],
  [(107, 611), (265, 727)],
  [(102, 734), (264, 800)],
  [(620, 315), (727, 415)],
        # Add more bounding boxes for Image 3
    ]),
    ("C:\AIML\ProducGPT\Image_Recognition/shelf_images/IMG-20240206-WA0013_annotated.jpg", [
         [(435, 391), (620, 569)],
   [(356, 594), (512, 780)],
   [(685, 455), (922, 615)],
   [(652, 268), (868, 421)],
   [(707, 637), (951, 802)],
   [(772, 93), (869, 277)],
   [(865, 96), (956, 329)],
   [(1032, 498), (1229, 747)],
   [(116, 414), (245, 691)],
   [(163, 198), (261, 305)],
   [(275, 204), (376, 310)],
    ]),
    ("C:\AIML\ProducGPT\C:\AIML\ProducGPT\Image_Recognition\shelf_images\IMG-20240206-WA0015_annotated.jpg", [
         [(559, 266), (699, 582)],
   [(697, 363), (859, 580)],
   [(863, 386), (988, 584)],
   [(990, 384), (1124, 587)],
   [(1129, 386), (1264, 588)],
   [(233, 261), (417, 577)],
   [(184, 628), (355, 750)],
   [(476, 294), (543, 586)],
    ]),
    ("C:\AIML\ProducGPT\Image_Recognition/shelf_images/IMG-20240206-WA0016_annotated.jpg", [
         [(265, 355), (427, 731)],
   [(414, 260), (628, 722)],
   [(642, 255), (830, 777)],
   [(841, 246), (1039, 728)],
   [(1055, 239), (1285, 731)],
   [(17, 362), (245, 756)],
   [(336, 53), (618, 210)],
   [(634, 96), (864, 163)],
   [(625, 162), (861, 224)],
    ]),
    # Add more images and bounding boxes as needed
]

# Threshold for considering a match (adjust as needed)
similarity_threshold = 0.30

# Loop through each image in the image_data list
for image_path, bounding_boxes in image_data:
    if not os.path.exists(image_path) or not os.access(image_path, os.R_OK):
        print(f"File does not exist or is not accessible: {image_path}")
        continue

            # Load the shelf image
    shelf_image = cv2.imread(image_path)
    if shelf_image is None:
        print(f"Failed to load image from {image_path}")
        continue

    # List to store matched bounding boxes
    matched_boxes = []

    # Loop through each bounding box
    for i, ((x1, y1), (x2, y2)) in enumerate(bounding_boxes):
        # Crop the bounding box region from the shelf image

        
        if x2>x1 and y2>y1:

            cropped_image = shelf_image[y1:y2, x1:x2]

            # Load the target product image
            target_product_image = image
            
            if target_product_image is None:
                print('Wrong path:')
            try:
                target_product_image = cv2.resize(target_product_image, (cropped_image.shape[1], cropped_image.shape[0]))
            except cv2.error as e:
                print(f"Error resizing target product image: {e}")
                continue
            # Compare the cropped image with the target product image
            similarity_score = compare_images(target_product_image, cropped_image)

            # Check if the similarity score exceeds the threshold
            if similarity_score >= similarity_threshold:
                matched_boxes.append(((x1, y1), (x2, y2)))

    # Draw bounding boxes on the shelf image for matched boxes
    for ((x1, y1), (x2, y2)) in matched_boxes:
        cv2.rectangle(shelf_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Calculate the occupancy percentage of the target product
    total_area = shelf_image.shape[0] * shelf_image.shape[1]
    occupancy_percentage = (sum([(x2 - x1) * (y2 - y1) for ((x1, y1), (x2, y2)) in matched_boxes]) / total_area) * 100

    # Display or save the shelf image with marked bounding boxes
    annotated_image_path = image_path.replace(".jpg", "_marked.jpg")
    cv2.imwrite(annotated_image_path, shelf_image)

    # Print the matched bounding boxes and occupancy percentage for each image
    # st.write(f"Image: {image_path}")
    # st.write("Matched Bounding Boxes:", matched_boxes)
    st.write("Occupancy Percentage of Target Product:", occupancy_percentage)
    print()
    break

cv2.destroyAllWindows()
