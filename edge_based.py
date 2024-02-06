# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the target image
# target_image = cv2.imread('test_inputs/apple.jpg')

# # Load the template image
# template_image = cv2.imread('test_inputs/apple_basket.jpg')

# template_height, template_width, _ = template_image.shape

# # Crop a region from the target image with the same size as the template
# target_region = target_image[0:template_height, 0:template_width]

# # Convert both images to grayscale (for template matching)
# target_gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)
# template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

# # Apply Canny edge detection to both images
# target_edges = cv2.Canny(target_gray, 100, 200)
# template_edges = cv2.Canny(template_gray, 100, 200)

# # Perform template matching on the edge images
# result = cv2.matchTemplate(target_edges, template_edges, cv2.TM_CCOEFF_NORMED)

# # Set a threshold to consider a match
# threshold = 0.2
# loc = np.where(result >= threshold)

# if len(loc[0]) == 0:
#     print("No Match Found")
# else:
#     # Draw rectangles around matches
#     for pt in zip(*loc[::-1]):
#         cv2.rectangle(target_image, pt, (pt[0] + template_width, pt[1] + template_height), (0, 255, 0), 2)

#     # Save the final image
#     cv2.imwrite('final.png', target_image)

#     # Display the result
#     plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# Python program to illustrate 
# template matching 
import cv2
import numpy as np
import imutils

# Read the main image
img_rgb = cv2.imread('test_inputs/apple_basket.jpg')

# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
template = cv2.imread('test_inputs/apple.jpg', 0)

# Store width and height of the template in w and h
w, h = template.shape[::-1]

# Define the scale for resizing (you need to set this value)
scale = 0.96  # Set the appropriate scale value

# Resize the image according to scale
resize = imutils.resize(img_gray, width=int(img_gray.shape[1] * scale))

# If the resized image is smaller than the template, break the loop
if resize.shape[0] < h or resize.shape[1] < w:
    print("Image too small for template matching.")
else:
    # Detect edges in the resized, grayscale image and apply template matching to find the template in the image
    result = cv2.matchTemplate(resize, template, cv2.TM_CCOEFF_NORMED)
    
    # Get the maximum correlation value and its location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Unpack the found variables and compute (x, y) coordinates of the bounding box
    (startX, startY) = (int(max_loc[0]), int(max_loc[1]))
    (endX, endY) = (int(max_loc[0] + w), int(max_loc[1] + h))
    
    # Draw a bounding box around the detected result and display the image
    cv2.rectangle(img_rgb, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()