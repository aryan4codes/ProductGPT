import cv2
import numpy as np
import os

# Load the target image
target_image = cv2.imread('target.png')

# Load the template image
template_image = cv2.imread('template.png')

template_height, template_width, _ = template_image.shape

# Crop a region from the target image with the same size as the template
target_region = target_image[0:template_height, 0:template_width]

# Convert both images to grayscale (for template matching)
target_gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)


# Perform template matching
result = cv2.matchTemplate(target_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# Set a threshold to consider a match
threshold = 0.2
loc = np.where(result >= threshold)
print(f"Number of matches found: {len(loc[0])}")

# Draw rectangles around the matched regions on the template image
template_with_boxes = template_image.copy()
template_height, template_width, _ = template_image.shape

for pt in zip(*loc[::-1]):
    cv2.rectangle(template_with_boxes, pt, (pt[0] + template_width, pt[1] + template_height), (0, 0, 255), 2)

# Display the template image with matching rectangles
cv2.imshow('Template with Matches', template_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Draw rectangles around the matched regions
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(targettemplate_image, pt, (pt[0] + template_image.shape[1], pt[1] + template_image.shape[0]), (0, 255, 0), 2)

# # Display the result
# cv2.imwrite('/C:/AIML/ProducGPT/final.png', target_image)
# cv2.imshow('Object Found', target_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # Save the final image
