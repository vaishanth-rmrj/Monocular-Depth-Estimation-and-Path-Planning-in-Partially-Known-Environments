import cv2
import numpy as np

# Load the map image
img = cv2.imread('dstarlite_map_based_v3/assets/Map_1.jpg', cv2.IMREAD_GRAYSCALE)

# Apply threshold to convert the grayscale image to binary
# Convert to binary format
img_bin = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)[1]

# Dilate the binary image to account for the robot size
robot_size = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (robot_size, robot_size))
dilated = cv2.dilate(thresh, kernel)

# Calculate distance transform to compute the cost values
# dist_transform = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
# max_distance = np.amax(dist_transform)
# scaled_distance = (dist_transform / max_distance) * 255

# # Convert the distance transform to uint8 type and invert it
# costmap = cv2.convertScaleAbs(scaled_distance)
# costmap = cv2.bitwise_not(costmap)

# Save the costmap as an image
cv2.imwrite('costmap.jpg', dilated)