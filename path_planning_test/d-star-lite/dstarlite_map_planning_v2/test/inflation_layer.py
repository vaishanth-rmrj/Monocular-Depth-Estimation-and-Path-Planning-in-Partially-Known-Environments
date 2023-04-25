import cv2
import numpy as np

# Load the image
img = cv2.imread("dstarlite_map_based/assets/Map_2.jpg", cv2.IMREAD_GRAYSCALE)

# Convert to binary format
img_bin = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)[1]

# Calculate the distance transform
dist_transform = cv2.distanceTransform(img_bin, cv2.DIST_L2, 3)

# Apply dilation operation
robot_size = 15 # Define the size of the robot
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (robot_size, robot_size))
inflated_dist_transform = cv2.dilate(dist_transform, kernel)

# Convert back to binary format
inflated_bin = cv2.threshold(inflated_dist_transform, robot_size, 1, cv2.THRESH_BINARY)[1]
inflated_bin = inflated_bin.astype(np.uint8)
bitwise_op = cv2.bitwise_xor(img_bin, inflated_bin)
bitwise_op = np.where(bitwise_op == 1, 0, 1)
bitwise_op*=255
# print(bitwise_op)

# Calculate the distance transform
dist_transform_2 = cv2.distanceTransform(inflated_bin, cv2.DIST_L2, 3)
img_normalized = cv2.normalize(dist_transform_2, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
img_dist_thres = np.where(img_normalized*2 > 1, 1, img_normalized*2)
img_dist_thres *= 255

# Save the result
cv2.imshow("Inflation Layer", img_dist_thres.astype(np.uint8))
cv2.imshow("Org Layer", img.astype(np.uint8))
cv2.waitKey(0)

cv2.imwrite("dstarlite_map_based_v3/assets/inflated_map_3.jpg", img_dist_thres.astype(np.uint8))