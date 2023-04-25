import cv2
import numpy as np

# Load the image
img = cv2.imread("dstarlite_map_based/assets/Map_2.jpg", cv2.IMREAD_GRAYSCALE)

# Convert to binary format
# map_binary = cv2.threshold(img, 150, 1, cv2.THRESH_BINARY)[1]

# inflation layer
# Apply dilation operation
# robot_size = 10 # Define the size of the robot
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (robot_size, robot_size))
# inflated_map = cv2.dilate(map_binary, kernel)
# Dilate the binary image to account for the robot size
kernel = np.ones((10,10),np.uint8)
dilated = cv2.dilate(img.astype(np.uint8), kernel)
inflated_map = dilated
print(inflated_map)
# cost map

# Calculate the distance transform
# dist_transform = cv2.distanceTransform(img_bin, cv2.DIST_L2, 0)
# print(dist_transform)



# # Convert back to binary format
# inflated_bin = cv2.threshold(inflated_dist_transform, robot_size, 1, cv2.THRESH_BINARY)[1]

# img_normalized = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# img_dist_thres = np.where(img_normalized*3 > 1, 1, img_normalized*3)
# img_dist_thres *= 255

# map_img = img_dist_thres.astype(np.uint8)
# Save the result
print(inflated_map)


while True:
    cv2.namedWindow("Inflated map", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Inflated map", inflated_map.astype(np.uint8))
    if cv2.waitKey(24) == ord('q'):
        break

# cv2.imwrite('inflated_map_1.jpg', map_img*255)