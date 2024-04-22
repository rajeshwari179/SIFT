import numpy as np
import cv2
import matplotlib.pyplot as plt

# load assets\dataset.jpeg
image = cv2.imread('assets/dataset.jpeg')

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# create a SIFT object
sift = cv2.SIFT_create()

# detect SIFT keypoints and descriptors in the image
keypoints, descriptors = sift.detectAndCompute(gray, None)

# draw the keypoints on the image, with size and orientation
image = cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# save the image
cv2.imwrite('assets/sift_keypoints.png', image)