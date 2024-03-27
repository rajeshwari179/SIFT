import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to extract features and descriptors from an image
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# Function to match features between two images
def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    return good_matches

# Function to estimate camera pose from matched features
def estimate_camera_pose(matched_keypoints1, matched_keypoints2, K):
    E, _ = cv2.findEssentialMat(matched_keypoints1, matched_keypoints2, K)
    _, R, t, _ = cv2.recoverPose(E, matched_keypoints1, matched_keypoints2, K)
    return R, t

# Function to triangulate 3D points from matched features and camera poses
# Function to triangulate 3D points from matched features and camera poses
def triangulate_points(matched_keypoints1, matched_keypoints2, R1, t1, R2, t2, K):
    P1 = np.dot(K, np.hstack((R1, t1)))
    P2 = np.dot(K, np.hstack((R2, t2)))

    # Reshape matched keypoints to have shape (2, N)
    matched_keypoints1 = np.squeeze(matched_keypoints1).T
    matched_keypoints2 = np.squeeze(matched_keypoints2).T

    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, matched_keypoints1, matched_keypoints2)
    points_3d = points_4d_homogeneous / points_4d_homogeneous[3]  # Normalize by homogeneous coordinate
    points_3d = points_3d[:3].T  # Transpose to get points in shape (N, 3)

    return points_3d

# Load images
images = [cv2.imread(f'{i:04d}.png') for i in range(0, 11)]

# Intrinsics matrix (assuming the same for all images)
focal_length = 500  # Adjust according to your camera parameters
image_width = images[0].shape[1]
image_height = images[0].shape[0]
K = np.array([[focal_length, 0, image_width / 2],
              [0, focal_length, image_height / 2],
              [0, 0, 1]])

# Extract features and descriptors for all images
keypoints_list = []
descriptors_list = []
for image in images:
    keypoints, descriptors = extract_features(image)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# Match features between consecutive images and perform SfM
all_3d_points = []
for i in range(len(images) - 1):
    matched_keypoints1 = keypoints_list[i]
    matched_keypoints2 = keypoints_list[i + 1]
    descriptors1 = descriptors_list[i]
    descriptors2 = descriptors_list[i + 1]
    
    good_matches = match_features(descriptors1, descriptors2)
    matched_keypoints1 = np.float
    32([matched_keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matched_keypoints2 = np.float32([matched_keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    R, t = estimate_camera_pose(matched_keypoints1, matched_keypoints2, K)
    if i == 0:
        R1 = np.eye(3)
        t1 = np.zeros((3, 1))
    else:
        R1 = prev_R
        t1 = prev_t
    prev_R = R
    prev_t = t
    
    points_3d = triangulate_points(matched_keypoints1, matched_keypoints2, R1, t1, R, t, K)
    all_3d_points.append(points_3d)

# Visualize the 3D points or further process them as needed
print("3D points reconstructed successfully!")




# Concatenate all 3D points
all_3d_points = np.concatenate(all_3d_points)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points
ax.scatter(all_3d_points[:,0], all_3d_points[:,1], all_3d_points[:,2], c='b', marker='o')

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()
