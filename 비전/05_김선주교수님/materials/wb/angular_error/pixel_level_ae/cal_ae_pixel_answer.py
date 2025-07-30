import cv2
import numpy as np

img1 = cv2.imread('sample_1.png') # (256, 256, 3) value 5~254
img2 = cv2.imread('sample_2.png') # (256, 256, 3) value 5~254

# v1 - iteration
mae = 0.
for i in range(img1.shape[0]):
	for j in range(img1.shape[1]):
		pixel1 = img1[i][j]
		pixel2 = img2[i][j]

		pixel1_normalized = pixel1 / np.linalg.norm(pixel1)
		pixel2_normalized = pixel2 / np.linalg.norm(pixel2)

		cos_sim = np.dot(pixel1_normalized, pixel2_normalized)

		ae = np.arccos(cos_sim) * 180 / np.pi
		mae += ae / (img1.shape[0] * img2.shape[1])
print(mae)


# v2 - parallelization
# normalize every pixel to be unit vector
img1_normalized_per_pixel = img1 / np.linalg.norm(img1, axis=2)[:, :, None]
img2_normalized_per_pixel = img2 / np.linalg.norm(img2, axis=2)[:, :, None]

# result of dot product of unit vectors == cos_sim_map
cos_sim_map = np.sum(img1_normalized_per_pixel * img2_normalized_per_pixel, axis=2) 

# convert cos similarity to angular error
ae_map = np.arccos(cos_sim_map) * 180 / np.pi

mae = np.mean(ae_map)
print(mae)
