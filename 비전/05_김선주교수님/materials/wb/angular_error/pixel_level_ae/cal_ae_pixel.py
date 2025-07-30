import cv2
import numpy as np

img1 = cv2.imread('sample_1.png') # (256, 256, 3) value 5~254
img2 = cv2.imread('sample_2.png') # (256, 256, 3) value 5~254

# Write your code for calculating pixel-level mae
img1_norm = img1 / np.linalg.norm(img1, axis=2)[:, :, None]
img2_norm = img2 / np.linalg.norm(img2, axis=2)[:, :, None]
cos_sim = np.sum(img1_norm * img2_norm, axis=2)
ae_map = np.arccos(cos_sim) * 180 / np.pi
mae = np.mean(ae_map)


print(mae)


