import cv2
import numpy as np

for i in range(5):
	r1 = np.random.randint(255)
	g1 = np.random.randint(255)
	b1 = np.random.randint(255)

	r2 = np.random.randint(255)
	g2 = np.random.randint(255)
	b2 = np.random.randint(255)

	rgb1 = np.array([r1, g1, b1])
	rgb2 = np.array([r2, g2, b2])
	# print("rgb1:", rgb1)
	# print("rgb2:", rgb2)

	rgb1_normalized = rgb1 / np.linalg.norm(rgb1) # unit vector
	rgb2_normalized = rgb2 / np.linalg.norm(rgb2) # unit vector
	# print("rgb1_normalized:", rgb1_normalized)
	# print("rgb2_normalized:", rgb2_normalized)

	cos_sim = np.dot(rgb1_normalized, rgb2_normalized)
	# print("cos_sim:", cos_sim)

	angular_error = np.arccos(cos_sim) * 180 / np.pi
	# print("angular_error:", angular_error)


	# visualize color
	img = np.zeros((256,256,3))
	img[:128, :, :] = rgb1
	img[128:, :, :] = rgb2
	cv2.imwrite(f"two_color_{angular_error:.2f}.png", img)
