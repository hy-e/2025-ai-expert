import cv2
import numpy as np 

# img size (1048, 1048)
img = cv2.imread("samsung-logo.jpeg") // 2

color1 = np.array([1.4, 1., 0.4])
color2 = np.array([0.5, 1., 1.8])

coeff1 = np.random.random((1048, 1048, 1))
coeff1[50:100, 100:200] = 1.
coeff1[900:1000, 800:820] = 0.
coeff2 = 1 - coeff1

img1 = img * color1 * coeff1
img2 = img * color2 * coeff2
img12 = img1 + img2

cv2.imwrite("img1.png", img1)
cv2.imwrite("img2.png", img2)
cv2.imwrite("img12.png", img12)