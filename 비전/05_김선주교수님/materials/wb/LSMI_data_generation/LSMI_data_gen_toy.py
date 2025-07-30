import cv2
import numpy as np


# 0. load img1 and img12 and get img2 
# (you can use cv2.imread())
# img size (1048, 1048)


# 1. extract illuminant colors 
# c1 box location [50:100, 100:200]
# c2 box location [900:1000, 800:820]
# (let c1 and c2 be the normalized illuminant colors)

print(c1, c2) # result values may be near [1.4, 1., 0.4], [0.5, 1., 1.8]


# 2. get the coefficient maps for illuminant 1 and 2
# (let coeff1 and coeff2 be the coefficient map for the illuminant1 and 2 respectively)


# 3. calculate the illuminant maps and by using them, get the white-balanced img for both img1, img12 
# (let wb_img1, wb_img12 be the white-balanced img for img1 and img12 respectively)


# 4. save the result images
cv2.imwrite("wb_img1.png", wb_img1)
cv2.imwrite("wb_img12.png", wb_img12)


