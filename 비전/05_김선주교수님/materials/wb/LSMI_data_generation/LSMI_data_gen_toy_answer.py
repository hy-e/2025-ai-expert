import cv2
import numpy as np

# 0. load img1 and img12 and get img2 
# img size (1048, 1048)
img1 = cv2.imread("img1.png")
img12 = cv2.imread("img12.png")
img2 = img12 - img1

# 1. extract illuminant colors 
# color1 box [50:100, 100:200]
# color2 box [900:1000, 800:820]
color1 = np.mean(np.mean(img1[50:100, 100:200], axis=0), axis=0) # (3,)
c1 = color1 / color1[1]
color2 = np.mean(np.mean(img2[900:1000, 800:820], axis=0), axis=0) # (3,)
c2 = color2 / color2[1]

print(c1, c2) # result values may be near [1.4, 1., 0.4], [0.5, 1., 1.8]

# 2. get the coefficient maps for illuminant 1 and 2
coeff1 = img1[..., 1] / (img12[..., 1] + 1e-6) # epsilon for preventing zero division
coeff2 = 1 - coeff1

# 3. calculate the illuminant maps and by using them, get the white-balanced img for both img1, img12 
illum1_map = c1 * coeff1[..., None] + 1e-6 # epsilon for preventing zero division
illum2_map = c2 * coeff2[..., None] + 1e-6 # epsilon for preventing zero division
illum12_map = illum1_map + illum2_map

wb_img1 =  img1 / illum1_map
wb_img2 =  img2 / illum2_map
wb_img12 = img12 / illum12_map

# 4. save the result images
cv2.imwrite("wb_img1.png", wb_img1)
cv2.imwrite("wb_img2.png", wb_img2)
cv2.imwrite("wb_img12.png", wb_img12)
cv2.imwrite("wb_img12_bright.png", wb_img12 * 2)



