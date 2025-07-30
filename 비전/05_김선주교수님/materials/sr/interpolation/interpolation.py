import cv2

input_img = cv2.imread("original.png") # (32, 32, 3)

nearest = cv2.resize(input_img, dsize=(100,100), interpolation=cv2.INTER_NEAREST)
bilinear = cv2.resize(input_img, dsize=(100,100), interpolation=cv2.INTER_LINEAR)
bicubic = cv2.resize(input_img, dsize=(100,100), interpolation=cv2.INTER_CUBIC)

cv2.imwrite("test_nearest_100.png", nearest)
cv2.imwrite("test_bilinear_100.png", bilinear)
cv2.imwrite("test_bicubic_100.png", bicubic)