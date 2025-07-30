import cv2
import numpy as np
import time


# iter 
img = cv2.imread("wb_img12_bright.png")
iter_s = time.time()
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		if img[i][j][1] > 200: # find white pixel
			img[i][j] = img[i][j] / 2
iter_e = time.time()
iter_latency = iter_e-iter_s
print(f"iter - latency: {iter_latency:.4f}")
cv2.imwrite("img_iter.png", img)


# para
img = cv2.imread("wb_img12_bright.png")
para_s = time.time()
mask = (img[..., 1:2] > 200).astype(img.dtype)
img = img - img * (mask * 0.5)
para_e = time.time()
para_latency = para_e-para_s
print(f"para - latency: {para_latency:.4f}")
cv2.imwrite("img_para.png", img)


print(f"<para> is x{int((iter_latency)/(para_latency))} faster than <iter>")