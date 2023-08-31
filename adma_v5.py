import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('C:\\Users\\Lenovo\\Downloads\\63.png')
# Convert the image to grayscale
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur
gausBlur = cv2.GaussianBlur(gray_img, (1, 1), 0)
# Apply bilateral filtering
bilateral = cv2.bilateralFilter(gausBlur, 9, 20, 20)
# Apply high pass filter
kernel = np.array([[0, -1, 0],
[-1, 32, -1],
[0, -1, 0]])
kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
high = cv2.filter2D(gausBlur, -1, kernel)
# Apply Canny edge detection
edges = cv2.Canny(high, 100, 200)
# Perform dilation and erosion
kernel1 = np.ones((0, 0), np.uint8)
dilated = cv2.dilate(edges, kernel1, iterations=1)
eroded = cv2.erode(dilated, kernel1, iterations=1)
cv2.imshow('cleaned',eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('bilat_gaus2.png',eroded)