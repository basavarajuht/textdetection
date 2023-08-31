import cv2
import numpy as np
from matplotlib import pyplot as plt

f = cv2.imread('C:\\Users\\Lenovo\\Desktop\\textdetect\\data\\data\\im\\10.png')
fg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
plt.imshow(fg)
plt.show()

I2 = cv2.GaussianBlur(fg, (0,0), 3)
I3 = cv2.adaptiveThreshold(I2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
I4 = cv2.bitwise_not(I3)
se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
I5 = cv2.dilate(I4, se1)
I6 = cv2.morphologyEx(I5, cv2.MORPH_BRIDGE, None)
plt.imshow(I6)
plt.show()