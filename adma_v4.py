import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('C:\\Users\\Lenovo\\Downloads\\62.png')
# Convert the image to grayscale
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply FFT (Fast Fourier Transform)
f = np.fft.fft2(gray_img)
fshift = np.fft.fftshift(f)
# Get the magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift))
# Apply inverse FFT to get the filtered image
fshift[np.abs(fshift) < 100] = 0 # Applying a high-pass filter
f_ishift = np.fft.ifftshift(fshift)
filtered_img = np.fft.ifft2(f_ishift)
filtered_img = np.abs(filtered_img)
# Normalize the filtered image for display
filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)
# Convert the filtered image to 8-bit unsigned integer
filtered_img = np.uint8(filtered_img)


# Apply thresholding to get a binary image
# _, binary_img = cv2.threshold(filtered_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
edges = cv2.Canny(filtered_img, 100, 200)
# Perform dilation and erosion
kernel1 = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel1, iterations=1)
eroded = cv2.erode(dilated, kernel1, iterations=1)
cv2.imshow('filtered',eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('fft_1.png',eroded)
