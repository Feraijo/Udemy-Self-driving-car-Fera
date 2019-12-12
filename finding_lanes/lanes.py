import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Image/test_image.jpg')

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height, width = image.shape
    triangle = np.array([[(200, height),
                         (1100, height),
                         (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    return mask

lane_image = np.copy(image)
c = canny(lane_image)

#plt.imshow(c)
#plt.show()
cv2.imshow('res', region_of_interest(c))
cv2.waitKey(0)