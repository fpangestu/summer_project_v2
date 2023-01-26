import numpy as np
import cv2

#Read image
img = cv2.imread('box-1.jpg')

#Gaussian blur
blurred = cv2.GaussianBlur(img, (5, 5), 0)

#Convert to graysscale
gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)

#Autocalculate the thresholding level
threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#Threshold
retval, bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

#Find contours
bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#Sort out the biggest contour (biggest area)
max_area = 0
max_index = -1
index = -1
for i in contours:
    area = cv2.contourArea(i)
    index=index+1
    if area > max_area :
        max_area = area
        max_index = index

#Draw the raw contours
cv2.drawContours(img, contours, max_index, (0, 255, 0), 3 )
cv2.imwrite("box-1-biggest-contour.png", img)

#Draw a rotated rectangle of the minimum area enclosing our box (red)
cnt=contours[max_index]
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img = cv2.drawContours(img,[box],0,(0,0,255),2)

#Show original picture with contour
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 1. Find contour
# 2. blur
# 3. Grayscale
# 4. threshold
# 5. detect contours