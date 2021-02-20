import cv2
import numpy as np

img = cv2.imread('test13.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
params = cv2.SimpleBlobDetector_Params()
detector = cv2.SimpleBlobDetector_create(params)




#Red color rangle  169, 100, 100 , 189, 255, 255
 # Filter by Area.
# params.filterByArea = True
# params.minArea = 0
# params.maxArea = 5000;

# Red Range
# lower_range = np.array([125, 100, 30])
# upper_range = np.array([255, 255, 255])
# lower_range = np.array([86, 31, 4])
# upper_range = np.array([220, 88, 50])
lower_green = np.array([222,87,267])
upper_green = np.array([202,7,187])


# mask = cv2.inRange(hsv, lower_range, upper_range)
mask = cv2.inRange(hsv, lower_green, upper_green)
reversemask=255-mask
keypoints = detector.detect(reversemask)
print(keypoints)
print(len(keypoints))

cv2.imshow('image', img)
cv2.imshow('mask', mask)
cv2.imshow('reverseMsk', reversemask)


cv2.waitKey(0)
cv2.destroyAllWindows()
