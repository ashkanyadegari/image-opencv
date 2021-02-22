import numpy as np
import cv2
import numpy as np
import math
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
 

img = cv2.imread('test13.png')
blur = cv2.medianBlur(img, 5)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
params = cv2.SimpleBlobDetector_Params()

# Red Range
lower_range = np.array([125, 100, 30])
upper_range = np.array([255, 255, 255])

lower_breen = np.array([30, 30, 100])
upper_breen = np.array([95, 255, 255])

# Change thresholds
params.minThreshold = 5
params.maxThreshold = 200

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)


mask = cv2.inRange(hsv, lower_range, upper_range)
mask2 = cv2.inRange(hsv, lower_breen, upper_breen)
reversemask = 255 - mask
reversemask2 = 255 - mask2

keypoints = detector.detect(reversemask)
keypoints2 = detector.detect(reversemask2)
print(keypoints)
print('red',len(keypoints))
print('blue green', len(keypoints2))

cv2.imshow('image', img)
cv2.imshow('mask', mask)
cv2.imshow('mask2',reversemask2)
cv2.imshow('reverseMsk', reversemask)



cv2.waitKey(0)
cv2.destroyAllWindows()
