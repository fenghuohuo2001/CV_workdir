"""
@Name: cv_test.py
@Auth: Huohuo
@Date: 2023/2/5-17:29
@Desc: 
@Ver : 
"""
import cv2
import numpy as np

img = cv2.imread("zhj.jpg")

# cv2.imshow("img", img)
# img = cv2.resize(img, (1512*10, 1161*10))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.GaussianBlur(img, (3, 3), 0)
th, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# th, result = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

# result = cv2.Canny(result, 0, 255)
# morph_kernel = np.ones((3, 3), np.uint8)
# result = cv2.morphologyEx(result, cv2.MORPH_OPEN, morph_kernel, 1)

# result = cv2.resize(result, (1512, 1161))
cv2.imshow("gas", result)
# cv2.imwrite("gas.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()