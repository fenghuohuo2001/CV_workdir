# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 3.canny.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/7/3 15:57
@Function：
"""
import cv2

# image = cv2.imread("SSR-gray.jpg", 0)
image = cv2.imread("MSRCR.jpg", 0)
image = cv2.GaussianBlur(image, (5, 5), 0)
canny = 255-cv2.Canny(image, 80, 150)

cv2.imshow("canny", canny)
cv2.imwrite("canny-MSRCR.jpg", canny)
cv2.waitKey(0)
cv2.destroyWindow()