# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 3.canny+msrcr.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/7/3 17:20
@Function：
"""
import cv2

image = cv2.imread("MSRCR.jpg", 0)
image = cv2.GaussianBlur(image, (5, 5), 0)
canny = cv2.Canny(image, 10, 150)

cv2.imshow("canny", canny)
cv2.imwrite("MSRCR-canny.jpg", canny)
cv2.waitKey(0)
cv2.destroyWindow()