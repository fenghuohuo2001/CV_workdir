# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 4.canny+retinex.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/7/3 11:32
@Function：
"""
import cv2

image = cv2.imread("SSR-gray.jpg", 0)
image = cv2.GaussianBlur(image, (5, 5), 0)
canny = 255-cv2.Canny(image, 10, 150)

cv2.imshow("canny", canny)
cv2.imwrite("canny-ssr.jpg", canny)
cv2.waitKey(0)
cv2.destroyWindow()