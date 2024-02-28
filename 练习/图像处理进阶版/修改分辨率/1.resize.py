# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.resize.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/11/10 13:53
@Function：
"""
import cv2

img = cv2.imread("4815.jpg")
re_img = cv2.resize(img, (640, 640))
cv2.imshow("resize", re_img)
cv2.waitKey(0)
cv2.destroyWindow()