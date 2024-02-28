# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1。图像加法.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/27 17:00
@Function：
"""
# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
img = cv2.imread('gray.jpg')
test = cv2.imread('test.jpg')

# 方法一：Numpy加法运算
result1 = img + test

# 方法二：OpenCV加法运算
result2 = cv2.add(img, test)

# 显示图像
cv2.imshow("original", img)
cv2.imshow("result1", result1)
cv2.imshow("result2", result2)

# 等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
