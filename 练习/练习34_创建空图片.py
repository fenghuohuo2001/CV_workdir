# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：年月日
"""
import cv2
import numpy as np

img = np.zeros((10, 10), np.uint8)
# 浅灰色背景
img.fill(200)       # 它的原理是把那一块单元赋成指定的值，也就是说任何值都可以
cv2.imshow('img', img)
cv2.waitKey(0)
