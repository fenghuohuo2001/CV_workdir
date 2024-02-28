# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年11月29日
"""
1 import cv2 as cv
2 import numpy as np
3 img = cv.imread("test.jpg",0)
4 _,contours,_ = cv.findContours(img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
5 cnt = contours[0]
6 rect = cv.minAreaRect(cnt)#这里得到的是旋转矩形
7 box = cv.boxPoints(rect)#得到端点
8 box = np.int0(box)#向下取整