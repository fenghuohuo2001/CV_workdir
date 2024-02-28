# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.加权平均法灰度化.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/25 17:00
@Function：
灰度化有三种方法：
最大值法(最大分量法)： R=G=B=max(R,G,B)
平均值法：R=G=B=(R+G+B)/3
加权平均法：R=G=B= a*R + b*G + c*B
"""
import cv2
import numpy as np

img = cv2.imread('lane2.png')   # BGR读入
h, w = img.shape[0:2]
# print(h, w)
gray = np.zeros((h, w), dtype=img.dtype)
gray_yellow = np.zeros((h, w), dtype=img.dtype)


for i in range(h):
    for j in range(w):
        gray[i, j] = (0.3 * int(img[i, j, 0]) + 0.59 * int(img[i, j, 1]) + 0.11 * int(img[i, j, 2]))
        # 2022.6.25 论文《基于树莓派嵌入式平台的车道线检测算法》凸显出黄色车道灰度化后特征 用lane2.png
        # 黄色RGB=255、255、0 ，BGR= 0 255 255
        # 橙黄色RGB=255、97、0 ，BGR= 0 165 255 具体查RGB表
        gray_yellow[i, j] = (0 * int(img[i, j, 0]) + 0.3 * int(img[i, j, 1]) + 0.7 * int(img[i, j, 2]))
        # 效果确实好一些

cv2.imshow('Gray', gray)
cv2.imshow('Gray_yellow', gray_yellow)
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyWindow()