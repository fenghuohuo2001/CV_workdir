# -*- 2 -*-
"""
功能：颜色规范化，最大值灰度处理
作者：fenghuohuo
日期：2021年11月8日
思路 大于0的全等与255（白），等于0附近的全=0
"""
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('scd.png')
height = img.shape[0]
width = img.shape[1]

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("img_gray", img_gray)
grayimg = np.zeros((height, width, 3), np.uint8)

for i in range(height):
    for j in range(width):
        # 获取图像RGB最大值
        gray = max(img[i, j][0], img[i, j][1], img[i, j][2])
        grayimg[i, j] = np.uint8(gray)

cv2.imshow("src", img)
cv2.imshow("gray", grayimg)

cv2.waitKey(0)
cv2.destroyAllWindows()