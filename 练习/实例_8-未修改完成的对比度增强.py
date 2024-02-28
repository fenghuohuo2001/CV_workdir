# -*- 2 -*-
"""
功能：颜色规范化，最大值灰度处理
作者：fenghuohuo
日期：2021年11月8日
不知道为什么报错
"""
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('example1.png')
height = img.shape[0]
width = img.shape[1]
loop_1 = int(height*2/5)
loop_2 = int(height*2/5)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray", img_gray)

img_half_1 = img_gray[0:loop_1, 0:width]
cv2.imshow("half_1", img_half_1)

flat_gray_1 = img_half_1.reshape(width * loop_1).tolist()
A = min(flat_gray_1)  # 最小灰度值
B = max(flat_gray_1)  # 最大灰度值
for i in range(loop_1):
    for j in range(width):
        img_gray[i, j] = 255/(B - A)*(img_half_1[i, j] - A)+0.5

img_half_2 = img_gray[loop_1:height, 0:width]
cv2.imshow("half_2", img_half_2)

flat_gray_2 = img_half_2.reshape(width * loop_2).tolist()
C = min(flat_gray_2)  # 最小灰度值
D = max(flat_gray_2)  # 最大灰度值
for i in range(loop_2):
    for j in range(width):
        img_gray[i, j+loop_1] = 255/(D - C)*(img_half_2[i, j] - C)+0.5

cv2.imshow("src", img)
cv2.imshow("result", img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()