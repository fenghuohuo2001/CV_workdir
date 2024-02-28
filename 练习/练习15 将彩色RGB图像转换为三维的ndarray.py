# -*- 练习15 -*-
"""
功能：将彩色RGB图像转换为三维的ndarray
作者：fenghuohuo
日期：2021年6月8日
"""
import cv2
import sys
import numpy as np
#读取图片
img = cv2.imread('xlb2.png',cv2.IMREAD_COLOR)
cv2.imshow("img",img)   #彩色图像输出
#得到三个颜色通道
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
#显示三个颜色通道
cv2.imshow("b",b)
cv2.imshow("g",g)
cv2.imshow("r",r)
cv2.waitKey(0)
cv2.destroyAllWindows()