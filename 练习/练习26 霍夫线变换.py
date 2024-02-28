# -*- 练习26 -*-
"""
功能：霍夫线变换
作者：fenghuohuo
日期：2021年9月16日
"""

import cv2
import numpy as np

img = cv2.imread("xianbianhuan.png")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,50,200)

# 概率霍夫变换函数
lines = cv2.HoughLines(edges,1,np.pi/180,30)

lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=30,maxLineGap=200)
'''
概率霍夫变换函数
HoughLinesP(image, rho, theta, threshold)
image：二值图（边缘检测器的输出）；
rho：距离分辨率，以像素为单位；一般用1 像素
theta：角度分辨率，单位为弧度；一般用np.pi/180度
threshold：交点的最小曲线数量大于阈值，才被视为一条直线。
maxLineGap：这是一个可选参数，表示两个线段之间的gap(缺口)小于该值，则进行连接
'''


# 得到所有线段的端点
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0,255,0))

cv2.imshow('img', img)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()