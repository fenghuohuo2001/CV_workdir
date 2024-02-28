# -*- 练习28-1 -*-
"""
功能：能输出端点的cv2.HoughLinesP
作者：fenghuohuo
日期：2021年10月15日
注意 实验结果显示这个并不适合于做检测
"""
import cv2
import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt

# 1.加载图片，转为二值图
img = cv.imread('zhi.jpg')
img = img[200:959,0:719]
h, w = img.shape[:2]
# print(h, w)

# 图片太大了 缩小一些方便观察
# img_resize = cv2.resize(img, (int(w/2), int(h/2)),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
img_resize = cv2.resize(img, (int(w), int(h)), fx=1, fy=1)
# cv2.imshow("img_resize", img_resize)

# 灰度化
gray = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)
# 降噪
blur = cv2.GaussianBlur(gray, (7, 7), 3)
# cv2.imshow("blur", blur)

# 二值化
# edges = cv.Canny(blur, 50, 150)
edges = cv.Canny(blur, 0, 30)
# cv2.imshow("edges", edges)

# img_erode = cv2.erode(edges,None,iterations=1)    #腐蚀
# img_dilate = cv2.dilate(img_erode,None,iterations=1)  #膨胀
# cv2.imshow("img_dilate",img_dilate)

# hough概率直线变换 lines中储存的是四个坐标点
lines = cv.HoughLinesP(edges, 1, np.pi/180, 10, 10, 90)

'''
函数解释
上述含义为 输入edges图片，像素间距为1， 角度间距为1度 

倒数第三个参数：表示检测一条直线所需最少的曲线交点（霍夫空间）
倒数第二个参数：如果检测的直线的长度小于这个设定的值就丢弃
倒数第一个参数：两段直线在同一个方向，断点之间的距离，要不要连成一条所需设置的阈值


# HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines
image参数表示边缘检测的输出图像，该图像为单通道8位二进制图像。

rho参数表示参数极径  以像素值为单位的分辨率，这里一般使用 1 像素。

theta参数表示参数极角  以弧度为单位的分辨率，这里使用 1度。

threshold参数表示检测一条直线所需最少的曲线交点。

lines参数表示储存着检测到的直线的参数对  的容器，也就是线段两个端点的坐标。

minLineLength参数表示能组成一条直线的最少点的数量，点数量不足的直线将被抛弃。

maxLineGap参数表示能被认为在一条直线上的亮点的最大距离。
'''


# 将检测出来直线显示出来,注意这里线是极坐标（ρ，θ）
'''
for x1, y1, x2, y2 in lines[i]:
    cv2.line(img_resize, (x1, y1), (x2, y2), (0, 255, 0), 2)
'''
i = 1
# 注意for已经是循环了
for line in lines:
    x1, y1, x2, y2= line[0]
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    cv.line(img_resize, (x1, y1), (x2,y2), (0, 0, 255), 2)  # 最后一个数字代表线段粗细
    cv2.putText(img_resize, "{:.0f}".format(i), (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    i += 1
cv2.imshow("hough",img_resize)

cv2.waitKey(0)
cv2.destroyAllWindows()