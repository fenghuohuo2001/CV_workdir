# -*- 练习28 -*-
"""
功能：直线检测
作者：fenghuohuo
日期：2021年10月14日
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

# hough直线变换
lines = cv.HoughLines(edges, 1, np.pi/180, 10, 10, 90)

i=1

# 将检测出来直线显示出来,注意这里线是极坐标（ρ，θ）
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = int(a * rho)
    y0 = int(b * rho)
    x1 = int(x0 + 1000*(-b))    # 这里为什么要*1000？ 答：延伸两点画直线,会使画出的直线贯穿于整张图片
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    x_mid = int((x0 + x2) / 2)
    y_mid = int((y0 + y2) / 2)
    cv.line(img_resize,(x1,y1),(x2,y2),(0,0,255))
    cv2.putText(img_resize, "{:.0f}".format(i), (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    i+=1


cv2.imshow("hough",img_resize)

cv2.waitKey(0)
cv2.destroyAllWindows()