# -*- 实例1 -*-
"""
功能：测三角形角度
作者：fenghuohuo
日期：2021年10月21日

问题：角度测量没问题，但是要具体提取到两条相交的线上
"""
import cv2
import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt

# 1.加载图片，转为二值图
img = cv.imread('angle=30.png')
h, w = img.shape[:2]
# print(h, w)

# 灰度化
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)
# 降噪
blur = cv2.GaussianBlur(gray, (7, 7), 3)
# cv2.imshow("blur", blur)

# 二值化
# edges = cv.Canny(blur, 50, 150)
edges = cv.Canny(blur, 0, 30)
# cv2.imshow("edges", edges)

# hough直线变换
lines = cv.HoughLines(edges, 1, np.pi/180, 150)


# 将检测出来直线显示出来,注意这里线是极坐标（ρ，θ）
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))    # 这里为什么要*1000？ 答：延伸两点画直线,会使画出的直线贯穿于整张图片
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255),6)
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    if x2 - x1 == 0:
        print("直线是竖直的")
    elif y2 - y1 == 0:
        print("直线是水平的")
    else:
        # 计算斜率
        k = -(y2 - y1) / (x2 - x1)
        # 求正反切，再将得到的弧度转换为角度
        angle = float(np.arctan(k) * 180 / np.pi)
        print("直线的倾斜角度为：" + str(angle) + "°")
        cv2.putText(img, "{:.4f}deg".format(angle), (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # cv2.putText(图片img,“文本内容”,(左下角坐标),字体,字体大小,(颜色)，线条粗细，线条类型)
        # 文本标注，注意输入的坐标一定要是int
        '''
        cv.FONT_HERSHEY_SIMPLEX
        cv.FONT_HERSHEY_PLAIN
        cv.FONT_HERSHEY_DUPLEX
        cv.FONT_HERSHEY_COMPLEX
        cv.FONT_HERSHEY_TRIPLEX
        cv.FONT_HERSHEY_COMPLEX_SMALL
        cv.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv.FONT_HERSHEY_SCRIPT_COMPLEX 
        cv.FONT_ITALIC
        '''


cv2.imshow("hough",img)
cv2.waitKey(0)
cv2.destroyAllWindows()