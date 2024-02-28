# -*- 实例 -*-
"""
功能：测量角度
作者：fenghuohuo
日期：2021年10月25日
"""
import math

import cv2
import cv2 as cv
import numpy as np

# 加载图片
img_Original = cv.imread('qiangjiao.jpg')
h, w = img_Original.shape[:2]
img_resize = cv2.resize(img_Original, (int(w/2), int(h/2)), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow("img_Original", img_resize)

# 图片预处理
gray = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

blur = cv.GaussianBlur(gray, (9, 9), 3)
cv2.imshow("blur", blur)

canny = cv.Canny(blur, 90, 150)
cv2.imshow("canny", canny)

# hough直线变换
lines = cv.HoughLines(canny, 1, np.pi/180, 176)

# 确定图片中最大直线长度
L = round(math.sqrt(pow(h/2-1,2.0)+pow(w/2-1,2.0))+1)
L1 = L/2
print(L)
# 将检测出来直线显示出来,注意这里线是极坐标（ρ，θ）
i = 1   # 用来确定直线检测顺序
for line in lines:
    rho, theta = line[0]
    print(theta*180/np.pi)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = int(a * rho)
    y0 = int(b * rho)
    x1 = int(x0 + L * (-b))    # 此处的数值要根据实际情况更改
    y1 = int(y0 + L * a)
    x2 = int(x0 - L * (-b))
    y2 = int(y0 - L * a)
    x_mid = int((x0 + x2)/2)
    y_mid = int((y0 + y2)/2)
    cv.line(img_resize, (x1, y1), (x2, y2), (0, 0, 255),2)
    print(i)
    while i>1:
        break
    cv2.putText(img_resize, "{:.1f}".format(i), (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    i += 1
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
        #cv2.putText(img_resize, "{:.1f}deg".format(angle), (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


cv2.imshow("angle", img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()