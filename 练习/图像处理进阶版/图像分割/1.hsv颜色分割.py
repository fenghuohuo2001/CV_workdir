# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.hsv颜色分割.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/17 14:51
@Function：
"""
import cv2
import numpy as np

def color_set(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # low_brown = np.array([30, 20, 10])
    # high_brown = np.array([55, 43, 46])
    low_brown = np.array([10, 0, 0])
    high_brown = np.array([150, 255, 255])

    mask = cv2.inRange(hsv, low_brown, high_brown)
    mask = 255-mask
    # cv2.imshow("mask", mask)

    '''函数很简单，参数有三个
       第一个参数：hsv指的是原图
       第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0
       第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0
       而在lower_red～upper_red之间的值变成255
    '''
    res = cv2.bitwise_and(img, img, mask=mask)  # 与运输操作就是1 & 1 = 1,其他为0
    # cv2.imshow("res", res)
    return res, mask

# img_path = r"D:\1.Desktop file\pythonProject\coursework\data\src\+\3.jpg"
img_path = r"D:\1.Desktop file\pythonProject\coursework\data\src\=\3.jpg"
img = cv2.imread(img_path)
# seg, mask = color_set(img)
# 闭运算 先膨胀 后腐蚀
kernel = np.ones((20, 20), np.uint8)
close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("close", close)

contours, hierarchy = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓数量:{len(contours)}")

areas = []
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    print(f"第{i}个轮廓面积：{area}")
    areas.append(area)

area_max = np.max(areas)
index_max = areas.index(area_max)
print(f"第{index_max}个轮廓是最大轮廓，面积为{area_max}")
para = 0.009
peri = cv2.arcLength(contours[index_max], True)
# 轮廓近似多边形角点
approxCurve = cv2.approxPolyDP(contours[index_max], peri * para, True)
draw = close.copy()
cv2.drawContours(draw, [approxCurve], -1, (0, 0, 255), 2)
'''
curve：源图像的某个轮廓；
epsilon：距离值，表示多边形的轮廓接近实际轮廓的程度，值越小，越精确；
closed：轮廓是否闭合。
最重要的参数就是 epsilon 简单记忆为：该值越小，得到的多边形角点越多，轮廓越接近实际轮廓，该参数是一个准确度参数。
该函数返回值为轮廓近似多边形的角点。
'''
print(f"轮廓近似多边形角点数量:{approxCurve.shape[0]}")

cv2.imshow("draw", draw)


cv2.waitKey(0)
cv2.destroyAllWindows()
