# -*- 练习 -*-
"""
功能：hough圆变换顶多是检测圆
作者：fenghuohuo
日期：年月日
"""
import cv2
import numpy as np

img = cv2.imread('D:\\1.Desktop file\\picture\\circle.png', 0)
img = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100,
                           param1=100, param2=30, minRadius=250, maxRadius=400)

# circles包括横坐标，纵坐标，半径
circles = np.uint16(np.around(circles))     # 近似取整
print(circles)
for i in circles[0, :]:      # 取所有列第0行元素 circles 是一个三维数组[[[]]],降维成1维
    print(i)
    # draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
# cv2.HoughCircles(image, method, dp, minDist, circles,
#                  param1, param2, minRadius, maxRadius)
# circles 包含3个参数： 圆心的横坐标，纵坐标，半径
# image为输入图像，需要灰度图
#
# method为检测方法,常用CV_HOUGH_GRADIENT
#
# dp为检测内侧圆心的累加器图像的分辨率于输入图像之比的倒数，如dp=1，累加器和输入图像具有相同的分辨率，如果dp=2，累计器便有输入图像一半那么大的宽度和高度
#
# minDist表示两个圆之间圆心的最小距离
#
# param1有默认值100，它是method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，它表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半
# 此参数是对应Canny边缘检测的最大阈值，最小阈值是此参数的一半 也就是说像素的值大于param1是会检测为边缘
# 
# param2有默认值100，它是method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，它表示在检测阶段圆心的累加器阈值，它越小，就越可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了
# 它表示在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了

# minRadius有默认值0，圆半径的最小值
#
# maxRadius有默认值0，圆半径的最大值
'''

'''
cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
作用
根据给定的圆心和半径等画圆
参数说明
img：输入的图片data
center：圆心位置
radius：圆的半径
color：圆的颜色
thickness：圆形轮廓的粗细（如果为正）。负厚度表示要绘制实心圆。
lineType： 圆边界的类型。
shift：中心坐标和半径值中的小数位数。

举例：
import numpy as np
import cv2
img = np.zeros((200,200,3),dtype=np.uint8)
cv2.circle(img,(60,60),30,(0,0,255))
cv2.imshow('img',img)
'''