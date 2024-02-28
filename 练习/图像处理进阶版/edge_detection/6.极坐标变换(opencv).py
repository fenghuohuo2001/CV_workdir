# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 5.极坐标变换(cmath).py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/18 13:19
@Function：
函数 cv2.cartToPolar 实现将原点移动到变换中心后的直角坐标向极坐标的转换，输入参数为直角坐标系的横坐标、纵坐标，输出为极坐标系的向量值、角度值。
函数 cv2.polarToCart 实现将原点移动到变换中心后的极坐标向直角坐标的转换，输入参数为极坐标系的向量值、角度值，输出为直角坐标系的横坐标、纵坐标。

cv2.cartToPolar(x, y[, magnitude[, angle[, angleInDegrees]]]) → magnitude, angle
cv2.polarToCart(magnitude, angle[, x[, y[, angleInDegrees]]]) → x, y

x, y：直角坐标系的横坐标、纵坐标，ndarray 多维数组，浮点型
magnitude, angle：极坐标系的向量值、角度值，ndarray 多维数组
angleInDegrees：弧度制/角度值选项，默认值 0 选择弧度制，1 选择角度制（[0,360] ）
返回值 magnitude, angle：极坐标系的向量值、角度值，ndarray 多维数组，与输入的 x, y 具有相同的尺寸和数据类型
返回值 x, y：直角坐标系的横坐标、纵坐标，ndarray 多维数组，与输入的 magnitude, angle 具有相同的尺寸和数据类型

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

x = np.float32([0, 1, 2, 0, 1, 2, 0, 1, 2]) - 1
y = np.float32([0, 0, 0, 1, 1, 1, 2, 2, 2]) - 1
n = np.arange(9)

r, theta = cv2.cartToPolar(x, y, angleInDegrees=True)
xr, yr = cv2.polarToCart(r, theta, angleInDegrees=1)
print(xr, yr)

plt.figure(figsize=(9, 5))
plt.subplot(121), plt.title("Cartesian coordinate"), plt.plot(x, y, 'o')
for i, txt in enumerate(n):
    plt.annotate(txt, (x[i], y[i]))
plt.subplot(122), plt.title("Polar coordinate"), plt.plot(r, theta, 'o')
for i, txt in enumerate(n):
    plt.annotate(txt, (r[i], theta[i]))
plt.show()

