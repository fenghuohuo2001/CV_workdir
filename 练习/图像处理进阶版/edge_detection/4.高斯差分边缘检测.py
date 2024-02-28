# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 4.高斯差分边缘检测.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/1 10:31
@Function：滤波效果太强了，建议用在明显边缘上
"""
import numpy as np
import cv2
from scipy import signal


def gaussConv(I, size, sigma):
    # 卷积核的高和宽
    H, W = size
    # 构建水平方向上的非归一化的高斯卷积核
    xr, xc = np.mgrid[0:1, 0:W]
    xc = xc - (W - 1)/2
    xk = np.exp(-np.power(xc, 2.0)/(2.0*pow(sigma, 2.0)))   # 原书中这一部分代码缺失
    # I与xk卷积
    I_xk = signal.convolve2d(I, xk, 'same', 'symm')
    # 构建垂直方向上的非归一化的高斯卷积核
    yr, yc = np.mgrid[0:H, 0:1]
    yr = yr - (H - 1)/2
    yk = np.exp(-np.power(yr, 2.0))
    # I_xk 与 yk 卷积
    I_xk_yk = signal.convolve2d(I_xk, yk, 'same', 'symm')
    I_xk_yk = I_xk_yk * (1.0/(2*np.pi*pow(sigma, 2.0)))
    return I_xk_yk

# 然后通过定义函数DoG实现高斯差分
def DoG(I, size, sigma, k=1.1):
    # 标准差为sigma的非归一化高斯卷积
    Is = gaussConv(I, size, sigma)
    # 标准差为k*sigma的非归一化高斯卷积
    Isk = gaussConv(I, size, k*sigma)
    # 两个高斯卷积的差分
    doG = Isk - Is
    doG = doG/(pow(sigma, 2.0)*(k - 1))
    return doG

image = cv2.imread("carriage.jpg", 0)
image = cv2.resize(image, (2400, 600))

# image = cv2.imread("rabbit.png", 0)
# image = cv2.imread("img_1.png", 0)
# image = cv2.imread("3045.jpg", 0)

cv2.imshow("src", image)
# 高斯差分边缘检测
sigma = 2
# sigma = 1

k = 1.1
# k = 0.5

size = (13, 13)
# size = (21, 21)
imageDoG = DoG(image, size, sigma, k)
cv2.imshow("imageDoG", imageDoG)
# 二值化边缘，对imageDoG进行阈值化处理
edge = np.copy(imageDoG)
edge[edge > 0] = 255
edge[edge <= 0] = 0
edge = edge.astype(np.uint8)
cv2.imshow("edge", edge)

# 图像边缘抽象化
asbstraction = -np.copy(imageDoG)
asbstraction = asbstraction.astype(np.float32)
asbstraction[asbstraction >= 0] = 1.0
asbstraction[asbstraction < 0] = 1.0 + np.tanh(asbstraction[asbstraction < 0])
cv2.imshow("asbstraction", asbstraction)

cv2.waitKey(0)
cv2.destroyAllWindows()
