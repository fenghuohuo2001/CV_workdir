# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 3.高斯Laplacian算子.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/1 9:40
@Function：
"""
import cv2
import numpy as np
from scipy import signal


def creatLoGKernel(sigma, size):
    # 高斯拉普拉斯算子
    H, W = size
    r, c = np.mgrid[0:H:1, 0:W:1]

    # r -= (H-1.0)/2.0              # 这里用-=会报错
    # c -= (W-1.0)/2.0
    # a+=b中a为int32类型，
    # 这种写法会尝试将运算后的结果保存在a中，但是加法运算后的结果是一个float64的数据，而float64不能存放在int32中，所以a+=b会报错

    r = r - (H - 1) / 2
    c = c - (W - 1) / 2
    # 方差
    sigma2 = pow(sigma, 2.0)
    # 高斯拉普拉斯核
    norm2 = np.power(r, 2.0) + np.power(c, 2.0)
    LoGKernel = (norm2/sigma2 - 2) * np.exp(-norm2/(2*sigma2))
    return LoGKernel

def LoG(image, sigma, size, _boundary='symm'):
    # 构建高斯拉普拉斯卷积核
    loGKernel = creatLoGKernel(sigma, size)
    # 图像矩阵与高斯拉普拉斯卷积核卷积
    img_conv_log = signal.convolve2d(image, loGKernel, 'same', boundary=_boundary)
    return img_conv_log

image = cv2.imread("carriage.jpg", 0)
image = cv2.resize(image, (2400, 600))

# image = cv2.imread("rabbit.png", 0)

img_conv_log = LoG(image, 6, (37, 37), 'symm')
# img_conv_log = LoG(image, 6, (20, 20), 'symm')

# 边缘的二值化显示
edge_binary = np.copy(img_conv_log)
edge_binary[edge_binary > 0] = 255
edge_binary[edge_binary <= 0] = 0
edge_binary = edge_binary.astype(np.uint8)
cv2.imshow("edge_brinary", edge_binary)

cv2.waitKey(0)
cv2.destroyAllWindows()