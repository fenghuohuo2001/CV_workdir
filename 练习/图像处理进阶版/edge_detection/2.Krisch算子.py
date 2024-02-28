# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 2.Krisch算子.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/1 9:02
@Function：铅笔素描效果
"""
import cv2
import numpy as np
from scipy import signal


def kirsch(image, _boundary='fill', _fillvalue=0):
    # 第一步：kirsch的8个边缘卷积算子分别和图像矩阵进行卷积，然后分别取绝对值得到边缘强度
    # 存储8个方向的边缘强度
    list_edge = []
    # 图像矩阵和k1进行卷积，然后取绝对值（即得到边缘强度）
    k1 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    image_k1 = signal.convolve2d(image, k1, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(image_k1))

    # 图像矩阵和k2进行卷积，然后取绝对值（即得到边缘强度）
    k2 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
    image_k2 = signal.convolve2d(image, k2, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(image_k2))

    # 图像矩阵和k3进行卷积，然后取绝对值（即得到边缘强度）
    k3 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    image_k3 = signal.convolve2d(image, k3, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(image_k3))

    # 图像矩阵和k4进行卷积，然后取绝对值（即得到边缘强度）
    k4 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    image_k4 = signal.convolve2d(image, k4, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(image_k4))

    # 图像矩阵和k5进行卷积，然后取绝对值（即得到边缘强度）
    k5 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    image_k5 = signal.convolve2d(image, k5, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(image_k5))

    # 图像矩阵和k6进行卷积，然后取绝对值（即得到边缘强度）
    k6 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    image_k6 = signal.convolve2d(image, k6, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(image_k6))

    # 图像矩阵和k7进行卷积，然后取绝对值（即得到边缘强度）
    k7 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    image_k7 = signal.convolve2d(image, k7, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(image_k7))

    # 图像矩阵和k8进行卷积，然后取绝对值（即得到边缘强度）
    k8 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    image_k8 = signal.convolve2d(image, k8, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(image_k8))

    # 第二步，对上述8个方向上的边缘强度，在对应位置取最大值作为图像最后的边缘强度
    edge = list_edge[0]
    for i in range(len(list_edge)):
        edge = edge*(edge >= list_edge[i]) + list_edge[i] * (edge < list_edge[i])
    return edge

# image = cv2.imread("carriage.jpg", 0)
# image = cv2.resize(image, (2400, 600))

image = cv2.imread("3045.jpg", 0)

edge = kirsch(image, _boundary='symm')
# 边缘强度的灰度级显示
edge = edge.astype(np.uint8)
cv2.imshow("edge", edge)

# 素描效果
pencilSketch = 255-edge
cv2.imshow("pencilSketch", pencilSketch)

cv2.waitKey(0)
cv2.destroyAllWindows()













