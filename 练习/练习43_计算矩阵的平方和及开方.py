# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年11月29日
"""
import cv2
import numpy as np
import scipy
from scipy import signal
import torch

src = np.array([[3, 10, 12, 19, 256], [240, 239, 8, 7, 10], [255, 180, 78, 9, 1], [170, 200, 197, 168, 50], [2, 10, 180, 140, 140]], np.float32)

sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

G_x = signal.convolve2d(src, sobel_x, mode='same')
G_y = signal.convolve2d(src, sobel_y, mode='same')
# 边界填充为0

edgeMag = np.sqrt(np.power(G_x, 2.0) + np.power(G_y, 2.0))

print("G_x=", G_x)
print("G_y=", G_y)
print("edgeMag=", edgeMag)