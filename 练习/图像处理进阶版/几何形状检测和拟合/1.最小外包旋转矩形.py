# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.最小外包旋转矩形.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/6 14:06
@Function：这里的旋转矩形是通过中心点坐标，尺寸和旋转角度来定义
"""
import cv2
import numpy as np


# opencv2X 输入点集有2种形式，N*2的二维ndarray，N*1*2的三维ndarray 数据类型只能为int32 或 float32
# 点集
points = np.array([[1, 1], [5, 1], [1, 10], [5, 10], [2, 5]], np.int32)

# 第二种点集形势
# points = np.array([[[1, 1]], [[5, 1]], [[1, 10]], [[5, 10]], [[2, 5]]], np.int32)

# 计算点集最小外包旋转矩形
ratateRect = cv2.minAreaRect(points)

print(ratateRect)