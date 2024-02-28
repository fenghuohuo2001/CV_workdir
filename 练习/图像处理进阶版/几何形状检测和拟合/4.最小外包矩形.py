# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 4.最小外包矩形.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/6 14:48
@Function：
"""
import cv2
import numpy as np

points = np.array([[1, 1], [5, 1], [1, 10], [5, 10], [2, 5]], np.int32)

# 最小外包直立矩形
rect = cv2.boundingRect(points)

# 打印结果，前两个是顶点坐标，后两个是对角顶点坐标
print(rect)