# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 3.最小外包圆.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/6 14:43
@Function：
"""
import cv2
import numpy as np

points = np.array([[1, 1], [5, 1], [1, 10], [5, 10], [2, 5]], np.int32)

# 计算点集的最小外包圆
circle = cv2.minEnclosingCircle(points)

# 打印结果 圆心坐标+圆的半径
print(circle)