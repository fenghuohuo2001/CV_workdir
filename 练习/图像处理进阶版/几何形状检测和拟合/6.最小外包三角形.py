# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 6.最小外包三角形.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/6 15:43
@Function：
"""

import cv2
import numpy as np

if __name__ == "__main__":
    points = np.array([[[1, 1]], [[5, 1]], [[1, 10]], [[5, 10]], [[2, 5]]], np.float32)
    # 最小外包三角形
    area, triangle = cv2.minEnclosingTriangle(points)
    # 打印三角形面积
    print(area)
    # 打印三角形三个顶点
    print(triangle)
    print(triangle.shape)
    print(triangle.dtype)

    # 黑色图板
    s = 30
    I = np.zeros((s, s), np.uint8)

    # 绘制三角形
    k = int(triangle.shape[0])
    for i in range(k-1):
        cv2.line(I, (int(triangle[i, 0, 0]), int(triangle[i, 0, 1])), (int(triangle[i+1, 0, 0]), int(triangle[i+1, 0, 1])), 255, 1)
    cv2.line(I, (int(triangle[k - 1, 0, 0]), int(triangle[k - 1, 0, 1])), (int(triangle[0, 0, 0]), int(triangle[0, 0, 1])), 255, 1)

    cv2.imshow("I", I)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

