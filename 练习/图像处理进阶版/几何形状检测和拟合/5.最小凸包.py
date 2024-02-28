# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 5.最小凸包.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/6 14:51
@Function：

convexHulll(inputArray points, outputArray hull, bool clockwise = false, bool returnPoints = true)
hull: 构成凸包的点
clockwise: hull中的点是按顺时针分布还是逆时针分布
returnPoints: 值为true时，hull中存储坐标点，值为false时，存储这些坐标点在点集中的索引
默认： 逆时针 索引
这一部分可以用于检测五边形区域角度
"""

import cv2
import numpy as np

# 主函数
if __name__ == "__main__":
    # 400*400黑色画板
    s = 400
    I = np.zeros((s, s), np.uint8)
    # 随机生成横、纵坐标均在[100, 300)之间的坐标点
    n = 80  # 随机生成n个坐标点，每一行存储一个坐标
    points = np.random.randint(100, 300, (n, 2), np.int32)
    # 在画板上用一个小圆标出这些点
    for i in range(n):
        cv2.circle(I, (points[i, 0], points[i, 1]), 2, 255, 2)

    # 求点集point的凸包
    convexhull = cv2.convexHull(points, clockwise=False, returnPoints=True)     # 逆时针 返回值为坐标点
    # 打印凸包信息
    print(type(convexhull))
    print(convexhull.shape)
    print(convexhull)

    # 依次连接凸包的各个点,shape[0]
    k = convexhull.shape[0]
    print(k)
    for i in range(k-1):    # 有11个点 k=11 但是i是从0开始计算，所以k-1
        cv2.line(I, (convexhull[i, 0, 0], convexhull[i, 0, 1]), (convexhull[i+1, 0, 0], convexhull[i+1, 0, 1]), 255, 2)
    # 上面还差首坐标和末坐标的相连，以下是首尾相接
    cv2.line(I, (convexhull[k-1, 0, 0], convexhull[k-1, 0, 1]), (convexhull[0, 0, 0], convexhull[0, 0, 1]), 255, 2)

    # 显示图片
    cv2.imshow("I", I)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



