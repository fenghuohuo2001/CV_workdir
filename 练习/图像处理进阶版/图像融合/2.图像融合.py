# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 2.图像融合.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/27 17:07
@Function：图像融合通常是指将2张或2张以上的图像信息融合到1张图像上，
融合的图像含有更多的信息，能够更方便人们观察或计算机处理。如下图所示，将两张不清晰的图像融合得到更清晰的图

图像融合是在图像加法的基础上增加了系数和亮度调节量。
图像加法：目标图像 = 图像1 + 图像2
图像融合：目标图像 = 图像1 * 系数1 + 图像2 * 系数2 + 亮度调节量
主要调用的函数是addWeighted，方法如下：
dst = cv2.addWeighter(scr1, alpha, src2, beta, gamma)
dst = src1 * alpha + src2 * beta + gamma
其中参数gamma不能省略
"""
# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
src1 = cv2.imread('src.jpg', 0)
src2 = cv2.imread('MSRCR.jpg', 0)
src2 = 255-src2
# 图像融合
result = cv2.addWeighted(src1, 0.3, src2, 0.7, 0)

# 显示图像
cv2.imshow("src1", src1)
cv2.imshow("src2", src2)
cv2.imshow("result", result)

# 等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
