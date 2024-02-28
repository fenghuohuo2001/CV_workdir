# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 2.自适应阈值分割.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/8/4 17:48
@Function：
cv2.adaptiveThreshold(src, maxthresh, formula mode, thresh mode, small area area, subtract value)
自适应阈值法 会每次取图片的一小部分计算阈值，这样图片不同区域的阈值就不同，适用于明暗分布不均匀的图片
formula mode:
ADAPTIVE_THRESH_MEAN_C: 小区域内取值
ADAPTIVE_THRESH_GAUSSIAN_C：小区域内加权求和，权重是高斯核

final thresh = small area thresh - subtract value

"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("src.jpg", 0)

# 均值自适应
th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
# 高斯加权自适应
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)

# 对比结果
title = ['src', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img, th1, th2]
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(title[i], fontsize=8)
    plt.xticks([])
    plt.yticks([])
plt.show()

