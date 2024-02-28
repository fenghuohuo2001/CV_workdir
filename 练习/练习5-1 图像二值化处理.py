# -*- 练习5 图像二值化处理 -*-
"""
功能：图像二值化处理
作者：fenghuohuo
日期：2021年6月7日
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('scratch.png', 0)
# global thresholding
# 大于127的部分设置255
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  #  当我们指定了阈值参数thresh，第一个返回值ret就是我们指定的thresh。换句话说，我们可以不指定阈值参数thresh。
# Otsu's thresholding
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# Otsu's thresholding
# 阈值一定要设为 0 !
ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1, img, 0, th2, img, 0, th3]
titles = [
    'Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
    'Original Noisy Image', 'Histogram', "Adaptive Thresholding",
    'Original Noisy Image', 'Histogram', "Otsu's Thresholding"
]
# 这里使用了 pyplot 中画直方图的方法, plt.hist, 要注意的是它的参数是一维数组
# 所以这里使用了( numpy ) ravel 方法,将多维数组转换成一维,也可以使用 flatten 方法
# ndarray.flat 1-D iterator over an array.
# ndarray.flatten 1-D array copy of the elements of an array in row-major order.
for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
plt.show()