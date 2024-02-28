# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年12月6日
bug是中文路径造成的
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

path = "D:/1.Desktop file/py_pro/文字识别/carriage.jpg"
img = cv2.imread(path)

kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])

output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)

plt.figure()
plt.subplot(221)
plt.title('Original Image')
plt.imshow(img[:,:,::-1])
plt.subplot(222)
plt.title('sharpen_1 Image')
plt.imshow(output_1[:,:,::-1])
plt.subplot(223)
plt.title('sharpen_2 Image')
plt.imshow(output_2[:,:,::-1])
plt.show()

