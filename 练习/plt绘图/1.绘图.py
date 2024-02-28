# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.绘图.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/17 11:17
@Function：
"""
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("1.jpg")

plt.figure(figsize=(10,5)) #设置窗口大小
plt.suptitle('Multi_Image') # 图片名称
plt.subplot(2,3,1), plt.title('image')
plt.imshow(img), plt.axis('off')
plt.subplot(2,3,2), plt.title('gray')
plt.imshow(gray,cmap='gray'), plt.axis('off') #这里显示灰度图要加cmap
plt.subplot(2,3,3), plt.title('img_merged')
plt.imshow(img_merged), plt.axis('off')
plt.subplot(2,3,4), plt.title('r')
plt.imshow(r,cmap='gray'), plt.axis('off')
plt.subplot(2,3,5), plt.title('g')
plt.imshow(g,cmap='gray'), plt.axis('off')
plt.subplot(2,3,6), plt.title('b')
plt.imshow(b,cmap='gray'), plt.axis('off')

plt.show()
