# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : createGraphSegmentation.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/9/11 9:59
@Function：

sigma  对原图像进行高斯滤波去噪
k      控制合并后的区域的数量,但不是区域数量
        k设置了一个观测尺度，因为较大的k会导致对较大分量的偏好。但是请注意，k不是最小分量大小。
        k越大 越偏向于检测大尺寸，即合并区域越少
min:   后处理参数，分割后会有很多小区域，当区域像素点的个数小于min时，选择与其差异最小的区域合并
input  输入图像（PPM格式）
output 输出图像（PPM格式）

"""
import numpy as np
import cv2 as cv
import random
#需要图像分割的图片
img_path = r"D:\WorkDirectory\school_project\dragline_detection\stay_cable_photo\J16\dirt\64388mm\left_back-2022-12-14-07-25-33-64388mm-65740mm.jpeg"
src = cv.imread(img_path)
src = cv.resize(src, (int(src.shape[1]/2), int(src.shape[0]/2)))
# src = cv.imread('fire.png')
# src = cv.imread('chip.jpg')
# 调用方法
segmentator = cv.ximgproc.segmentation.createGraphSegmentation(sigma=0.3, k=300, min_size=5000)

segment = segmentator.processImage(src)
# 返回的是每个像素点种类序号

seg_image = np.zeros(src.shape, np.uint8)
print(np.max(segment))
for i in range(np.max(segment)):
  # 將第 i 個分割的座標取出
  y, x = np.where(segment == i)

  # 隨機產生顏色
  color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]

  # 設定第 i 個分割區的顏色
  for xi, yi in zip(x, y):
    seg_image[yi, xi] = color

# 將原始圖片與分割區顏色合併
result = cv.addWeighted(src, 0.3, seg_image, 0.7, 0)

# 顯示結果
cv.imwrite("waterresult.jpg", result)
# cv.imshow("src", src)
cv.imshow("Result", result)
cv.waitKey(0)
cv.destroyAllWindows()
