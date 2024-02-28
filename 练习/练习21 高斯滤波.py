# -*- 练习21 -*-
"""
功能：高斯滤波
作者：fenghuohuo
日期：2021年6月22日
"""
import cv2

img = cv2.imread("gaoerfu.png")
cv2.imshow("yuantu",img)
blur = cv2.GaussianBlur(img,(5,5),0)
# 用模板确定的邻域内像素的加权平均灰度值去替代模板中心像素点的值
# 0 是指自动计算标准差
# (5,5)表示高斯矩阵的长和宽都为5（必须正奇数），标准差取0 如果σ较小，那么生成的模板中心系数越大，而周围的系数越小，这样对图像的平滑效果就不是很明显
# σ越大越模糊

# cv2.medianBlur()    # 中值滤波
# cv2.blur          # 均值滤波


cv2.imshow("1lvbohou",blur)
# cv2.imshow("2lvboqian",img)
cv2.waitKey(0)
cv2.destroyAllWindows()