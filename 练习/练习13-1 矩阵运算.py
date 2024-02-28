# -*- 练习13-1 -*-
"""
功能：矩阵的加，减，点乘，点除
作者：fenghuohuo
日期：2021年6月7日
"""
import numpy as np
import cv2

src1 = np.array([[23,123,90],[100,250,0]],np.uint8)
src2 = np.array([[125,150,60],[100,10,40]],np.uint8)
dst1 = src1 + src2  #加法
print(dst1)
print(dst1.dtype)

dst2 = cv2.add(src1,src2,dtype=cv2.CV_32F)  #加法
print(dst2)
print(dst2.dtype)

dst3 = src1 - src2  #减法
print(dst3)
print(dst3.dtype)

dst4 = src1 * src2  #点乘
print(dst4)
print(dst4.dtype)

dst5 = src2 / src1  #点除(需要将src1中改为float32，就不会报错，因为40/0=0)
print(dst5)
print(dst5.dtype)
