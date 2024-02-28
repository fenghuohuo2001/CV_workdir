# -*- 练习17 -*-
"""
功能：以（0，0）为中心放大缩小
作者：fenghuohuo
日期：2021年6月10日
"""
import numpy as np

sx = input("请输入沿x轴放大倍数：")
sy = input("请输入沿y轴放大倍数：")

z = np.array([1,2,1],np.int32)
A = np.array([[sx,0,0],[0,sy,0],[0,0,1]],np.float32)
result = np.dot(A , z)  #矩阵乘法 点乘才是直接用 *

print(z)
print(A)
print(result)