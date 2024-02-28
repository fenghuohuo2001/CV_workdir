# -*- 练习16 -*-
"""
功能：平移
作者：fenghuohuo
日期：2021年6月8日
"""
import numpy as np

tx = input("请输入沿x轴平移距离：")
ty = input("请输入沿y轴平移距离：")

z = np.array([1,2,1],np.int32)
A = np.array([[1,0,tx],[0,1,ty],[0,0,1]],np.int32)
result = np.dot(A , z)  #矩阵乘法 点乘才是直接用 *

print(z)
print(A)
print(result)