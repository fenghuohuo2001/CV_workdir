# -*- 练习16-3 -*-
"""
功能：以（0，0）为旋转中心旋转
作者：fenghuohuo
日期：2021年6月10日
"""
import numpy as np
import math

a = float(input("请输入旋转的角度："))
α = math.radians(a)
a = math.cos(α)
b = -math.sin(α)
c = math.sin(α)
d = math.cos(α)

z = np.array([1,2,1],np.int32)
A = np.array([[a,b,0],[c,d,0],[0,0,1]],np.float32)
result = np.dot(A , z)  #矩阵乘法 点乘才是直接用 *

print(z)
print(A)
print(result)