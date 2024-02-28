# -*- 练习4 构建darray对象 -*-
"""
功能：构建darray对象
作者：fenghuohuo
日期：2021年06月03日
"""
import numpy as np
z = np.zeros((2,4),np.uint8) #构造2行4列的矩阵
print(z)

x = np.ones((4,4),np.int32)
print(x)
print(x[0:2,0:2])

sum = x + z
print(sum)