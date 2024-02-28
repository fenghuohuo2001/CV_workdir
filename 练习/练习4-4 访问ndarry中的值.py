# -*- 练习4-4 -*-
"""
功能：访问二维数组中的值
作者：fenghuohuo
日期：年月日
"""
import numpy as np
m = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]],np.int32)
print(m)
print(m[1,3])   #第1行第3列，三维数组同理
print(m[:,1])   #第1列
print(m[1,:])   #第1行
print(m[0:2,0:2])   #第0行第0列到第1行第一列的矩形区域
