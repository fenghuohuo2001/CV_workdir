# -*- 练习13-2 -*-
"""
功能：矩阵减法
作者：fenghuohuo
日期：2021年6月7日
"""
import numpy as np

src1 = np.array([[1,2,3],[4,5,6]],np.uint8)
src2 = np.array([[6,5],[4,3],[2,1]],np.uint8)
src3 = np.array([[6,5],[4,3]],np.uint8)

dst1 = np.dot(src1,src2)     #乘法
print(dst1)

dst2 = np.log(src2)     #对数运算，指数运算用exp
print(dst2)
