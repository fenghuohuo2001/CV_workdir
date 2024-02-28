# -*- 练习13-3 -*-
"""
功能：幂指数和开平方运算
作者：fenghuohuo
日期：2021年6月7日
"""
import numpy as np

src = np.array([[25,40],[10,100]],np.uint8)
dst1 = np.power(src,2)      #对src中每一个数值进行幂指数运算
print(dst1)

dst2 = np.power(src,2.0)
print(dst2)     #幂指数数据类型对返回结果有很大影响，为不损失精度，设为浮点型即可