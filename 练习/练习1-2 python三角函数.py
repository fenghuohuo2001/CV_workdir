# -*- 练习1-2 -*-
"""
功能：输入角度输出三角函数值
作者：fenghuohuo
日期：年月日
"""
import math

x = input ("输入角度：")
y = float(x)
z = math.radians(y)     #转换为弧度
s = math.cos(z)
print (s)
