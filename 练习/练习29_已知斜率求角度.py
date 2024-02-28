# -*- 练习29 -*-
"""
功能：求角度
作者：fenghuohuo
日期：2021年10月21日
"""
import numpy as np

x1 = int(input("输入第一个x1："))
y1 = int(input("输入第一个y1："))
x2 = int(input("输入第一个x2："))
y2 = int(input("输入第一个y2："))

if x2 - x1 == 0:
    print("直线是竖直的")
elif y2 - y1 == 0:
    print("直线是水平的")
else:
    # 计算斜率
    k = -(y2 - y1)/(x2 - x1)
    # 求正反切，再将得到的弧度转换为角度
    angle = np.arctan(k) * 180/np.pi
    print("直线的倾斜角度为：" + str(angle) + "°")


