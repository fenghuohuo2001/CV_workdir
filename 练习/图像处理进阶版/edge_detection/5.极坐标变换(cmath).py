# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 5.极坐标变换(cmath).py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/18 13:19
@Function：
"""
import cv2
import cmath
import math


cn = complex(3, 4)



# 转化为极坐标
l, theta = cmath.polar(cn)
print(l, theta)

# 弧度转为角度
theta_pi = math.degrees(theta)
print(theta_pi)

