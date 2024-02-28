# -*- 练习20-2 -*-
"""
功能：极坐标转换为笛卡尔坐标
作者：fenghuohuo
日期：2021年6月22日
"""
import cv2
import numpy as np

angle = np.array([[30,31],[30,31]],np.float32)
r = np.array([[10,10],[11,11]],np.float32)

x,y = cv2.polarToCart(r,angle,angleInDegrees=True)
#此时变换中心为（0，0），但是实际变换中心为（-12，15）

x+=-12  #改变变换中心
y+=15

print("x={}".format(x),end='\n')
print("y={}".format(y))
