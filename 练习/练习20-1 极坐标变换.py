# -*- 练习20-1 -*-
"""
功能：笛卡尔坐标变为极坐标
作者：fenghuohuo
日期：2021年6月22日
"""
import cv2
import numpy as np

x = np.array([[0,1,2],[0,1,2],[0,1,2]],np.float64)-1 #x为浮点型数组
y = np.array([[0,0,0],[1,1,1],[2,2,2]],np.float64)-1 #y为与x具有相同尺寸和数据类型的数组

r , theta = cv2.cartToPolar(x, y, angleInDegrees=True)
#极坐标变换，angleInDegrees当值为True时返回值为角度，否则为弧度

print("极坐标r={}".format(r),end='\n')
print("极坐标theta={}".format(theta))