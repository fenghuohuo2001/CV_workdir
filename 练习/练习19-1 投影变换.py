# -*- 练习19-1 -*-
"""
功能：投影变换
作者：fenghuohuo
日期：2021年6月21日
"""
import cv2
import numpy as np
src = np.array([[0,0],[200,0],[0,200],[200,200]],np.float32)
#原坐标
dst = np.array([[100,20],[200,20],[50,70],[250,70]],np.float32)
#投影变换后依次对应的坐标
p = cv2.getPerspectiveTransform(src,dst)
#计算投影变换矩阵的函数
print('投影矩阵为:{}'.format(p))