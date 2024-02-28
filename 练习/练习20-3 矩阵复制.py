# -*- 练习20-3 -*-
"""
功能：矩阵复制
作者：fenghuohuo
日期：2021年6月22日
"""
import cv2
import numpy as np

a = np.array([[1,2],[3,4]])
b = np.tile(a,(2,3))   #将a分别在垂直方向上复制2次 水平方向上复制3次

print("复制前的矩阵a=","{}".format(a),sep='\n')
print("复制后的矩阵b=","{}".format(b),sep='\n')