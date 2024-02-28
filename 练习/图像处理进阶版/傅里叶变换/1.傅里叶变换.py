# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.傅里叶变换.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/1 15:48
@Function：先对图像矩阵进行傅里叶变换，然后进行傅里叶逆变换，接着取实部，就可以恢复原图像
"""
import cv2

# cv2.dft(src, dst, flags)
'''
  flags DFT_COMPLEX_OUTPUT 输出复数形式
        DFT_REAL_OUTPUT    只输出实部
        DFT_INVERSE        傅里叶逆变换
        DFT_SCALE          是否除以M*N
        DFT_ROWS           输入矩阵的每行进行傅里叶变换或者逆变换
'''

def fft2Image(src):
    # 得到行、列
    r, c = src.shape[:2]