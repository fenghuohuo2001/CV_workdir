"""
@Name: filterpy.py
@Auth: Huohuo
@Date: 2023/6/12-16:26
@Desc: 
@Ver : 
code_idea
"""
import numpy as np
# -------------------------------------------
#       kalman filter 的基本实现： 预测 + 更新
# -------------------------------------------

# ---------------------------------
#             1. 初始化
# 预先设定
#           状态变量           dim_x
#           观察维度变量        dim_z
#           协方差矩阵          P            * 一般初始化为单位矩阵
#           运动形式和观察矩阵   H
# ---------------------------------

class KalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0, x=None, P=None,
                Q=None, B=None, F=None, H=None, R=None):


