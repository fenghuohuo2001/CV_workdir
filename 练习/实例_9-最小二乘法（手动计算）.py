# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年11月29日
"""
import numpy as np
import matplotlib.pyplot as plt

def Least_squares(x,y):
    '''
    公式：（x - x_）*（y - y_）/((x - x_)**2)
    '''
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(50):
        k = (x[i]-x_)*(y[i]-y_)     # k是斜率
        m += k
        p = np.square(x[i]-x_)
        n += p
    a = m/n
    b = y_ - a*x_
    return a,b

def fit(points):
    '''
    公式： (sum(xy) - n * (x_) * (y_)) / (sum(x**2) - n*(x_**2))
    '''
    m = len(points)
    x_mean = np.mean(points[:, 0])
    sum_xy = 0
    sum_x_square = 0
    sum_delta = 0
    for i in range(m):
        x = points[i, 0]
        y = points[i, 1]
        sum_xy += y*(x-x_mean)
        sum_x_square += x**2
    # 计算拟合的斜率
    k = sum_xy/(sum_x_square+m*(x_mean**2))

    for i in range(m):
        x = points[i, 0]
        y = points[i, 1]
        sum_delta += (y - k * x)
    b = sum_delta / m
    return k, b

# loss function
def loss(a, b, points):
    total_cost = 0
    m = len(points)
    for i in range(m):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - a*x - b)**2
    return total_cost/m