# -*- 练习20-4 -*-
"""
功能：图片极坐标变换
作者：fenghuohuo
日期：2021年6月22日
"""
import numpy as np


def polar(I,center,r,theta=(0,360),rstep=1.0,thetastep=360.0/(180*8)):

# 参数I代表输入图像，center代表极坐标变换中心，r是代表最小最大距离的二元数组
# thete代表角度范围，默认[0,360],rstep代表r的变换步长，thetastep代表角度的变换步长，默认1/4
# 对于灰度值的差值，选择最近邻差值方法

    # 得到距离的最小、最大范围
    minr,maxr = r

    #得到角度的最小、最大范围
    mintheta,maxtheta = theta

    #输出图像的高，宽
    H = int((maxr-minr)/rstep)+1
    W = int((maxtheta-mintheta)/thetastep)+1
    O = 125*np.ones((H,W),I.dtype)

    #极坐标变换
    r = np.linspace(minr,maxr,H)    #构建等差数列
    r = np.tile(r,(W,1))
    r = np.transpose(r)