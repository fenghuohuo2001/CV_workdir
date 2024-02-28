# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 双线性插值.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/10/9 14:57
@Function：
插值法第一步： 计算目标图的坐标点对应原图中的那个坐标点来填充
计算公式：   srcX = dstX * (srcWidth / dstWidth)   # 对应原图中的横坐标 = 计算目标图横坐标 * 横向扩张倍数
           srcY = dstY * (srcHeight / dstHeight) # 对应原图中的纵坐标 = 计算目标图纵坐标 * 纵向扩张倍速
问题： 计算出的srcX，srcY有可能是小数
最邻近插值法（Nearest-neighborinterpolation）:四舍五入取整，但会产生锯齿边缘。
双线性插值（Bilinear Interpolation）:利用与坐标轴平行的两条直线去把小数坐标分解到相邻的四个整数坐标点。权重与距离成反比
                                  按面积比重插值
双三次插值（Bicubic Interpolation）:与双线性插值类似，用了相邻的16个点。
但是需要注意的是，前面两种方法能保证两个方向的坐标权重和为1，但是双三次插值不能保证这点，所以可能出现像素值越界的情况，需要截断。

测试效果：就是图片放大，清晰度不高
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

def BiLinear_interpolation(img, dstH, dstW):
    srcH, srcW, _ = img.shape
    # (待填充数组， （每个轴需要填充的数值数目），（参数输入方式：（before1，after1），（beforeN，afterN）），mode)
    # 从高维到低维，在前/后增加元素，填充为常数0 constant_values=(x, y)前面用x填充，后面用y填充
    img = np.pad(img, ((0, 1), (0, 1), (0, 0)), 'constant')
    # 建立与目标图像大小一致的空图像
    retimg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            # 由于i j初始均为0 因此需要+1
            srcx = (i+1)*(srcH/dstH)-1
            srcy = (j+1)*(srcW/dstW)-1
            # 向下取整
            x = math.floor(srcx)
            y = math.floor(srcy)
            # 计算小数位
            u = srcx - x
            v = srcy - y
            # 四个角点方向的插补值计算权重 得到当前点的插值结果
            retimg[i, j] = (1-u)*(1-v)*img[x, y] + u*(1-v)*img[x+1, y] + (1-u)*v*img[x, y+1] + u*v*img[x+1, y+1]
    return retimg


im_path = "3007_text.jpg"
image = np.array(Image.open(im_path))
image2 = BiLinear_interpolation(image, image.shape[0]*8, image.shape[1]*8)
image2 = Image.fromarray(image2.astype('uint8')).convert('RGB')
image2.save('out.jpg')










