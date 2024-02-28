# -*- 练习16-4 -*-
"""
功能：缩小-平移-旋转
作者：fenghuohuo
日期：2021年6月21日
"""

import numpy as np
import cv2
import sys
import math

#主函数


image = cv2.imread('image.png',cv2.IMREAD_GRAYSCALE)
cv2.imwrite("image.png",image,[int(cv2.IMWRITE_EXR_COMPRESSION),9])
'''使用函数cv2.imwrite(file，img，num)保存一个图像。第一个参数
        是要保存的文件名，第二个参数是要保存的图像。可选的第三个
        参数，它针对特定的格式：对于JPEG，其表示的是图像的质量，
        用0 - 100的整数表示，默认95;对于png ,第三个参数表示的是压缩
        级别。默认为3.
         cv2.imwrite('1.png',img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
         cv2.imwrite('1.png',img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    '''#cv2.imwrite用法
#原图高、宽
h , w = image.shape[ : 2]
#仿射变换矩阵，缩小2倍
A1 = np.array([[0.5,0,0],[0,0.5,0]],np.float32)
d1 = cv2.warpAffine(image,A1,(w , h),borderValue=125)
'''
    cv2.warpAffine()函数主要是利用变换矩阵M对图像进行如旋转、仿射、平移等变换
    cv2.warpAffine(src, M, dsize,dst,flags,borderMode,borderValue) → dst
    src	    输入图像
    M	    变换矩阵，一般反映平移或旋转的关系，为InputArray类型的2×3变换矩阵。
    dsize	输出图像的大小
    flags	插值方法的组合（int 类型）
    borderMode	边界像素模式（int 类型）
    borderValue	边界填充值; 默认情况下，它为0，也就是边界填充默认是黑色。
    其中flags表示插值方式，有以下取值

    flags取值	含义
    cv2.INTER_LINEAR	线性插值(默认)
    cv2.INTER_NEAREST	最近邻插值
    cv2.INTER_AREA	区域插值
    cv2.INTER_CUBIC	三次样条插值
    cv2.INTER_LANCZOS4	Lanczos插值
'''#cv2.warpAffine 用法

#先缩小2倍，再平移
A2 = np.array([[0.5,0,w/4],[0,0.5,h/4]],np.float32)
d2 = cv2.warpAffine(image,A2,(w , h),borderValue=125)

#在d2的基础上，绕图像中心点旋转
A3 = cv2.getRotationMatrix2D((w/2.0, h/2.0) , 30 , 1 )
        #第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
d3 = cv2.warpAffine(d2,A3,( w , h ),borderValue=125)
'''
cv2.rotate（image，cv2.ROTATE_90_COUNTERCLOCKWISE）图像旋转
ROTATE_90_COUNTERCLOCKWISE  顺时针270°
ROTATE_90_CLOCKWISE         顺时针90°
ROTATE_180                  顺时针180°
'''#旋转函数rotate

cv2.imshow("image",image)
cv2.imshow("d1",d1)
cv2.imshow("d2",d2)
cv2.imshow("d3",d3)
cv2.waitKey(0)
cv2.destroyAllWindows()