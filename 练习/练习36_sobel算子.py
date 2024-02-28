# -*- 练习36 -*-
"""
功能：sobel算子的opencv实现
作者：fenghuohuo
日期：2021年11月22日
注：求导，对于连续函数就叫求导。 对于不连续函数，就是差分。
        一阶差分就是一阶导。二阶差分就是二阶导。
        图像的值是位于一个范围的整数，准确来说，应该是差分。
"""
import cv2
import numpy as np

# path = "D:\\1.Desktop file\\picture\\result.png"
path = "example1.png"
img = cv2.imread(path)
cv2.imshow("src", img)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
'''
# xy = cv2.Sobel(img, cv2.CV_16S, 1, 1)     # 两个方向同时检测，即检测斜线
# 利用Sobel方法可以进行sobel边缘检测
# img表示源图像，即进行边缘检测的图像
# cv2.CV_64F表示64位浮点数即64float。
# 这里不使用numpy.float64，因为可能会发生溢出现象。用cv的数据则会自动
# 第三和第四个参数分别是对X和Y方向的导数（即dx,dy），对于图像来说就是差分，这里1表示对X求偏导（差分），0表示不对Y求导（差分）。其中，X还可以求2次导。
# 注意：对X求导就是检测X方向上是否有边缘。
# 第五个参数ksize是指核的大小。
# 理解：在3*3的核内 X方向右边像素值减去左边像素值 // Y方向下面像素值减去上面像素值

# 在经过处理后，需要用convertScaleAbs()函数将其转回原来的uint8形式，否则将无法显示图像，而只是一副灰色的窗口。
# cv2.convertScaleAbs(InputArray src, OutputArray dst, double alpha=1, double beta=0)
# InputArray src, OutputArray dst, double alpha=1, double beta=0
# 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint8类型的图片
'''
absX = cv2.convertScaleAbs(x)   # 将得到的负数取绝对值 并转换成一个无符号8位类型
absY = cv2.convertScaleAbs(y)   # 注意 xy分开计算之后再合并，要比同时计算xy效果要好

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# addWeighted()函数是将两张相同大小，相同类型的图片（叠加）线性融合的函数，可以实现图片的特效。
# cv2.addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype=-1)
'''
src1	需要加权的第一个数组，常常填一个Mat
alpha	第一个数组的权重
src2	第二个数组，需要和第一个数组拥有相同的尺寸和通道数
beta	第二个数组的权重值，值为1-alpha
gamma	一个加到权重总和上的标量值，可以理解为加权和后的图像的偏移量
dst	输出的数组，和输入的两个数组拥有相同的尺寸和通道数。dst = src1[I] * alpha + src2[I] * beta + gamma
dtype	可选，输出阵列的深度，有默认值-1。当两个输入数组具有相同深度时，这个参数设置为-1（默认值），即等同于src1.depth()。
'''


cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("result", dst)
cv2.imwrite("D:\\1.Desktop file\\picture\\img.png", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()