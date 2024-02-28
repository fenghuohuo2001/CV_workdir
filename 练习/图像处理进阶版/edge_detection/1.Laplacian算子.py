# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年11月29日
python 1.Laplacian算子.py rabbit.png
"""
import sys
import cv2
import numpy as np
from scipy import signal

def laplacian(image, _boundary='fill', _fillvalue=0):
    # 拉普拉斯卷积核
    laplacianKernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
    # laplacianKernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
    # 图像矩阵与拉普拉斯算子卷积
    i_conv_lap = signal.convolve2d(image, laplacianKernel, mode='same', boundary = _boundary, fillvalue=_fillvalue)
    return i_conv_lap

# if __name__ =="__main__":
#     if len(sys.argv) > 1:
#         image = cv2.imread(sys.argv[1], 0)    # 这一部分相当于命令行参数
#     else:
#         print("Usge:python laplacian.py imageFile")

# image = cv2.imread("rabbit.png", 0)
image = cv2.imread("3045.jpg", 0)

# image = cv2.imread("carriage.jpg", 0)   # 4:1
# image = cv2.resize(image, (2400, 600))


# 显示原图
cv2.imshow("image.jpg", image)

# 拉普拉斯算子
i_conv_lap = laplacian(image, 'symm')

# 第一种情形：对卷积结果进行阈值化处理
threshEdge = np.copy(i_conv_lap)
threshEdge[threshEdge > 0] = 255
threshEdge[threshEdge <= 0] = 0
threshEdge = threshEdge.astype(np.uint8)
cv2.imshow("threshEdge", threshEdge)
cv2.imwrite("1.Laplacian_threshEdge.png", threshEdge)

# 第二种情形：对卷积结果进行抽象化处理
asbstraction = np.copy(i_conv_lap)
asbstraction = asbstraction.astype(np.float32)
asbstraction[asbstraction >= 0] = 1.0
asbstraction[asbstraction < 0] = 1.0 + np.tanh(asbstraction[asbstraction<0])  # 双曲正切函数
cv2.imshow("asbstraction", asbstraction)
cv2.imwrite("1.Laplacian_asbstraction.png", asbstraction)

canny = cv2.Canny(image, 40, 150)
cv2.imshow("canny", canny)

# 加入去噪滤波
gas_filter = cv2.GaussianBlur(asbstraction, (5, 5), 0)
mid_filter = cv2.medianBlur(asbstraction, 5)
avg_filter = cv2.blur(asbstraction, (5, 5))

cv2.imshow("gas", gas_filter)
cv2.imshow("mid", mid_filter)
cv2.imshow("avg", avg_filter)

cv2.waitKey(0)
cv2.destroyAllWindows()
