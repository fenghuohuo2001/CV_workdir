"""
@Name: getgaborker.py
@Auth: Huohuo
@Date: 2023/7/4-15:10
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

# -----------------------------------------
#          定义Gabor滤波器参数
# -----------------------------------------
def getgaborkernel():
    ksize = (32, 32)  # 滤波器的大小
    sigma = 5.0  # 高斯核的标准差
    theta = np.pi / 1  # Gabor滤波器的方向
    lambd = 10.0  # 波长（与频率相关）
    gamma = 0.5  # 空间纵横比（椭圆度）
    psi = 0  # 相位偏移
    ktype = cv2.CV_32F  # 输出核的数据类型

    gabor_kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)

    return gabor_kernel

