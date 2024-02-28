"""
@Name: 2.高斯滤波.py
@Auth: Huohuo
@Date: 2023/3/12-16:21
@Desc: 
@Ver : 
code_idea
"""

import numpy as np
import cv2

'''
kernel_size
sigma： 值越大，卷积核区域中，越边缘部分对滤波结果影响越大
        值越小，卷积核区域中，越边缘部分对滤波结果影响越小
'''
def gaussian_filter(img, kernel_size=3, sigma=1.3):
    # 首先判断img通道数
    if len(img.shape) == 3:
        H, W, C = img.shape
        # 若channel<3,即灰度图，就将图片扩展出通道维度，channel=1
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    # zero_padding
    pad = kernel_size // 2  # 取商
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float32)
    # 将原图片填入
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float32)
    # prepare kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for x in range(-pad, -pad + kernel_size):
        for y in range(-pad, -pad + kernel_size):
            # 定义高斯卷积核
            kernel[y + pad, x + pad] = np.exp(-(x**2 + y**2)/(2 * (sigma ** 2)))
    kernel /= (2 * np.pi * sigma * sigma)
    kernel /= kernel.sum()

    temp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad +y, pad + x, c] = np.sum(kernel * temp[y:y+kernel_size, x:x+kernel_size, c])
    out = np.clip(out, 0, 255)  # 限制上下限
    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)
    return out

img = cv2.imread("data/crop_0.png")
# opencv
img_gaussian_opencv = cv2.GaussianBlur(img, (15, 15), 1.3)
img_gaussian_myself = gaussian_filter(img, kernel_size=15, sigma=1.3)
cv2.imshow("src", img)
cv2.imshow("img_gaussian_opencv", img_gaussian_opencv)
cv2.imshow("img_gaussian_myself", img_gaussian_myself)
cv2.waitKey(0)



