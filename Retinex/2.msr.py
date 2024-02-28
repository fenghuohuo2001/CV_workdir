"""
@Name: 2.msr.py
@Auth: Huohuo
@Date: 2023/3/13-8:56
@Desc: 
@Ver : 
code_idea
MSR是在SSR基础上发展来的，优点是可以同时保持图像高保真度与对图像的动态范围进行压缩的同时，MSR也可实现色彩增强、颜色恒常性、局部动态范围压缩、全局动态范围压缩，也可以用于X光图像增强。

为了得到更好的效果，人们又开发出所谓的多尺度视网膜增强算法（MSR， Multi-Scale Retinex），最为经典的就是3尺度的，大、中、小，既能实现图像动态范围的压缩，又能保持色感的一致性较好。同单尺度相比，该算法有在计算Log[R(x,y)]的值时步骤有所不同：

（1）需要对原始图像进行每个尺度的高斯模糊，得到模糊后的图像Li(x,y),其中小标i表示尺度数。

(2) 对每个尺度下进行累加计算

Log[R(x,y)] = Log[R(x,y)] + Weight(i)* ( Log[Ii(x,y)]-Log[Li(x,y)]);

其中Weight(i)表示每个尺度对应的权重，要求各尺度权重之和必须为1，经典的取值为等权重。

其他的步骤和单尺度的没有区别。
"""

import cv2
import numpy as np

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

def SSR(src, ker_size=3, sigma=1.3):
    # calculate the gaussian filter result : L(x,y)
    img_L = cv2.GaussianBlur(src, (ker_size, ker_size), sigma)      # sigma = 0.3*((ksize-1)*0.5-1)+0.8
    img_L = np.where(img_L == 0, 0.01, img_L)
    # --------Log[R(x,y)] = Log[I(x,y)]-Log[L(x,y)]----------
    # log_img_I = np.log10(src.copy().astype(np.float32))
    log_img_I = np.log10(src.copy())
    # log_img_L = np.log10(img_L.copy().astype(np.float32))
    log_img_L = np.log10(img_L.copy() + 0.01)
    log_img_R = log_img_I - log_img_L
    img_R = np.power(10, log_img_R)
    return img_R


def msr(img, sigma_list=[15, 80, 250]):
    board = np.zeros(img.shape)
    for sigma in sigma_list:
        board = SSR(img, ker_size=3, sigma=sigma)
    board += board / len(sigma_list)
    return board

def adjust_gamma(new_imgs, gamma=10):
    board = np.zeros(new_imgs.shape)
    for i in range(new_imgs.shape[2]):
        board[:, :, i] = (new_imgs[:, :, i] - np.min(new_imgs[:, :, i])) / (np.max(new_imgs[:, :, i]) - np.min(new_imgs[:, :, i])) * 255
        board[:, :, i] = np.power(new_imgs[:, :, i], gamma)
    result = np.clip(board, 0, 255)
    return result


img = cv2.imread("data/img.png")

img_msr = msr(img)
cv2.imshow("img_msr", img_msr)
img_gamma = adjust_gamma(img_msr)
cv2.imshow("img_gamma",img_gamma)

cv2.imwrite("MSR_result/img_MSR.png", img_msr*255)
cv2.imwrite("MSR_result/img_gamma.png", img_msr*255)
cv2.waitKey(0)




















