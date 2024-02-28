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

def SSR(src, ker_size, sigma):
    # calculate the gaussian filter result : L(x,y)
    img_L = cv2.GaussianBlur(src, (ker_size, ker_size), sigma)
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
    board1 = np.zeros(img.shape)
    for sigmai in sigma_list:
        board = SSR(img, 0, sigmai)
        board1 += (board / 3)       # 这部分数值计算有问题
    return board


def adjust_gamma(new_imgs, gamma=10):
    board = np.zeros(new_imgs.shape)
    for i in range(new_imgs.shape[2]):
        # board[:, :, i] = (new_imgs[:, :, i] - np.min(new_imgs[:, :, i])) / (np.max(new_imgs[:, :, i]) - np.min(new_imgs[:, :, i])) * 255
        board[:, :, i] = np.power(new_imgs[:, :, i], gamma)
    result = np.clip(board, 0, 255)
    return result


img = cv2.imread("data2/3848.jpg")
# -----------------msr------------------
img_msr = msr(img)
cv2.imshow("img_msr", img_msr)
# -----------------retinex------------------
img_gamma = adjust_gamma(img_msr)
cv2.imshow("img_gamma",img_gamma)

# cv2.imwrite("MSR_result/img_MSR.png", img_msr*255)
# cv2.imwrite("MSR_result/img_gamma.png", img_gamma*255)
cv2.waitKey(0)




















