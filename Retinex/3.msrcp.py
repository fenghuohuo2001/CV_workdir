"""
@Name: 3.msrcp.py
@Auth: Huohuo
@Date: 2023/3/14-15:59
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np
from utils import show

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

def colorRestoration(img, alpha, beta):
    color_restoration = np.zeros(img.shape)
    for i in range(img.shape[2]):
        img_sum = np.sum(img[:, :, i])
        color_restoration[:, :, i] = beta * (np.log10(alpha * img[:, :, i]) - np.log10(img_sum))
    return color_restoration



def msrcp(img):
    Int = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3
    msr_img = msr(Int)
    balance = gray_stretch(msr_img)
    img_msrcp = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            B = max(img[i, j, 0], img[i, j, 2], img[i, j, 3])
            for c in range(img.shape[2]):
                A = min(255/B, balance[i, j, c]/Int[i, j, c])
                img_msrcp[i, j, c] = A * img[i, j, c]
    return img_msrcp

def adjust_gamma(new_imgs, gamma=10):
    board = np.zeros(new_imgs.shape)
    for i in range(new_imgs.shape[2]):
        board[:, :, i] = (new_imgs[:, :, i] - np.min(new_imgs[:, :, i])) / (np.max(new_imgs[:, :, i]) - np.min(new_imgs[:, :, i])) * 255
        board[:, :, i] = np.power(new_imgs[:, :, i], gamma)
    result = np.clip(board, 0, 255)
    return result

def gray_stretch(img_gray_stretch):
    for i in range(img_gray_stretch.shape[2]):
        img_gray_stretch[:, :, i] = (img_gray_stretch[:, :, i] - np.min(img_gray_stretch[:, :, i])) / (np.max(img_gray_stretch[:, :, i]) - np.min(img_gray_stretch[:, :, i])) * 255
    # 通过线性变换将数据转换成8位[uint8]
    log_uint8 = cv2.convertScaleAbs(img_gray_stretch)
    return log_uint8

img = cv2.imread("data/img.png")

img_msrcr = msrcp(img)
cv2.imshow("img_msrcr", img_msrcr)
# img_gamma = adjust_gamma(img_msrcr)
# cv2.imshow("img_gamma",img_gamma)
# img_stretch= gray_stretch(img_msrcr)
# img_stretch= gray_stretch(img_msrcr)
# cv2.imwrite("MSRCR_result/img_MSRCR.png", img_msrcr*255)
# cv2.imwrite("MSRCR_result/img_gamma.png", img_gamma*255)
# cv2.imwrite("MSRCR_result/img_stretch.png", img_stretch*255)
cv2.waitKey(0)


