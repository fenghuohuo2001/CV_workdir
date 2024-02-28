"""
@Name: 1.ssr_gamma.py
@Auth: Huohuo
@Date: 2023/3/11-13:03
@Desc: 
@Ver : 
code_idea
算法流程：
高斯滤波、对数、反对数Map、图像拉伸


"""
import cv2
import numpy as np

# function : i(x, y) = L(x, y) *R(x, y)
# 输入：src、滤波半径sigma
def SSR(src, ker_size=3, sigma=1.3):
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

def gray_stretch(img_gray_stretch):
    for i in range(img_gray_stretch.shape[2]):
        img_gray_stretch[:, :, i] = (img_gray_stretch[:, :, i] - np.min(img_gray_stretch[:, :, i])) / (np.max(img_gray_stretch[:, :, i]) - np.min(img_gray_stretch[:, :, i])) * 255
    # 通过线性变换将数据转换成8位[uint8]
    log_uint8 = cv2.convertScaleAbs(img_gray_stretch)
    return log_uint8

def adjust_gamma(new_imgs, gamma=10):
    board = np.zeros(new_imgs.shape)
    for i in range(new_imgs.shape[2]):
        board[:, :, i] = (new_imgs[:, :, i] - np.min(new_imgs[:, :, i])) / (np.max(new_imgs[:, :, i]) - np.min(new_imgs[:, :, i])) * 255
        board[:, :, i] = np.power(new_imgs[:, :, i], gamma)
    result = np.clip(board, 0, 255)
    return result


img = cv2.imread("data/img.png")
img_R = SSR(img, ker_size=0, sigma=15)      # [15, 80, 250]
cv2.imshow("img_r",img_R)
# img_R = cv2.cvtColor(img_R.astype(np.float32), cv2.COLOR_BGR2GRAY)
img_stretch = gray_stretch(img_R)
img_gamma = adjust_gamma(img_R)
cv2.imshow("img_stretch",img_stretch)
cv2.imshow("img_gamma",img_gamma)

cv2.imwrite("SSR_result/img_SSR.png", img_R)
cv2.imwrite("SSR_result/img_stretch.png", img_stretch)
cv2.imwrite("SSR_result/img_gamma.png", img_gamma)
cv2.waitKey(0)