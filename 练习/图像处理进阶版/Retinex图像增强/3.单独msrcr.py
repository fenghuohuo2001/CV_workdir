# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 3.单独msrcr.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/7/3 17:24
@Function：
"""
import numpy as np
import cv2

def singleScaleRetinexProcess(img, sigma):
    temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(temp == 0, 0.01, temp)
    retinex = np.log10(img + 0.01) - np.log10(gaussian)
    return retinex

def multiScaleRetinexProcess(img, sigma_list):
    retinex = np.zeros_like(img * 1.0)
    for sigma in sigma_list:
        retinex = singleScaleRetinexProcess(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration

def multiScaleRetinexWithColorRestorationProcess(img, sigma_list, G, b, alpha, beta):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinexProcess(img, sigma_list)
    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)
    return img_msrcr

def touint8(img):
    for i in range(img.shape[2]):
        img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / \
                       (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255
    img = np.uint8(np.minimum(np.maximum(img, 0), 255))
    return img

def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        # counts是新列表unique元素在旧列表中第一次出现的位置，并以列表形式储存在s中
        unique, counts = np.unique(img[:, :, i], return_counts=True)    # 去除其中重复的元素
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img

def MSRCR(img, sigma_list=[15, 80, 250], G=5, b=25, alpha=125, beta=46, low_clip=0.01, high_clip=0.99):
    msrcr = multiScaleRetinexWithColorRestorationProcess(img, sigma_list, G, b, alpha, beta)
    msrcr = touint8(msrcr)
    msrcr = simplestColorBalance(msrcr, low_clip, high_clip)
    return msrcr


def main():

    image = 255-cv2.imread("src.jpg")

    msrcr = MSRCR(image)

    cv2.imshow("Retinex", msrcr)
    cv2.imwrite("MSRCR.jpg", msrcr)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()