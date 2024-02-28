# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 2.多尺度MSR.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/7/2 16:38
@Function：
"""
import cv2
import numpy as np


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

# retinex MMR
def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


def MSR_image(image):
    scales = [3, 5, 9]  # [3,5,9] [15, 101, 301]
    b_gray, g_gray, r_gray = cv2.split(image)
    b_gray = MSR(b_gray, scales)
    g_gray = MSR(g_gray, scales)
    r_gray = MSR(r_gray, scales)
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result

if __name__ == "__main__":
    image = cv2.imread("src.jpg")
    image = 255 - image
    cv2.imshow("MRC", image)
    image_msr = MSR_image(image)
    cv2.imshow("MSR", image_msr)
    # 灰度化
    gray = cv2.cvtColor(image_msr, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray", gray)
    # cv2.imwrite("MSR-gray.jpg", gray)
    # 二值化
    th, thr_pic = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # th, thr = cv2.threshold(gray, 50, 150, cv2.THRESH_BINARY)
    cv2.imshow("thr", thr_pic)
    # cv2.imwrite("MSR-thr.jpg", thr_pic)

    cv2.waitKey(0)
    cv2.destroyWindow()