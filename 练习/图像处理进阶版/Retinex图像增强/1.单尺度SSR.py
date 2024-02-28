# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.单尺度SSR.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/7/2 8:59
@Function：
"""


# SSR
import cv2
import numpy as np

# 将图像中的0像素点值用min像素点值代替，去除0元素
def replaceZeroes(data):
    # 用于得到数组array中非零元素的位置（数组索引）的函数
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    # L_blur = cv2.medianBlur(src_img, s- ize)
    # L_blur = cv2.blur(src_img, (size, size))
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


# 对三通道进行ssr
def SSR_image(image):
    size = 3
    # 通道拆分
    b_gray, g_gray, r_gray = cv2.split(image)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    # 通道合并
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result


if __name__ == "__main__":
    image = cv2.imread("src.jpg")
    image = 255 - image
    cv2.imshow("SRC", image)
    image_ssr = SSR_image(image)
    cv2.imshow("SSR", image_ssr)
    # 灰度化
    gray = cv2.cvtColor(image_ssr, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray", gray)
    # cv2.imwrite("SSR-gray.jpg", gray)
    # 二值化
    th, thr_pic = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # th, thr = cv2.threshold(gray, 50, 150, cv2.THRESH_BINARY)
    cv2.imshow("thr", thr_pic)
    # cv2.imwrite("SSR-thr.jpg", thr_pic)

    cv2.waitKey(0)
    cv2.destroyWindow()
