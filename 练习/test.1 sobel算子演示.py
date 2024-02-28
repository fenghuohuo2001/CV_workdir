# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年11月29日
"""
import cv2
import os
import numpy as np


def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

img = cv2.imread("xiaogongjian.png", 0)
cv2.imshow("src", img)
# img_gauss = cv2.GaussianBlur(img, (9, 9), 3)
# cv2.imshow("gas", img_gauss)

# kernel = np.ones((6, 6), np.uint8)
# img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
# cv2.imshow("open", img_open)

# img_diff = cv2.absdiff(img, img_open)
# cv2.imshow("diff", img_diff)
img_sobel = sobel(img)
cv2.imshow("sobel", img_sobel)


cv2.waitKey(0)
cv2.destroyAllWindows()

