# -*- 2 -*-
"""
功能：颜色规范化，最大值灰度处理
作者：fenghuohuo
日期：2021年11月8日
功能已经可以使用
"""
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    # path = "D:\\1.Desktop file\\picture\\result.png"
    path = "example1.png"
    img = cv2.imread(path)
    cv2.imshow("src", img)

    dst = sobel(img)
    result = grayTensile(dst)

    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

def grayTensile(dst):
    height = dst.shape[0]
    width = dst.shape[1]
    loop_1 = int(height/3)      # 选择对比度提高区域

    img_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    img_rect = img_gray[0:loop_1, 0:width]

    flat_gray = img_rect.reshape(width * loop_1).tolist()
    A = min(flat_gray)  # 最大灰度值
    B = max(flat_gray)  # 最小灰度值

    for i in range(loop_1):
        for j in range(width):
            img_gray[i, j] = 255/(B - A)*(img_rect[i, j] - A)+0.5
    return img_gray

main()