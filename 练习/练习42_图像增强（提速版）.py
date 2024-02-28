# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年11月29日
"""
import cv2
import numpy as np

def contrast_enhancement(img):
    height = int(img.shape[0])
    width = int(img.shape[1])

    flat_gray = img.reshape(width * height).tolist()
    A = min(flat_gray)  # 最大灰度值
    B = max(flat_gray)  # 最小灰度值

    img = np.uint8(255 / (B - A) * (img - A) + 0.5)
    return img

img = cv2.imread("img_2.png", 0)
img_enhan = contrast_enhancement(img)
cv2.imshow("img_enhan", img_enhan)              # 这里输出的可能是一个数组而不是一张图片的格式，需要调试查看问题所在
cv2.waitKey(0)
cv2.destroyAllWindows()