# -*- 练习30 -*-
"""
功能：大津法自适应求阈值
作者：fenghuohuo
日期：2021年11月8日
"""
import numpy as np
import cv2
import cv2 as cv

img = cv2.imread("3.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", img_gray)
ret, img_thr = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("result", img_thr)
print(ret)
cv2.waitKey(0)
cv2.destroyAllWindows()