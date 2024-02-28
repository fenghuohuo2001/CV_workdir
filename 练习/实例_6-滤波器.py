# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：年月日
"""
import cv2
import cv2 as cv

img = cv2.imread('example1.png')
cv2.imshow("org", img)

gas = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("gas", gas)

cv2.waitKey(0)
cv2.destroyAllWindows()