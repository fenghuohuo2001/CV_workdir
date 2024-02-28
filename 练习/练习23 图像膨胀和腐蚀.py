# -*- 练习23 -*-
"""
功能：图像膨胀和腐蚀 图像腐蚀 加上高斯模糊 就可以使得图像的色彩更加突出
作者：fenghuohuo
日期：2021年6月22日
"""
import cv2

img = cv2.imread('1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_value = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('value', img_value)
img_dilate = cv2.dilate(img_value, None, iterations=5)  # 膨胀
img_erode = cv2.erode(img, None, iterations=5)    # 腐蚀
# iteration的值越高，模糊程度(腐蚀程度)就越高 呈正相关关系

cv2.imshow('dilate', img_dilate)
cv2.imshow('erode', img_erode)
cv2.imshow('normal', img)
cv2.waitKey(0)
cv2.destroyAllWindows()