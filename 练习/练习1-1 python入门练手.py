# -*- 练习1 python入门练手 -*-
"""
功能：字符串各位识别
作者：fenghuohuo
日期：2021年06月03日
"""
import cv2

img = cv2.imread('rabbit.png')

img_erode = cv2.erode(img,None,iterations=1)    #腐蚀
#iteration的值越高，模糊程度(腐蚀程度)就越高 呈正相关关系
img_blur = cv2.GaussianBlur(img_erode,(5,5),0)

cv2.imshow('erode',img_erode)
cv2.imshow('blur',img_blur)
cv2.imshow('normal',img)
cv2.waitKey(0)
cv2.destroyAllWindows()