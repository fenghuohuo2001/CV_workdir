# -*- 练习5-1 -*-
"""
功能：多阈值的图像二值化处理
作者：fenghuohuo
日期：2021年6月7日
"""
import cv2
img1 = cv2.imread('rabbit.png', cv2.IMREAD_GRAYSCALE)   # cv2.THRESH_OTSU   作为图像自适应二值化的一个很优的算法Otsu大津算法的参数
ret, binary = cv2.threshold(img1, 175, 255, cv2.THRESH_BINARY)  # cv2.THRESH_BINARY    大于阈值的部分被置为255，小于部分被置为0
ret, binaryinv = cv2.threshold(img1, 175, 255, cv2.THRESH_BINARY_INV)   # cv2.THRESH_BINARY_INV    大于阈值部分被置为0，小于部分被置为255
ret, trunc = cv2.threshold(img1, 175, 255, cv2.THRESH_TRUNC)    # cv2.THRESH_TRUNC     大于阈值部分被置为threshold，小于部分保持原样
ret, tozero = cv2.threshold(img1, 175, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_TOZERO   小于阈值部分被置为0，大于部分保持不变
ret, tozeroinv = cv2.threshold(img1, 175, 255, cv2.THRESH_TOZERO_INV)   # cv2.THRESH_TOZERO_INV    大于阈值部分被置为0，小于部分保持不变
while True:
    cv2.imshow('img1', img1)
    cv2.imshow('binary', binary)
    cv2.imshow('binaryinv', binaryinv)
    cv2.imshow('trunc', trunc)
    cv2.imshow('tozero', tozero)
    cv2.imshow('tozeroinv', tozeroinv)
    if cv2.waitKey(1) == 27:
        break