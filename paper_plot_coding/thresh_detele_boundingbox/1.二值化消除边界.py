"""
@Name: 1.二值化消除边界.py
@Auth: Huohuo
@Date: 2023/3/20-15:59
@Desc: 
@Ver : 
code_idea
"""

import cv2
import numpy as np

img_path = r"D:\WorkDirectory\mywork\ocr.pytorch-master\ocr.pytorch-master\train_code\train_crnn\rcnn_all_data\train_data\rcnn_train\3028_2_4.jpg"

img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("thresh", img_thresh)
print(img.shape[1])
k_size = int(img.shape[1]/6)
k = np.ones((k_size, k_size), np.uint8)
print(k_size)
img_erode = cv2.erode(img_thresh, k, 1)
cv2.imshow("erode", img_erode)
img_dilate = cv2.dilate(img_erode, k, 1)
cv2.imshow("dilate", img_dilate)
# 边界消除后 框选轮廓
# image, contours = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, image = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_board = np.zeros_like(img_dilate)
cv2.drawContours(img_board, contours, -1, (255, 255, 255), 3)
cv2.imshow("img_board", img_board)



cv2.waitKey(0)
cv2.destroyAllWindows()
