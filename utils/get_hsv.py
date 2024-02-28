"""
@Name: get_hsv_batch.py
@Auth: Huohuo
@Date: 2023/2/16-17:34
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# img_path = r"D:\WorkDirectory\school_project\dragline_detection\stay_cable_photo\J16\dirt\64388mm\left_back-2022-12-14-07-25-33-64388mm-65740mm.jpeg"
img_path = "../detection_dir/cut_area.jpg"
image = cv2.imread(img_path)
# image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
image = cv2.resize(image, (int(image.shape[1]), int(image.shape[0])))
#image=cv2.imread('szy.png')这是直接标定图片
HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
def getpos(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN: #定义一个鼠标左键按下去的事件
        print(HSV[y,x])

cv2.imshow("imageHSV",HSV)
# cv2.imshow('image',image)
cv2.setMouseCallback("imageHSV",getpos)
cv2.waitKey(0)


