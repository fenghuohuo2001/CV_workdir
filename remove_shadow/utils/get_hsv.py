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


import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pixel_value = hsv[y, x]
        print("HSV value at (x={}, y={}): {}".format(x, y, pixel_value))

# 加载图像
img_path = "../result/fft/gauss1d_color.jpg"
image = cv2.imread(img_path)

# 创建窗口并绑定鼠标回调函数
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    cv2.imshow('Image', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()



