"""
@Name: click_pic_get_gray.py
@Auth: Huohuo
@Date: 2023/7/4-15:18
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        gray_value = gray_image[y, x]
        print("灰度值：", gray_value)


image = cv2.imread("../data/test.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
