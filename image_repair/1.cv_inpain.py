"""
@Name: 1.cv_inpain.py
@Auth: Huohuo
@Date: 2023/3/8-18:47
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

img = cv2.imread("5903.jpg", 0)
mask = np.zeros((img.shape[0], img.shape[1]))   # mask 要对应需要修复区域

img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow("src.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
