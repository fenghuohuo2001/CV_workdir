# -*- 2 -*-
"""
功能：颜色规范化，最大值灰度处理
作者：fenghuohuo
日期：2021年11月8日
功能已经可以使用
"""
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# path = "D:\\1.Desktop file\\picture\\img.jpg"
path = "example5.jpg"
img = cv2.imread(path)
height = img.shape[0]
width = img.shape[1]
loop_1 = int(height/2)
loop_2 = int(height)
diff = loop_2 - loop_1

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray", img_gray)

img_half = img_gray[loop_1:loop_2, 0:width]
# cv2.imshow("half", img_half)

flat_gray = img_half.reshape(width * diff).tolist()
A = min(flat_gray)  # 最大灰度值
B = max(flat_gray)  # 最小灰度值

for i in range(diff):
    for j in range(width):
        img_gray[i+loop_1, j] = 255/(B - A)*(img_half[i, j] - A)+0.5
        img_gray[i + loop_1, j] = 255 / (B - A) * (img_half[i, j] - A) + 0.5

cv2.imshow("src", img)
cv2.imshow("result", img_gray)

name1 = "D:\\1.Desktop file\\picture\\result.png"
# name2 = "D:\\1.Desktop file\\picture\\src1.png"
cv2.imwrite(name1, img_gray)    # imwrite中第一个参数是文件名，可以加上存放路径
# cv2.imwrite(name2, img)

# cv2.imwrite("2.png", img)     # 也可以直接保存在当前文件夹内

cv2.waitKey(0)
cv2.destroyAllWindows()