# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年11月29日
"""

import cv2

dsize = (500, 400)

img = cv2.imread("rabbit.png")
dst = cv2.resize(img, dsize)

cv2.imshow("src", img)
cv2.imshow("dst", dst)

cv2.imwrite("rabbit_(500, 400).jpg", dst)


cv2.waitKey(0)
cv2.destroyAllWindows()