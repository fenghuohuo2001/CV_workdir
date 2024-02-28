# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 9. 最小外接圆.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/18 14:00
@Function：
"""
import cv2

for i in range(nums):

    (x, y), radius = cv2.minEnclosingCircle(contours[i])
    center = (int(x), int(y))
    radius = int(radius)
    image = cv2.circle(original, center, radius, (255, 0, 0), 2)
    cv2.imwrite(r'C:\Users\Lenovo\Desktop\result.jpg', image)
    cv2.imshow("result", image)

cv2.waitKey()
