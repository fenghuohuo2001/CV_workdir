# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 7.图像中圆环区域展开成为矩形长条.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/18 13:35
@Function：
"""
import cv2
import numpy as np
from PIL import Image
import math

# 通过圆得到矩形
def get_huan_by_circle(img, circle_center, radius, radius_width):
    # 建立一张空白图像，高为radius_width（圆环厚度）， 宽为圆环周长
    black_img = np.zeros((radius_width, int(2 * radius * math.pi), 3), dtype='uint8')
    for row in range(0, black_img.shape[0]):
        for col in range(0, black_img.shape[1]):
            theta = math.pi * 2 / black_img.shape[1] * (col + 1)  # +origin_theta
            rho = radius - row - 1
            p_x = int(circle_center[0] + rho * math.sin(theta) + 0.5) - 1
            p_y = int(circle_center[1] - rho * math.cos(theta) + 0.5) - 1

            black_img[row, col, :] = img[p_y, p_x, :]
    return black_img


img = cv2.imread('circle.png')
img_rec = get_huan_by_circle(img, (153, 153), 153, 75)

cv2.imshow("rec", img_rec)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 通过矩形得到圆
# def get_circle_by_huan(img):
#     cv2.imshow('bk', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     h, w, _ = img.shape
#     radius = w / (math.pi * 2)
#     black_img = np.zeros((int(2 * radius) + 1, int(2 * radius) + 1, 3), dtype='uint8')
#     circle_center = ((int(2 * radius) + 1) // 2, (int(2 * radius) + 1) // 2)
#     for row in range(0, img.shape[0]):
#         for col in range(0, img.shape[1]):
#             rho = radius - row - 1
#             theta = (col + 1) * (math.pi * 2) / img.shape[1]  # +origin_theta
#             p_x = int(circle_center[0] + rho * math.sin(theta) + 0.5)
#             p_y = int(circle_center[1] - rho * math.cos(theta) - 0.5)
#             black_img[p_y, p_x, :] = img[row, col, :]
#         # print(col)
#     black_img = cv2.blur(black_img, (3, 3))
#     return black_img
#
#
# img = cv2.imread('bk.jpg')
# get_circle_by_huan(img)