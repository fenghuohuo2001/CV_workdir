"""
@Name: 4.rect_clc_batch.py
@Auth: Huohuo
@Date: 2023/2/19-13:59
@Desc:
@Ver :
code_idea
(170, 0, 372, 540)
"""
import os

import cv2
import numpy as np

data_path = 'D:\WorkDirectory\school_project\dragline_detection\pre_data'
class_name = ['dirt', 'normal', 'scratch']

# 遍历目录下图片文件
for filename in os.listdir(data_path):

    rect_data = []
    rect_xmin = []
    rect_ymin = []
    rect_w = []
    rect_h = []

    for img_path in os.listdir(data_path + '/' + filename):
        print(filename + '/' + img_path)
        img = cv2.imread(data_path + '/' + filename + '/' + img_path)

        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        r = cv2.selectROI('input', img, False)  # 返回 (x_min, y_min, w, h)
        print("input:", r)
        rect_data.append(r)

        rect_xmin.append(r[0])
        rect_ymin.append(r[1])
        rect_w.append(r[2])
        rect_h.append(r[3])
        print(rect_data)

        print("输出设定rect值")
        print((min(rect_xmin), min(rect_ymin), max(rect_w), max(rect_h)))









# def main():
#
#     return 0
#
# if __name__ == '__main__':
#     main()

