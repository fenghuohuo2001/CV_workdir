"""
@Name: 4.GradCut分割.py
@Auth: Huohuo
@Date: 2023/2/19-13:29
@Desc: 
@Ver : 
code_idea
"""
import os
import numpy as np
import cv2
import time

def gradcut(img_copy, rect):
    mask = np.zeros(img_copy.shape[:2], np.uint8)

    # 创建零填充的背景和前景模型
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 使用矩形分割
    cv2.grabCut(img_copy, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

    result = cv2.bitwise_and(img, img, mask=mask2)
    cv2.imshow("result", result)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()

data_path = 'D:\WorkDirectory\school_project\dragline_detection\pre_data'
class_name = ['dirt', 'normal', 'scratch']

rect = (170, 0, 372, 540)       # 中等范围`


# 遍历目录下图片文件
for filename in os.listdir(data_path):

    for img_path in os.listdir(data_path + '/' + filename):
        print(filename + '/' + img_path)
        time_start = time.time()
        img = cv2.imread(data_path + '/' + filename + '/' + img_path)

        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        img_copy = img.copy()
        gradcut(img_copy, rect)
        time_end = time.time()
        print("spend time : {}".format(time_end-time_start))
        print("end time : {}".format(time_end))


