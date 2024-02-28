"""
@Name: 4.建立rect.py
@Auth: Huohuo
@Date: 2023/2/19-13:35
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

img_path = r"D:\WorkDirectory\school_project\dragline_detection\stay_cable_photo\J16\dirt\64388mm\left_back-2022-12-14-07-25-33-64388mm-65740mm.jpeg"
# img_path = r"D:\WorkDirectory\cv_workdir\img_repain\data\crop_0.png"

img = cv2.imread(img_path)
# img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
r = cv2.selectROI('input', img, False)  # 返回 (x_min, y_min, w, h)
print("input:", r)

print(r[0])