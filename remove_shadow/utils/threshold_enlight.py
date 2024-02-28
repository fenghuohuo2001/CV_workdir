"""
@Name: threshold_enlight.py
@Auth: Huohuo
@Date: 2023/7/4-15:21
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

# -------------------------------------------------
#               基于阈值的处理方法
#           设定亮度阈值  threshold = 30
# -------------------------------------------------
def thresh_light(img, threshold = 30):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 将低亮度区域置为高亮度
    highlights_image = np.where(gray_image < threshold, 40, gray_image)
    return highlights_image