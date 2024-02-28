"""
@Name: grayscale_stretching.py
@Auth: Huohuo
@Date: 2023/7/4-15:24
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np


# ---------------------------------------------
#                  灰度拉伸
# ---------------------------------------------
def strech(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图像的最小灰度值和最大灰度值：
    min_value = np.min(gray_image)
    max_value = np.max(gray_image)

    # 灰度拉伸
    stretched_image = cv2.convertScaleAbs(gray_image, alpha=255 / (max_value - min_value),
                                          beta=-255 * min_value / (max_value - min_value))

    return stretched_image