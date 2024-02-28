"""
@Name: main-remove_shadow.py
@Auth: Huohuo
@Date: 2023/7/4-13:57
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np
from remove_shadow.utils.extremum_filtering import *
from remove_shadow.utils.mouse_callback import *
from remove_shadow.utils.fft_filtering import *
from remove_shadow.utils.getgaborker import *
from remove_shadow.utils.threshold_enlight import *
from remove_shadow.utils.grayscale_stretching import *

# -------------------------------------------------
#                读入图片
# -------------------------------------------------
img_path = "./data/test.jpg"

img = cv2.imread(img_path)
# img = cv2.imread(img_path)[0:1080, 0:1960]
print(img.shape)
cv2.imshow('src', img)
# -------------------------------------------------
#               基于阈值的处理方法
#           这部分继续改进---将低于阈值的部分变成相邻高灰度值
# -------------------------------------------------
# img_result = thresh_light(img, 30)

# -------------------------------------------------
#               基于灰度拉伸的处理方法
# -------------------------------------------------
img_result = strech(img)

cv2.imshow("img_result", img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()


