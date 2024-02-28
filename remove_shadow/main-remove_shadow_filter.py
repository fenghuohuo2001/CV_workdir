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

# -------------------------------------------------
#                读入图片
# -------------------------------------------------
img_path = "./data/test.jpg"

img_gray = cv2.imread(img_path, 0)
cv2.imwrite("result/fft/img_gray.jpg", img_gray)

img = cv2.imread(img_path)
# img = cv2.imread(img_path)[0:1080, 0:1960]
print(img.shape)

# img = cv2.resize(img, (1920, 1080))
# cv2.imshow('src', img)
# -------------------------------------------------
#              确定阴影栅格间隙的像素大小
#  distance=15  ==>>  kernel_size=15//2=7
# -------------------------------------------------
# get_coord(img)


# img_blur = cv2.GaussianBlur(img, (31, 31), 0)
# img_blur = cv2.medianBlur(img, 31, 0)
# img_blur = cv2.blur(img, (23, 23))
# img_blur = max_filter_gray(img, kernel_size=9)

img_blur = fft_fiter_count(img)
# img_blur = fft_fiter_mid(img)

# gabor_kernel = getgaborkernel()
# img_blur = cv2.filter2D(img, cv2.CV_8UC3, gabor_kernel)




# cv2.imshow("img_blur", img_blur)

cv2.imwrite("result/fft/img_blur.jpg", img_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()


