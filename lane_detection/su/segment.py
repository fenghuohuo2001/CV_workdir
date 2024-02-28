"""
@Name: segment.py
@Auth: Huohuo
@Date: 2023/6/11-9:59
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

# ------------------------------------
#               sobel算子
# ------------------------------------
def sobel(img, x_weight, y_weight):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, x_weight, absY, y_weight, 0)
    return dst

# ------------------------------------
#               形态学运算
# open = True:      开运算
# open = False:     闭运算
# ------------------------------------
def open_close(img, k_x=3, k_y=3, open=True):
    morph_kernel = np.ones((k_x, k_y), np.uint8)
    if open:
        img_erode = cv2.erode(img, morph_kernel, 1)
        img_dilate = cv2.dilate(img_erode, morph_kernel, 1)
        return img_dilate

    else:
        img_dilate = cv2.dilate(img, morph_kernel, 1)
        img_erode = cv2.erode(img_dilate, morph_kernel, 1)
        return img_erode

img_path = "data/1003.png"
img = cv2.imread(img_path, 0)
cv2.imshow("img_gray", img)

# roi区域手动提取
img_h, img_w = img.shape
print(img.shape)
# img = img[5:-5, 100:190]
cv2.imshow("img_cut", img)

# 二值化
thresh, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# canny
# img_blur = cv2.GaussianBlur(img_binary, (3, 3), 0)
# img_blur = cv2.medianBlur(img_binary, 3, 0)
img_blur = cv2.blur(img_binary, (3, 3))
cv2.imshow("img_blur", img_blur)

# open or close
img_open = open_close(img_blur)

# 边缘检测
# img_edge = cv2.Canny(img_blur, 50, 150)
img_edge = sobel(img_open, 0, 1)

cv2.imshow("img_edge", img_edge)

cv2.waitKey(0)
cv2.destroyAllWindows()