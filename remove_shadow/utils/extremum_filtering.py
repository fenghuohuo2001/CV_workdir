"""
@Name: extremum_filtering.py
@Auth: Huohuo
@Date: 2023/7/4-14:04
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

# ------------------------------------------------------
# 最大最小值滤波：首先排序周围像素和中心像素值，
# 将中心像素值与最大和最小值比较，如果比最小值小，
# 则替换中心像素为最小值，如果比最大值大，则替换中心像素为最大值。
# -----------------------------------------------------
def max_filter(img, kernel_size=3, filter_type="max"):
    h, w = img.shape[0], img.shape[1]
    pad = kernel_size // 2
    out_img = img.copy()
    # pad_img用于计算
    pad_img = np.zeros((h + pad*2, w + pad*2, 3), dtype=np.uint8)
    pad_img[pad:pad+h, pad:pad+w, :] = img.copy()

    if filter_type == "max":
        '''
        x, y = (0,0) k=3
        pad_img[y:y+kernel_size, x:x+kernel_size] = [0:1, 0:1]
        '''
        for y in range(h):
            for x in range(w):
                out_img[y, x, :] = np.max(pad_img[y-kernel_size:y+kernel_size, x-kernel_size:x+kernel_size, :])

    elif filter_type == "min":
        for y in range(h):
            for x in range(w):
                out_img[y, x] = np.min(pad_img[y-kernel_size:y+kernel_size, x-kernel_size:x+kernel_size])

    return out_img

def max_filter_gray(img, kernel_size=3, filter_type="max"):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[0], img.shape[1]
    pad = kernel_size // 2
    out_img = img.copy()
    # pad_img用于计算
    pad_img = np.zeros((h + pad*2, w + pad*2), dtype=np.uint8)
    pad_img[pad:pad+h, pad:pad+w] = img.copy()

    if filter_type == "max":
        '''
        x, y = (0,0) k=3
        pad_img[y:y+kernel_size, x:x+kernel_size] = [0:1, 0:1]
        '''
        for y in range(h):
            for x in range(w):
                out_img[y, x] = np.max(pad_img[y:y+kernel_size, x:x+kernel_size])

    elif filter_type == "min":
        for y in range(h):
            for x in range(w):
                out_img[y, x] = np.min(pad_img[y:y+kernel_size, x:x+kernel_size])

    return out_img