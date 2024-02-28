"""
@Name: 1.traditional_process.py
@Auth: Huohuo
@Date: 2023/4/6-11:34
@Desc: 
@Ver : 
code_idea
"""
import cv2
import time
import numpy as np

# ----------对比度拉伸---------------
def contrast_enhancement(img):
    height = int(img.shape[0])
    width = int(img.shape[1])

    flat_gray = img.reshape(width * height).tolist()
    A = min(flat_gray)  # 最大灰度值
    B = max(flat_gray)  # 最小灰度值

    img = np.uint8(255 / (B - A) * (img - A) + 0.5)
    return img

# ----------sobel算子---------------
def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

# ----------读入图片----------------
time_start = time.time()
img = cv2.imread("../data/img.png")
# print(img.shape)
src = img[300:1000, 400:1100]

# src = img[300:1000, 1400:2000]
cv2.imwrite("../data/img_cut.png", src)
# ---------图像预处理---------------
img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
img_gray = contrast_enhancement(img_gray)

# ---------滤波--------------------
# 定义的滤波算子大小应该能自适应
# 先用长宽中的最小边/1000测试
# 保证卷积核尺寸为奇数
# 暂时用高斯滤波
# ‘高频噪声下的螺栓表面缺陷检测_严琴’ 文章中说用中值滤波效果好
# ---------------------------------
kernel_size = int(min(img_gray.shape)/400)
# print("img_gray.shape", img_gray.shape)
# 保证卷积核尺寸为奇数
if kernel_size < 5:
    kernel_size = 5
if kernel_size % 2 == 1:
    pass
else:
    kernel_size += 1
# print("kernel_size", kernel_size)
img_filter = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)

# ----------二值化------------------
# 先尝试形态学处理
# 后续向连通域滤波方向拓展
# ---------------------------------
# img_atresh = cv2.adaptiveThreshold(img_filter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernel_size*9, 2)
# img_atresh_INV = cv2.adaptiveThreshold(img_filter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, kernel_size*9, 2)
# cv2.imwrite("../data/img_atresh_gas.png", img_atresh)
# cv2.imwrite("../data/img_atresh_INV_gas.png", img_atresh_INV)

# ---------形态学处理---------------
# morph_kernel = np.ones((kernel_size, kernel_size), np.uint8)
# img_erode = cv2.erode(img_atresh, morph_kernel, 1)
# cv2.imwrite("../data/img_erode_gas.png", img_erode)
# img_dilate = cv2.dilate(img_atresh, morph_kernel, 1)
# cv2.imwrite("../data/img_dilate_gas.png", img_dilate)
# img_open = cv2.morphologyEx(img_atresh, cv2.MORPH_OPEN, morph_kernel, 1)
# cv2.imwrite("../data/img_open_gas.png", img_open)

# --------连通域滤波-----------------
# 需要限制连通区域条件，将面积小的部分删除
# 面积用统计像素点个数来表示
# stats:每一个标记的统计信息，是一个5列的矩阵，每一行对应每个连通区域的外接矩形的x、y、width、height和面积
# ---------------------------------
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_atresh, connectivity=8, ltype=None)
#
# # 效果展示
# output = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
# for i in range(1, num_labels):
#     mask = (labels == i)      # 生成掩膜
#     # print(np.sum(mask))
#     if np.sum(mask) > 100:
#         output[:, :, 0][mask] = np.random.randint(0, 255)
#         output[:, :, 1][mask] = np.random.randint(0, 255)
#         output[:, :, 2][mask] = np.random.randint(0, 255)
# cv2.imwrite("../data/img_output_gas.png", output)
# ---------------------------------
#       边缘提取
# 根据实际处理效果，
# 1.可向harris角点检测方向拓展
# 2.Hough圆检测
# ---------------------------------
# img_sobel = sobel(img_filter)
img_canny = cv2.Canny(img_filter, 0, 255)
# cv2.imwrite("../data/img_sobel_gas.png", img_sobel)
cv2.imwrite("../data/img_canny_gas.png", img_canny)

# -------Hough圆检测---------------
circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT, 1, 100,
                            param1=100, param2=30, minRadius=30, maxRadius=50)
circles = np.uint16(np.around(circles))
img_white = np.zeros((src.shape[1], src.shape[0], 3), np.uint8)
img_white.fill(255)
for i in circles[0, :]:      # 取所有列第0行元素 circles 是一个三维数组[[[]]],降维成1维
    # print("i", i)
    # draw the outer circle
    cv2.circle(img_white, (i[0], i[1]), i[2], (0, 0, 255), 1)
    cv2.circle(src, (i[0], i[1]), i[2], (0, 0, 255), 1)
    print("圆心坐标为：", (i[0], i[1]))
    print("圆半径为：", i[2])
    # draw the center of the circle
    cv2.circle(img_white, (i[0], i[1]), 5, (0, 0, 255), 10)
    cv2.circle(src, (i[0], i[1]), 5, (0, 0, 255), 10)
cv2.imwrite("../data/img_white.png", img_white)
cv2.imwrite("../data/img_result.png", src)

# --------统计运行时间------------------
time_end = time.time()
use_time = time_end - time_start
print("using time:{:5f}s".format(use_time))