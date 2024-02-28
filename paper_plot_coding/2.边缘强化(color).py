"""
@Name: 2.边缘强化.py
@Auth: Huohuo
@Date: 2023/3/13-15:00
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

from utils import show

def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


def stress(img):
    img_sobel = sobel(img)
    # img_sobel = cv2.cvtColor(img_sobel, cv2.COLOR_BGR2GRAY)
    # print(img_sobel.shape)
    # img_sobel = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # img_sobel = cv2.cvtColor(img_sobel, cv2.COLOR_GRAY2BGR)
    # img_add_sobel = np.zeros_like(img)
    # for i in range(3):
    #     img_add_sobel[:, :, i] = cv2.addWeighted(img[:, :, i], 0.5, img_sobel, 0.5, 0)
    img_add_sobel = cv2.addWeighted(img, 0.5, img_sobel, 0.5, 0)

    # cv2.imshow("img_sobel",img_sobel)
    # cv2.imshow("img_add_sobel",img_add_sobel)
    # cv2.waitKey(0)
    return img_add_sobel

img_path = "data/1.png"
img = cv2.imread(img_path)
board = stress(img)
show = show()
show.color_pic_show(1, 2, img, board)
# show.gray_pic_show(1, 2, img, board)
# board = np.zeros(img.shape)
# for i in range(img.shape[2]):
# #     board[:, :, i] = stress(img[:, :, i])
#     board[:, :, i] = (board[:, :, i] - np.min(board[:, :, i])) / (np.max(board[:, :, i]) - np.min(board[:, :, i])) * 255
# board = (board - np.min(board)) / (np.max(board) - np.min(board)) * 255

# board = (board - np.min(board)) / (np.max(board) - np.min(board)) * 255
# 对比度增强
def contrast_enhancement(img):
    height = int(img.shape[0])
    width = int(img.shape[1])

    flat_gray = img.reshape(width * height).tolist()
    A = min(flat_gray)  # 最大灰度值
    B = max(flat_gray)  # 最小灰度值

    img = np.uint8(255 / (B - A) * (img - A) + 0.5)
    return img

# board = contrast_enhancement(board)
# show.color_pic_show(1, 2, img, board)
# show.gray_pic_show(1, 2, img, board)

# show.gray_pic_show(1, 2, 255-img, 255-board)
# cv2.imwrite("data/2_result.png", 255-board)