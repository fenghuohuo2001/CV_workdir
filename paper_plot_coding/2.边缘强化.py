"""
@Name: 2.边缘强化.py
@Auth: Huohuo
@Date: 2023/3/13-15:00
@Desc: 
@Ver : 
code_idea
"""
import cv2
from utils import show

def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


img_path = "data/2.png"
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_sobel = sobel(img_gray)
img_canny = cv2.Canny(img_gray, 50, 150, apertureSize=3, L2gradient=False)
# 强化
img_add_sobel = cv2.addWeighted(img_gray, 0.5, img_sobel, 0.5, 0)
# img_add_canny = cv2.addWeighted(img_gray, 0.5, img_canny, 0.5, 0)

show = show()
# show.gray_pic_show(1, 3, 255-img_gray, 255-img_sobel, 255-img_add_sobel)
show.gray_pic_show(1, 2, 255-img_gray, 255-img_add_sobel)
cv2.imwrite("data/2_result.png", 255-img_add_sobel)