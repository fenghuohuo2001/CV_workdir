# -*- 练习31 -*-
"""
功能：求圆心坐标
作者：fenghuohuo
日期：2021年11月15日
初版 但是定位不准确 需要试验台支撑 取阈值

思路流程：
灰度化 -> 降噪 -> 二值化仅保留内轮廓 -> 开运算 -> 提取内轮廓 -> 在空白图片上画出内轮廓并保存
-> hough圆变换检测圆轮廓与圆心 -> 画出圆轮廓 -> 重复操作画出外部轮廓

问题：
cv2.findcontour中 需要寻找检索内外轮廓的参数

2021.11.22
使用sobel算子后bug报错，问题猜测：二值化部分阈值设置后 仅剩部分轮廓 导致读取轮廓时为空集

"""
import cv2
import numpy as np
# path = "D:\\1.Desktop file\\picture\\img.jpg"
path = "circle_3-1.png"
img_src = cv2.imread(path)    # 这里用的图片是对比度增强后的图片，原图片上有反光不便处理
height = img_src.shape[0]
width = img_src.shape[1]

img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", img_gray)


img_blur = cv2.GaussianBlur(img_gray, (3, 3), 5)
# cv2.namedWindow('blur', 0)
# cv2.imshow("blur", img_blur)



# ret, img_value = cv2.threshold(img_blur, 50, 150, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret, img_value = cv2.threshold(img_blur, 30, 90, cv2.THRESH_BINARY)
# cv2.namedWindow('value', 0)
# cv2.imshow("value", img_value)

img_erode = cv2.erode(img_value, (5, 5), iterations=10)
img_dilate = cv2.dilate(img_erode, (5, 5), iterations=10)
# cv2.namedWindow('open', 0)
# cv2.imshow("open", img_dilate)

# img_done = 255 - img_dilate
# cv2.namedWindow('done', 0)
# cv2.imshow("done", img_done)

contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 创建空白图片
img_write = np.zeros((width, height), np.uint8)
img_write.fill(255)

# cv2.drawContours(img_src, contours, -1, (0, 0, 255), 3)

cv2.drawContours(img_write, contours, -1, (0, 0, 255), 3)
cv2.imwrite("D:\\1.Desktop file\\picture\\circle.png", img_write)
# cv2.namedWindow("write", 0)
# cv2.imshow("write", img_write)

circles = cv2.HoughCircles(img_write, cv2.HOUGH_GRADIENT, 1, 100,
                           param1=100, param2=30, minRadius=0, maxRadius=1000)
circles = np.uint16(np.around(circles))     # 近似取整
print(circles)
# 画出轮廓
for i in circles[0, :]:      # 取所有列第0行元素 circles 是一个三维数组[[[]]],降维成1维
    print(i)
    # draw the outer circle
    cv2.circle(img_src, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img_src, (i[0], i[1]), 2, (0, 0, 255), 3)

'''重复外轮廓'''
# 也可以用循环写 但是没必要
ret1, img_value1 = cv2.threshold(img_blur, 230, 250, cv2.THRESH_BINARY)
# cv2.namedWindow('value1', 0)
# cv2.imshow("value1", img_value1)

img_erode1 = cv2.erode(img_value1, (5, 5), iterations=10)
img_dilate1 = cv2.dilate(img_erode1, (5, 5), iterations=10)
# cv2.namedWindow('open1', 0)
# cv2.imshow("open1", img_dilate1)

contours1, hierarchy1 = cv2.findContours(img_dilate1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img_write1 = np.zeros((width, height), np.uint8)
img_write1.fill(255)

cv2.drawContours(img_write1, contours1, -1, (0, 0, 255), 3)
# cv2.imwrite("D:\\1.Desktop file\\picture\\circle1.png", img_write1)

circles1 = cv2.HoughCircles(img_write1, cv2.HOUGH_GRADIENT, 1, 100,
                            param1=100, param2=30, minRadius=0, maxRadius=1000)
circles1 = np.uint16(np.around(circles1))     # 近似取整
print(circles1)
# 画出轮廓
for i in circles1[0, :]:      # 取所有列第0行元素 circles 是一个三维数组[[[]]],降维成1维
    print(i)
    # draw the outer circle
    cv2.circle(img_src, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img_src, (i[0], i[1]), 2, (0, 0, 255), 3)

Doc1 = circles[0, :]
doc1 = Doc1[0, :]
# print("doc1=", doc1)
# print("doc1[0]", doc1[0])
Doc2 = circles1[0, :]
doc2 = Doc2[0, :]

x1 = abs(float(doc1[0])-(doc2[0]))
y1 = abs(float(doc1[1])-(doc2[1]))

distance = (pow((pow(x1, 2) + pow(y1, 2)), 0.5))
# pow(a,b)是计算a的b次方的一个函数
print("圆心之间像素距离为", distance)

cv2.namedWindow('src', 0)
cv2.imshow("src", img_src)
cv2.waitKey(0)
cv2.destroyAllWindows()