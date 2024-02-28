# -*- 练习24 -*-
"""
功能：查找物体轮廓
作者：fenghuohuo
日期：2021年6月22日
"""
import cv2

img = cv2.imread("xlb2.png")
img = cv2.GaussianBlur(img,(5,5),0)     #高斯滤波
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)   #调低了阈值，注意对照亮度调整
# 图像阈值处理：阈值的作用是根据设定的值处理图像的灰度值，
# 比如灰度大于某个数值像素点保留。
# 通过阈值以及有关算法可以实现从图像中抓取特定的图形，比如去除背景等
#cv2.threshold(src, thresh, maxval, type[, dst])，返回值为retval, dst
'''其中：
src是灰度图像
thresh是起始阈值
maxval是最大值
type是定义如何处理数据与阈值的关系。有以下几种：
选项	                    像素值>thresh	其他情况
cv2.THRESH_BINARY	    maxval	        0
cv2.THRESH_BINARY_INV	0	            maxval
cv2.THRESH_TRUNC	    thresh	        当前灰度值
cv2.THRESH_TOZERO	    当前灰度值	    0
cv2.THRESH_TOZERO_INV	0	            当前灰度值
'''#各参数含义

image,contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
'''
contours 中是三维坐标组

轮廓检测函数接受的参数为二值图，所以需要将图像先转换为灰度，再转化为二值图
cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]]) 
第一个参数是寻找轮廓的图像；
第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
 cv2.RETR_EXTERNAL  表示只检测外轮关系
 cv2.RETR_CCOMP     建立两个等级的轮廓，上廓
 cv2.RETR_LIST      检测的轮廓不建立等级面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
 cv2.RETR_TREE      建立一个等级树结构的轮廓。
第三个参数method为轮廓的近似办法
 cv2.CHAIN_APPROX_NONE      存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
 cv2.CHAIN_APPROX_SIMPLE    压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
 cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS 使用teh-Chinl chain 近似算法
'''

cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
'''
绘制轮廓，各参数含义：
第一个参数是指明在哪幅图像上绘制轮廓；
第二个参数是轮廓本身，在Python中是一个list。
第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。后面的参数很简单。其中thickness表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。绘制参数将在以后独立详细介绍。
'''

cv2.imshow("img", img)
cv2.waitKey(0)
'''
python3.7更新后会出现：
出现：ValueError: too many values to unpack
如果你仅仅使用一个变量a去接受返回值，调用len(a),你会发现长度为3，
也就是说这个函数实际上返回了三个值
第一个，也是最坑爹的一个，它返回了你所处理的图像
第二个，正是我们要找的，轮廓的点集
第三个，各层轮廓的索引
'''
print("contours:类型：", type(contours))
print("第0 个contours:", type(contours[0]))
print("contours 数量：", len(contours))

print("contours[0]点的个数：", len(contours[0]))
print("contours[1]点的个数：", len(contours[1]))
