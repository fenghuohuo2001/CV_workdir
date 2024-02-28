# -*- 练习 -*-
"""
功能：
作者：fenghuohuo
日期：2021年11月29日
"""
#coding:utf-8
import cv2 as cv

# 读取原始图像
img= cv.imread('img/faces3.jpg')
#face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_detect = cv.CascadeClassifier("lbpcascade_frontalface_improved.xml")
# 检测人脸
# 灰度处理
gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)

# 检查人脸 按照1.1倍放到 周围最小像素为5
face_zone = face_detect.detectMultiScale(gray, scaleFactor = 2, minNeighbors = 2) # maxSize = (55,55)
print ('识别人脸的信息：\n',face_zone)

# 绘制矩形和圆形检测人脸
for x, y, w, h in face_zone:
    # 绘制矩形人脸区域
    cv.rectangle(img, pt1 = (x, y), pt2 = (x+w, y+h), color = [0,0,255], thickness=2)
    # 绘制圆形人脸区域 radius表示半径
    cv.circle(img, center = (x + w//2, y + h//2), radius = w//2, color = [0,255,0], thickness = 2)

# 设置图片可以手动调节大小
cv.namedWindow("Easmount-CSDN", 0)

# 显示图片
cv.imshow("Easmount-CSDN", img)

# 等待显示 设置任意键退出程序
cv.waitKey(0)
cv.destroyAllWindows()
