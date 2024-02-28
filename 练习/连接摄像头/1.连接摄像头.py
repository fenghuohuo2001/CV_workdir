# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.连接摄像头.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/21 20:10
@Function：
"""
import cv2

camera = cv2.VideoCapture(0)
ret, frame = camera.read()
cv2.imshow("vedio", frame)
cv2.waitKey(0)
cv2.destroyWindow("vedio")  # 注意 这里有点不一样