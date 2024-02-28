"""
@Name: 1_read_cap.py
@Auth: Huohuo
@Date: 2023/11/30-8:30
@Desc: 
@Ver : 
code_idea
"""

# -*- coding: utf-8 -*-
import cv2

cv2.namedWindow("left")
cv2.namedWindow("right")
# camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
camera = cv2.VideoCapture(1)

# 设置分辨率 左右摄像机同一频率，同一设备ID；左右摄像机总分辨率1280x480；分割为两个640x480、640x480
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0

while True:
    ret, frame = camera.read()
    # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
    left_frame = frame[0:720, 0:1280]
    right_frame = frame[0:720, 1280:2560]

    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)


    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("save_image/left_image/left_" + str(counter) + ".jpg", left_frame)
        cv2.imwrite("save_image/right_image/right_" + str(counter) + ".jpg", right_frame)
        counter += 1
camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")
