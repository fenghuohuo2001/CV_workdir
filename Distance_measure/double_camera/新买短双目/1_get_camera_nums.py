"""
@Name: 1_get_camera_nums.py
@Auth: Huohuo
@Date: 2023/12/6-9:48
@Desc: 
@Ver : 
code_idea
"""
import cv2

for i in range(5):
    video = cv2.VideoCapture(i)
    ret, frame = video.read()
    if ret:
        print(i)
    else:
        break
