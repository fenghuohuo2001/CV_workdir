"""
@Name: 2_get_max_resolution.py
@Auth: Huohuo
@Date: 2023/11/30-8:36
@Desc: 
@Ver : 
code_idea
"""
import cv2

def get_max_resolution(camera_index):
    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)

    # 获取摄像头的最大分辨率
    max_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    max_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 释放摄像头资源
    cap.release()

    return max_width, max_height

# 摄像头索引，通常为0，1，2等
camera_index = 0

max_width, max_height = get_max_resolution(camera_index)

print(f"摄像头最大分辨率为 {max_width} x {max_height}")

