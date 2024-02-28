"""
@Name: 2-videos_concat.py
@Auth: Huohuo
@Date: 2023/5/10-17:47
@Desc: 
@Ver : 
code_idea
"""
import cv2
import numpy as np

video1_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\dla34_culane\20230510_172856_lr_6e-04_b_1\result_dla34_culane.mp4"
video2_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\r18_culane\20230510_173825_lr_6e-04_b_1\result_r18_culane.mp4"
video3_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\r34_culane\20230510_173020_lr_6e-04_b_1\result_r34_culane.mp4"
video4_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\r101_culane\20230510_173150_lr_3e-04_b_1\result_r101_culane.mp4"

output_path = r"D:\WorkDirectory\mywork\CLRNet-main\CLRNet-main\work_dirs\clr\compared.mp4"

# 读取四个视频文件
video1 = cv2.VideoCapture(video1_path)
video2 = cv2.VideoCapture(video2_path)
video3 = cv2.VideoCapture(video3_path)
video4 = cv2.VideoCapture(video4_path)

# 获取视频的帧率和大小
fps = int(video1.get(cv2.CAP_PROP_FPS))
frame_size = (int(video1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 创建一个新视频并设置编解码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_size[0]*2, frame_size[1]*2))

# 循环读取四个视频的帧，将其缩放并放置在新视频中：
while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()
    ret3, frame3 = video3.read()
    ret4, frame4 = video4.read()
    if not ret1 or not ret2 or not ret3 or not ret4:
        break
    frame1 = cv2.resize(frame1, frame_size)
    frame1 = cv2.putText(frame1, "dla34", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame2 = cv2.resize(frame2, frame_size)
    frame2 = cv2.putText(frame2, "r18", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame3 = cv2.resize(frame3, frame_size)
    frame3 = cv2.putText(frame3, "r34", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame4 = cv2.resize(frame4, frame_size)
    frame4 = cv2.putText(frame4, "r101", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame_top = np.concatenate((frame1, frame2), axis=1)
    frame_bottom = np.concatenate((frame3, frame4), axis=1)
    frame_combined = np.concatenate((frame_top, frame_bottom), axis=0)
    out.write(frame_combined)

# 释放视频对象和关闭输出视频：
video1.release()
video2.release()
video3.release()
video4.release()
out.release()

