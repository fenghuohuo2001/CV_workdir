"""
@Name: Change_Video_Size.py
@Auth: Huohuo
@Date: 2023/7/13-16:52
@Desc: 
@Ver : 
code_idea
"""

import cv2

# *** 打开视频文件 这部分写需要处理的视频路径 可以是绝对路径 如果在一个文件夹下，就将输入视频文件名改为以下内容 ***
video = cv2.VideoCapture('Original.mp4')

# 获取原始视频的宽度和高度
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# *** 设置新的分辨率 ***
new_width = 640
new_height = 360

# 创建VideoWriter对象，用于写入视频
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# 视频输出名称
FPS = 1         # *** 视频输出帧率 ***
output_video = cv2.VideoWriter('output_video.avi', fourcc, FPS, (new_width, new_height))

# 逐帧读取并修改分辨率后写入新视频
while True:
    ret, frame = video.read()
    # *** 将保存出的这个图像发给我 会存在程序所在的文件夹下 ***
    # cv2.imwrite("start.jpg", frame)
    if not ret:
        break

    # 修改分辨率
    # resized_frame = cv2.resize(frame, (new_width, new_height))
    # resized_frame = cv2.resize(frame, (new_width, new_height))
    # *** 截取区域 ***
    resized_frame = frame[60:421, :]
    # cv2.imwrite("start.jpg", frame)

    # 写入新视频
    output_video.write(resized_frame)

    # 显示新视频
    # cv2.imshow('Resized Video', resized_frame)
    # *** 按q提取终止 ***
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
video.release()
output_video.release()
cv2.destroyAllWindows()

