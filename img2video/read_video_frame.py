"""
@Name: read_video_frame.py
@Auth: Huohuo
@Date: 2023/6/6-9:37
@Desc: 
@Ver : 
code_idea
"""
import cv2

# 打开视频文件
video_path = ""
cap = cv2.VideoCapture(video_path)

# 检查视频文件是否成功打开
if not cap.isOpened():
    print("error, cannot open the video file !!!")
    exit()

# 读取并显示视频帧
while True:
    # 逐帧读取视频
    ret, frame = cap.read()

    # 如果成功读取帧
    if ret:
        # 窗口显示
        cv2.imshow("Video Frame", frame)

        # 按q退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()

