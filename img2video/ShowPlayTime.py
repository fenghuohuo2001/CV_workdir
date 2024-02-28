"""
@Name: ShowPlayTime.py
@Auth: Huohuo
@Date: 2023/7/14-8:53
@Desc: 
@Ver : 
code_idea
"""
import cv2

def get_current_duration(cap):
    # 获取视频的帧率和总帧数
    rate = cap.get(cv2.CAP_PROP_FPS)
    print(rate)
    frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # 获取当前帧索引
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    # 计算当前播放时长（单位：秒）
    current_duration = current_frame / rate

    # 转换为分钟和秒
    minutes = int(current_duration // 60)
    seconds = int(current_duration % 60)

    return minutes, seconds

# 打开视频文件
video_path = "output.mp4"
cap = cv2.VideoCapture(video_path)

rate = cap.get(cv2.CAP_PROP_FPS)

# 创建 VideoWriter 对象，用于保存带有播放时长的视频
output_path = "output_show_time.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = rate
out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


if cap.isOpened():
    while True:
        # 读取视频帧
        ret, frame = cap.read()

        if not ret:
            break

        # 获取当前播放时长
        minutes, seconds = get_current_duration(cap)
        seconds = seconds - 1
        # 在视频帧上绘制播放时长
        duration_text = f"Duration: {minutes:02d}:{seconds:02d}"
        print(duration_text)
        img_h, img_w = frame.shape[0], frame.shape[1]
        cv2.putText(frame, duration_text, (img_w-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # 显示当前帧
        cv2.imshow('Video', frame)
        # 将带有播放时长的帧写入输出视频文件
        out.write(frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
else:
    print("无法打开视频文件")

