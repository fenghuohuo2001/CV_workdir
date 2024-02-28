import cv2

# 创建VideoCapture对象，参数为摄像头索引号或视频文件路径
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920*2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取并显示摄像头图像，按下'q'键退出循环
while True:
    # 读取摄像头图像帧
    ret, frame = cap.read()


    # 检查是否成功读取图像帧
    if not ret:
        print("无法获取图像帧")
        break

    # 显示图像帧
    cv2.imshow("Camera", frame)

    # 检测按键事件，按下'q'键退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头资源
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()
