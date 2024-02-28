import cv2

# 打开摄像头
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # 读取一帧
    ret, frame = cap.read()

    # 检查帧是否成功读取
    if not ret:
        print("Error: Could not read frame.")
        break

    # 展示实时画面
    cv2.imshow("Camera", frame)

    # 检测键盘输入
    key = cv2.waitKey(1) & 0xFF

    # 按下 'q' 键拍摄并保存图像
    if key == ord('q'):
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured successfully.")
        break

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
