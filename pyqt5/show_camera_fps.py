import cv2
import time

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头，如果有多个摄像头可以尝试不同的索引

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 帧率计算相关变量
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("无法读取摄像头图像")
        break

    # 水平翻转图像
    frame = cv2.flip(frame, 1)

    # 计算帧率
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示帧
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
