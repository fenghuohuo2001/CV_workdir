"""
@Name: get_video_detectp.py
@Auth: Huohuo
@Date: 2023/5/17-10:46
@Desc: 
@Ver : 
code_idea
"""
import cv2
import time

# 加载ONNX模型
net = cv2.dnn.readNetFromONNX("yolo5-mobilenetv3-person.onnx")

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

    # 图像预处理
    resized_frame = cv2.resize(frame, (640, 640))
    blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)

    # 设置输入并进行推理
    net.setInput(blob)
    outputs = net.forward()

    # 后处理
    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            confidence = detection[4]
            if confidence > 0.5:  # 设置置信度阈值为0.5
                x, y, w, h = detection[:4]
                x1 = int((x - w / 2) * frame.shape[1])  # 转换回原始帧的坐标
                y1 = int((y - h / 2) * frame.shape[0])
                x2 = int((x + w / 2) * frame.shape[1])
                y2 = int((y + h / 2) * frame.shape[0])
                boxes.append([x1, y1, x2, y2])  # 将边界框添加到列表中
                confidences.append(float(confidence))

    # 应用非最大值抑制
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i[0]  # 提取索引
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 计算帧率
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示帧
    cv2.imshow("Frame", frame)

    # 按下 'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
