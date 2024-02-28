"""
@Name: ReadJson2Correct.py
@Auth: Huohuo
@Date: 2023/7/13-15:05
@Desc: 
@Ver : 
code_idea
"""
import json
import cv2
import numpy as np

camera = cv2.VideoCapture(1,cv2.CAP_DSHOW)
i = 0

while 1:
    (grabbed, img) = camera.read()
    h1, w1 = img.shape[0], img.shape[1]
    u, v = img.shape[:2]

    # 从 JSON 文件中读取参数
    # with open('camera_params-150.json', 'r') as f:
    with open('camera_params-170.json', 'r') as f:
        data = json.load(f)

    ret = data['ret']
    mtx = np.array(data['mtx'], np.float32)
    dist = np.array(data['dist'], np.float32)
    rvecs = np.array(data['rvecs'], np.float32)
    tvecs = np.array(data['tvecs'], np.float32)

    # 假设你已经有了 frame, u, v, w1, h1 的值

    # 计算 newcameramtx 和 roi
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))

    # 纠正畸变
    dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # 使用 initUndistortRectifyMap 和 remap 函数进行纠正畸变
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w1, h1), 5)
    dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # 裁剪图像，输出纠正畸变后的图片
    x, y, w1, h1 = roi
    dst1 = dst1[y:y + h1, x:x + w1]
    cv2.imshow('dst2', dst2)
    cv2.imshow('dst1', dst1)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q保存一张图片
        cv2.imwrite("result/frame.jpg", dst2)
        break

camera.release()
cv2.destroyAllWindows()