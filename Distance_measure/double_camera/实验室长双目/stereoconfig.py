import numpy as np
# 双目相机参数


'''
stereoCalibrate : 

Camera matrix left : 

[[956.53784563   0.         635.7738819 ]
 [  0.         956.91996163 345.65884057]
 [  0.           0.           1.        ]]
distCoeffs left  : 

[[ 0.11156626 -0.29895051 -0.01469405 -0.01117116  0.47799518]]
cameraMatrix left : 

[[949.4282706    0.         608.86641232]
 [  0.         950.5964532  334.46265161]
 [  0.           0.           1.        ]]
distCoeffs left : 

[[ 0.03259011  0.0963601  -0.01839325 -0.01662552 -0.04383499]]
R : 

[[ 0.99939397  0.00162848  0.03477131]
 [-0.00213605  0.99989164  0.01456508]
 [-0.03474382 -0.01463052  0.99928915]]
T : 

[[-46.64882858]
 [  0.6594959 ]
 [  1.17342414]]
E : 

[[-2.04069186e-02 -1.18294576e+00  6.41936085e-01]
 [-4.48045459e-01 -6.80585915e-01  4.66564700e+01]
 [-5.59452191e-01 -4.66448478e+01 -7.02375359e-01]]
F : 

[[ 2.86982979e-08  1.66291512e-06 -1.45656744e-03]
 [ 6.29313077e-07  9.55551671e-07 -6.34146856e-02]
 [ 5.19016028e-04  6.09224592e-02  1.00000000e+00]]

进程已结束,退出代码0


'''
import cv2
class stereoCamera(object):
    def __init__(self):

        self.cam_matrix_left = np.array([[956.53784563, 0, 635.7738819],
                                         [0, 956.91996163, 345.65884057],
                                         [0, 0, 1]])
        self.cam_matrix_right = np.array([[949.4282706, 0, 608.86641232],
                                          [0, 950.5964532, 334.46265161],
                                          [0, 0, 1]])

        self.distortion_l = np.array([[0.11156626, -0.29895051, -0.01469405, -0.01117116, 0.47799518]])
        self.distortion_r = np.array([[0.03259011, 0.0963601, -0.01839325, -0.01662552, -0.04383499]])

        self.R = np.array([[ 0.99939397, 0.00162848, 0.03477131],
                            [-0.00213605, 0.99989164, 0.01456508],
                            [-0.03474382, -0.01463052, 0.99928915]])

        self.T = np.array([[-46.64882858], [0.6594959], [1.17342414]])
        self.baseline = 500         # 两个摄像头之间的距离
        self.size = (1280, 720)

param = stereoCamera()


# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(param.cam_matrix_left, param.distortion_l,
                                                                  param.cam_matrix_right, param.distortion_r, param.size, param.R,
                                                                  param.T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(param.cam_matrix_left, param.distortion_l, R1, P1, param.size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(param.cam_matrix_right, param.distortion_r, R2, P2, param.size, cv2.CV_16SC2)

print(left_map1, left_map2)
print(right_map1, right_map2)


if __name__ == "__main__":
    # 打开摄像头

    cam = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # 设置双目的宽度
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # 读取左右摄像头的帧
        ret, frame = cam.read()

        frame_left = frame[0:720, 0:1280]
        frame_right = frame[0:720, 1280:2560]

        # 如果成功读取帧，进行校正
        if ret:
            # 校正左右摄像头的图像
            corrected_frame_left = cv2.remap(frame_left, left_map1, left_map2, cv2.INTER_LINEAR)
            corrected_frame_right = cv2.remap(frame_right, right_map1, right_map2, cv2.INTER_LINEAR)

            # 水平拼接校正后的图像
            display_frame = np.hstack((corrected_frame_left, corrected_frame_right))

            # 展示校正后的图像
            cv2.imshow('Stereo Vision (Rectified)', display_frame)
            cv2.imshow('Stereo Vision (l)', corrected_frame_left)
            cv2.imshow('Stereo Vision (r)', corrected_frame_right)

        # 检查是否按下 'q' 键，如果是则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cam.release()

    # 关闭所有窗口
    cv2.destroyAllWindows()