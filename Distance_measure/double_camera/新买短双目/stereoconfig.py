import numpy as np
# 双目相机参数


'''
Camera matrix left : 

[[681.48496387   0.         598.56318032]
 [  0.         682.34219262 332.36849001]
 [  0.           0.           1.        ]]
distCoeffs left  : 

[[-0.06950884  0.01206451 -0.00471395 -0.00494272 -0.00078401]]
cameraMatrix left : 

[[688.30369672   0.         603.78120526]
 [  0.         690.39903009 338.53737437]
 [  0.           0.           1.        ]]
distCoeffs left : 

[[-0.0433816  -0.02993394 -0.00425043 -0.00447005  0.01139159]]
R : 

[[ 9.99987304e-01  1.83950472e-03 -4.69120141e-03]
 [-1.83841327e-03  9.99998282e-01  2.36959950e-04]
 [ 4.69162924e-03 -2.28332575e-04  9.99988968e-01]]
T : 

[[-59.12173299]
 [ -0.11599802]
 [  5.70172853]]
E : 

[[ 9.93791370e-03 -5.70169225e+00 -1.17347817e-01]
 [ 5.97903339e+00 -3.01106100e-03  5.90943328e+01]
 [ 2.24686721e-01 -5.91214180e+01 -1.45536530e-02]]
F : 

[[-1.49092518e-07  8.54315824e-05 -2.71057715e-02]
 [-8.94275917e-05  4.49794521e-08 -5.48828223e-01]
 [ 2.80444403e-02  5.58135680e-01  1.00000000e+00]]

进程已结束,退出代码0



'''
import cv2
class stereoCamera(object):
    def __init__(self):

        self.cam_matrix_left = np.array([[681.48496387, 0, 598.56318032],
                                         [0, 682.34219262, 332.36849001],
                                         [0, 0, 1]])
        self.cam_matrix_right = np.array([[688.30369672, 0, 603.78120526],
                                          [0, 690.39903009, 338.53737437],
                                          [0, 0, 1]])

        self.distortion_l = np.array([[-0.06950884, 0.01206451, -0.00471395, -0.00494272, -0.00078401]])
        self.distortion_r = np.array([[-0.0433816, -0.02993394, -0.00425043, -0.00447005, 0.01139159]])

        self.R = np.array([[9.99987304e-01, 1.83950472e-03, -4.69120141e-03],
                            [-1.83841327e-03, 9.99998282e-01, 2.36959950e-04],
                            [4.69162924e-03, -2.28332575e-04, 9.99988968e-01]])

        self.T = np.array([[-59.12173299], [-0.11599802], [5.70172853]])
        self.baseline = 610         # 两个摄像头之间的距离
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

    cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
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