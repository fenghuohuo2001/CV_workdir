import cv2
import numpy as np

def cat2images(limg, rimg):
    HEIGHT = limg.shape[0]
    WIDTH = limg.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH*2+20,3))
    imgcat[:,:WIDTH,:] = limg
    imgcat[:,-WIDTH:,:] = rimg
    for i in range(int(HEIGHT / 32)):
        imgcat[i*32,:,:] = 255
    return imgcat

left_image = cv2.imread("save_image/left_image/left_0.jpg")
right_image = cv2.imread("save_image/right_image/right_0.jpg")

imgcat_source = cat2images(left_image,right_image)
HEIGHT = left_image.shape[0]
WIDTH = left_image.shape[1]
cv2.imwrite('result/baseline_correct/imgcat_source.jpg', imgcat_source)

camera_matrix0 = np.array([[940.66941728,       0     , 661.44395517],
                           [      0     , 940.40065844, 636.75282],
                           [      0     ,       0     ,       1     ]]
                        ) .reshape((3,3)) #即上文标定得到的 cameraMatrix1

distortion0 = np.array([0.0803803, -0.02345474, -0.01409531, -0.00166995, -0.22859165]) #即上文标定得到的 distCoeffs1

camera_matrix1 = np.array([[1091.11026, 0, 1117.16592],
                           [0, 1090.53772, 633.28256],
                           [0, 0, 1]]
                        ).reshape((3,3)) #即上文标定得到的 cameraMatrix2
distortion1 = np.array([-2.20291244e-02, 6.07832397e-01, -1.36355978e-02, 7.28391607e-04, -1.27152050e+00]) #即上文标定得到的 distCoeffs2

R = np.array([[ 0.94967792, -0.02450897,  0.31226775],
              [ 0.04051368,  0.99817108, -0.04486792],
              [-0.31059698,  0.05526119,  0.94893399]]
            ) #即上文标定得到的 R
T = np.array([[-144.4902528], [12.92476715], [21.98931425]]) #即上文标定得到的T


(R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2) = \
    cv2.stereoRectify(camera_matrix0, distortion0, camera_matrix1, distortion1, np.array([WIDTH,HEIGHT]), R, T) #计算旋转矩阵和投影矩阵

(map1, map2) = \
    cv2.initUndistortRectifyMap(camera_matrix0, distortion0, R_l, P_l, np.array([WIDTH,HEIGHT]), cv2.CV_32FC1) #计算校正查找映射表

rect_left_image = cv2.remap(left_image, map1, map2, cv2.INTER_CUBIC) #重映射

#左右图需要分别计算校正查找映射表以及重映射
(map1, map2) = \
    cv2.initUndistortRectifyMap(camera_matrix1, distortion1, R_r, P_r, np.array([WIDTH,HEIGHT]), cv2.CV_32FC1)

rect_right_image = cv2.remap(right_image, map1, map2, cv2.INTER_CUBIC)

imgcat_out = cat2images(rect_left_image,rect_right_image)
cv2.imwrite('result/baseline_correct/imgcat_out.jpg', imgcat_out)
