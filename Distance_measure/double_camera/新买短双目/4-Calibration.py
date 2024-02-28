import cv2
import os
import numpy as np

leftpath = 'save_image/left_image'
rightpath = 'save_image/right_image'
CHECKERBOARD = (7,10)  #棋盘格内角点数
square_size = (20,20)   #棋盘格大小，单位mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
imgpoints_l = []    #存放左图像坐标系下角点位置
imgpoints_r = []    #存放左图像坐标系下角点位置
objpoints = []   #存放世界坐标系下角点位置
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp[0,:,0] *= square_size[0]
objp[0,:,1] *= square_size[1]


for ii in os.listdir(leftpath):
    img_l = cv2.imread(os.path.join(leftpath,ii))
    gray_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD)   #检测棋盘格内角点

    if ret_l:
        print(ii)
        objpoints.append(objp)
        corners2_l = cv2.cornerSubPix(gray_l,corners_l,(11,11),(-1,-1),criteria)
        imgpoints_l.append(corners2_l)
        img = cv2.drawChessboardCorners(img_l, CHECKERBOARD, corners2_l, ret_l)
        cv2.imshow('ChessboardCornersimg', img_l)
        cv2.waitKey(1)
        #cv2.imwrite('./ChessboardCornersimg.jpg', img)

for ii in os.listdir(rightpath):
    img_r = cv2.imread(os.path.join(rightpath, ii))
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD)
    if ret_r:
        print(ii)
        # objpoints.append(objp)
        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        imgpoints_r.append(corners2_r)
        img = cv2.drawChessboardCorners(img_r, CHECKERBOARD, corners2_r,ret_r)
        cv2.imshow('ChessboardCornersimg', img_r)
        cv2.waitKey(1)
        # cv2.imwrite('./ChessboardCornersimg.jpg', img)


ret, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1],None,None)  #先分别做单目标定
ret, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1],None,None)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1])   #再做双目标定

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1,
                                                                  cameraMatrix2, distCoeffs2, gray_r.shape[::-1], R, T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray_r.shape[::-1], cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray_r.shape[::-1], cv2.CV_16SC2)
print(left_map1, left_map2)
print(right_map1, right_map2)

print("stereoCalibrate : \n")
print("Camera matrix left : \n")
print(cameraMatrix1)
print("distCoeffs left  : \n")
print(distCoeffs1)
print("cameraMatrix left : \n")
print(cameraMatrix2)
print("distCoeffs left : \n")
print(distCoeffs2)
print("R : \n")
print(R)
print("T : \n")
print(T)
print("E : \n")
print(E)
print("F : \n")
print(F)
cv2.destroyAllWindows()