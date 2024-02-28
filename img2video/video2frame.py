# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 1.video2frame.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/9/8 18:42
@Function：
"""
import json
import os

import cv2
import numpy as np

correct_json_path = "D:\WorkDirectory\mywork\yolov5-master-person\deploy\source\camera_params-150.json"

def load_camera_parameters(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    ret = data['ret']
    mtx = np.array(data['mtx'], np.float32)
    dist = np.array(data['dist'], np.float32)
    rvecs = np.array(data['rvecs'], np.float32)
    tvecs = np.array(data['tvecs'], np.float32)
    return mtx, dist

def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y + h, x:x + w]
    return undistorted_img

def video2frame(videos_path, frames_save_path, time_interval):
    vidcap = cv2.VideoCapture(videos_path)
    '''
    .read()按帧读取视频; ret,frame是获.read()方法的两个返回值
    如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False
    frame就是每一帧的图像，是个三维矩阵
    '''
    success, image = vidcap.read()

    count = 0
    while success:
        success, image = vidcap.read()
        image = cv2.resize(image, (640, 480))
        mtx, dist = load_camera_parameters(correct_json_path)
        image = undistort_image(image, mtx, dist)


        count += 1
        if count % time_interval == 0:
            cv2.imencode('.png', image)[1].tofile(frames_save_path + '/' + "avi2_%d.png" % count)
            '''
            ret, img = cv2.imencode()
            img is in ndarray format
            '''

            print("now imwrite", count)
        # if count == 20:
            # break
    print("the number of pic is", count)

if __name__ == '__main__':
    # 文件夹路径
    folder_path = r'E:\datasets\watercar'

    # 文件名称
    dir_name = "7-10 - Trim2.avi"

    videos_path = os.path.join(folder_path, dir_name)
    print(videos_path)

    frames_save_path = r"E:\datasets\watercar\pic/7-10-2"
    if not os.path.exists(frames_save_path):
        os.makedirs(frames_save_path)
    time_interval = 5  # 隔一帧保存一次
    video2frame(videos_path, frames_save_path, time_interval)
